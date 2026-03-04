# rag/rag_agent.py — Agentic RAG: retrieval-augmented generation over social + news data
# LLM: open-source LLaMA 3.3 via Groq

import streamlit as st
from phi.agent.agent import Agent
from phi.model.groq import Groq
from phi.tools import Toolkit

import config
from rag.document_store import DocumentStore


class SentimentRetrieverTool(Toolkit):
    """Custom Phi tool that retrieves relevant documents from the vector store."""

    def __init__(self, doc_store: DocumentStore):
        super().__init__(name="sentiment_retriever")
        self.doc_store = doc_store
        self.register(self.search_sentiment)

    def search_sentiment(self, query: str) -> str:
        """Search the sentiment document store for posts related to the query.

        Args:
            query: Natural language query about stock sentiment, e.g.
                   'What is the public mood around Tesla?'

        Returns:
            Formatted context string with relevant posts and their sentiment scores.
        """
        results = self.doc_store.search(query, top_k=5)
        if not results:
            return "No relevant social media posts found for this query."

        context_parts = []
        for i, doc in enumerate(results, 1):
            source = doc.get("source", "unknown")
            score = doc.get("combined_score", doc.get("sentiment_score", "N/A"))
            label = doc.get("sentiment_label", "unknown")
            timestamp = doc.get("timestamp", "unknown")
            text = doc.get("text", "")[:300]

            context_parts.append(
                f"[{i}] Source: {source} | Sentiment: {label} ({score:+.2f}) | Time: {timestamp}\n"
                f"    {text}\n"
            )

        header = f"Found {len(results)} relevant posts (out of {self.doc_store.size} total):\n\n"
        return header + "\n".join(context_parts)

from typing import Optional

def create_rag_agent(doc_store: DocumentStore) -> Optional[Agent]:
    """Create a RAG-enabled agent that can answer questions about stock sentiment.

    Uses the open-source LLaMA 3.3 70B model via Groq for generation,
    and retrieves context from the in-memory document store.
    """
    api_key = config.get_groq_api_key()
    if not api_key:
        return None

    try:
        retriever_tool = SentimentRetrieverTool(doc_store)

        agent = Agent(
            name="Sentiment RAG Agent",
            role="Answer questions about public sentiment and social media discussions around stocks",
            model=Groq(id=config.LLM_MODEL_ID, api_key=api_key),
            tools=[retriever_tool],
            instructions=[
                "You are a financial sentiment analyst. Use the sentiment_retriever tool to find relevant social media posts.",
                "Always search for posts before answering any sentiment-related question.",
                "Summarize the overall mood, highlight key themes, and note any extreme opinions.",
                "Cite specific posts by their source and timestamp when relevant.",
                "Structure your response as: Overall Mood → Key Themes → Notable Posts → Risk Signals.",
                "If sentiment data is limited, say so clearly rather than guessing.",
            ],
            show_tool_calls=True,
            markdown=True,
        )
        return agent

    except Exception as e:
        st.warning(f"Could not create RAG agent: {e}")
        return None


def query_rag(doc_store: DocumentStore, question: str) -> str:
    """Run a RAG query and return the response as a string.

    This is a convenience wrapper for programmatic use.
    """
    agent = create_rag_agent(doc_store)
    if agent is None:
        return "RAG agent could not be initialized. Check API key configuration."

    try:
        response = agent.run(question)
        return response.content if hasattr(response, "content") else str(response)
    except Exception as e:
        return f"RAG query failed: {e}"
