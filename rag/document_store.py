# rag/document_store.py — In-memory vector store for social + news documents
# Uses open-source sentence-transformers (all-MiniLM-L6-v2) + numpy cosine similarity

import streamlit as st
import numpy as np
from datetime import datetime
from config import EMBEDDING_MODEL


@st.cache_resource(show_spinner=False)
def _load_embedding_model():
    """Load the sentence-transformers model (open-source, Apache 2.0).

    Model: all-MiniLM-L6-v2 — 80MB, 384-dim embeddings, very fast.
    """
    try:
        from sentence_transformers import SentenceTransformer
        return SentenceTransformer(EMBEDDING_MODEL)
    except ImportError:
        st.warning("📦 `sentence-transformers` not installed. Run `pip install sentence-transformers`.")
        return None
    except Exception as e:
        st.warning(f"Could not load embedding model: {e}")
        return None


class DocumentStore:
    """In-memory vector database using sentence-transformers."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model_name = model_name
        self.encoder = None
        self.documents: list[dict] = []
        self.embeddings: Optional[np.ndarray] = None
        self._model = None

    def _get_model(self):
        if self._model is None:
            self._model = _load_embedding_model()
        return self._model

    def add_documents(self, docs: list[dict]):
        """Add documents and compute their embeddings."""
        if not docs:
            return

        model = self._get_model()
        if model is None:
            self.documents.extend(docs)
            return

        texts = [d.get("text", "") for d in docs]
        new_embeddings = model.encode(texts, show_progress_bar=False, normalize_embeddings=True)

        if self.embeddings is None:
            self.embeddings = new_embeddings
        else:
            self.embeddings = np.vstack([self.embeddings, new_embeddings])

        self.documents.extend(docs)

    def search(self, query: str, top_k: int = 10) -> list[dict]:
        """Find the top-k most relevant documents for a query."""
        if not self.documents:
            return []

        model = self._get_model()
        if model is None or self.embeddings is None:
            # Fallback: keyword search
            return self._keyword_search(query, top_k)

        query_embedding = model.encode([query], normalize_embeddings=True)
        scores = np.dot(self.embeddings, query_embedding.T).flatten()
        top_indices = np.argsort(scores)[::-1][:top_k]

        results = []
        for idx in top_indices:
            doc = self.documents[idx].copy()
            doc["relevance_score"] = float(scores[idx])
            results.append(doc)

        return results

    def _keyword_search(self, query: str, top_k: int) -> list[dict]:
        """Simple keyword-based fallback search."""
        query_words = set(query.lower().split())
        scored = []
        for doc in self.documents:
            text_lower = doc.get("text", "").lower()
            matches = sum(1 for w in query_words if w in text_lower)
            if matches > 0:
                scored.append((matches, doc))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [d for _, d in scored[:top_k]]

    def clear(self):
        """Clear all documents and embeddings."""
        self.documents = []
        self.embeddings = None

    @property
    def size(self) -> int:
        return len(self.documents)
