# config.py — Centralized configuration for DALAAL AI

import streamlit as st
import os

# ──────────────────────────── API Keys ────────────────────────────
def get_groq_api_key():
    """Get Groq API key from Streamlit secrets or env."""
    try:
        return st.secrets["GROQ_API_KEY"]
    except Exception:
        return os.environ.get("GROQ_API_KEY", "")


def get_reddit_credentials():
    """Get Reddit API credentials from Streamlit secrets or env."""
    try:
        return {
            "client_id": st.secrets.get("REDDIT_CLIENT_ID", os.environ.get("REDDIT_CLIENT_ID", "")),
            "client_secret": st.secrets.get("REDDIT_CLIENT_SECRET", os.environ.get("REDDIT_CLIENT_SECRET", "")),
            "user_agent": st.secrets.get("REDDIT_USER_AGENT", os.environ.get("REDDIT_USER_AGENT", "dalaal-ai/1.0")),
        }
    except Exception:
        return {
            "client_id": os.environ.get("REDDIT_CLIENT_ID", ""),
            "client_secret": os.environ.get("REDDIT_CLIENT_SECRET", ""),
            "user_agent": os.environ.get("REDDIT_USER_AGENT", "dalaal-ai/1.0"),
        }


# ──────────────────────────── LLM Config ────────────────────────────
# Using open-source LLaMA 3 via Groq (free tier)
LLM_MODEL_ID = "llama-3.3-70b-versatile"

# ──────────────────────────── Sentiment Models ────────────────────────────
# All open-source HuggingFace models
FINBERT_MODEL = "ProsusAI/finbert"                          # Finance-specific sentiment
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"  # Lightweight embeddings for RAG

# ──────────────────────────── Stock Mappings ────────────────────────────
COMMON_STOCKS = {
    # US Stocks
    "NVIDIA": "NVDA", "APPLE": "AAPL", "GOOGLE": "GOOGL",
    "MICROSOFT": "MSFT", "TESLA": "TSLA", "AMAZON": "AMZN",
    "META": "META", "NETFLIX": "NFLX", "AMD": "AMD",
    "PALANTIR": "PLTR", "COINBASE": "COIN", "SNOWFLAKE": "SNOW",
    # Indian Stocks — NSE
    "TCS": "TCS.NS", "RELIANCE": "RELIANCE.NS", "INFOSYS": "INFY.NS",
    "WIPRO": "WIPRO.NS", "HDFC": "HDFCBANK.NS", "TATAMOTORS": "TATAMOTORS.NS",
    "ICICIBANK": "ICICIBANK.NS", "SBIN": "SBIN.NS", "MARUTI": "MARUTI.NS",
    "BHARTIARTL": "BHARTIARTL.NS", "HCLTECH": "HCLTECH.NS", "ITC": "ITC.NS",
    "AXISBANK": "AXISBANK.NS", "ADANI": "ADANIENT.NS", "BAJFINANCE": "BAJFINANCE.NS",
    "LT": "LT.NS", "SUNPHARMA": "SUNPHARMA.NS", "TITAN": "TITAN.NS",
}

PERIOD_MAP = {
    "1 Week": "5d",
    "1 Month": "1mo",
    "3 Months": "3mo",
    "6 Months": "6mo",
    "1 Year": "1y",
    "2 Years": "2y",
    "5 Years": "5y",
}

# ──────────────────────────── Reddit Config ────────────────────────────
SUBREDDITS = [
    "wallstreetbets", "stocks", "investing",
    "IndianStockMarket", "IndianStreetBets",
]
REDDIT_POST_LIMIT = 50

# ──────────────────────────── UI Config ────────────────────────────
APP_TITLE = "DALAAL AI — Smart Stock Intelligence"
APP_ICON = "📈"
