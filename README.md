# DALAAL AI — Smart Stock Intelligence 📈

> AI-powered stock market dashboard with real-time data, social sentiment analysis, and agentic RAG — built entirely on **open-source models**.

## ✨ Features

| Feature | Tech |
|---------|------|
| 📊 Real-time stock data | yfinance (US + Indian markets) |
| 📈 Interactive charts | Plotly (candlestick, volume, RSI, MACD, Bollinger) |
| 🤖 AI analysis | LLaMA 3.3 70B via Groq (open-source) |
| 🧠 Social sentiment | VADER + FinBERT (ProsusAI, open-source) |
| 📡 Social scraping | Reddit (PRAW) + Twitter/X (snscrape, open-source) |
| 💬 Agentic RAG | Sentence-transformers + LLaMA for grounded Q&A |
| 📥 Export | PDF (fpdf2) + CSV reports |
| 🌙 Dark mode | Full theme toggle |

## 🚀 Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Set your API keys
# Option 1: Streamlit secrets (.streamlit/secrets.toml)
# GROQ_API_KEY = "your-groq-api-key"
# REDDIT_CLIENT_ID = "your-reddit-client-id"
# REDDIT_CLIENT_SECRET = "your-reddit-client-secret"

# Option 2: Environment variables
export GROQ_API_KEY="your-groq-api-key"

# Run
streamlit run app.py
```

## 🏗️ Architecture

```
dalaal-ai/
├── app.py                  # Main Streamlit UI (orchestration)
├── config.py               # Centralized config & constants
├── data/
│   └── market_data.py      # yfinance data layer + caching
├── charts/
│   └── technical.py        # Plotly charts + indicators
├── agents/
│   └── financial_agents.py # Phi agents (LLaMA via Groq)
├── sentiment/
│   ├── reddit_scraper.py   # Reddit PRAW scraper
│   ├── twitter_scraper.py  # Twitter snscrape scraper
│   ├── analyzer.py         # VADER + FinBERT dual analysis
│   └── visualizations.py   # Sentiment charts
├── rag/
│   ├── document_store.py   # Vector store (MiniLM embeddings)
│   └── rag_agent.py        # RAG-enabled LLM agent
├── utils/
│   └── export.py           # PDF/CSV export
└── requirements.txt
```

## 🧠 Open-Source Models Used

| Model | Purpose | License |
|-------|---------|---------|
| LLaMA 3.3 70B | Financial analysis & RAG generation | Meta License |
| ProsusAI/finbert | Finance-specific sentiment | Apache 2.0 |
| VADER | Fast polarity scoring | MIT |
| all-MiniLM-L6-v2 | Document embeddings for RAG | Apache 2.0 |

## 📋 API Keys Required

| Service | Required | Free Tier |
|---------|----------|-----------|
| Groq | ✅ Yes | ✅ Free |
| Reddit API | Optional | ✅ Free |
| Twitter/X | No (snscrape) | N/A |
