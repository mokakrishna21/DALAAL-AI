# sentiment/twitter_scraper.py — Twitter/X & Web discussion scraper
# Uses DuckDuckGo news + text search for stock discussions
# (Direct Twitter API scraping is no longer viable without paid API access)

import streamlit as st
from datetime import datetime


def scrape_twitter(query: str, limit: int = 30) -> list[dict]:
    """Find stock discussions from Twitter/X, news, and the web using DuckDuckGo.

    Direct Twitter scraping (snscrape, ntscraper) no longer works reliably
    since Twitter/X blocked public scraping in 2023. This uses DuckDuckGo
    news and text search to find recent stock discussions and sentiment.

    Returns list of dicts: {text, author, likes, timestamp, url, source}
    """
    posts = []

    # 1. News search — most reliable for financial content
    posts.extend(_search_news(query, limit))

    # 2. Text search for Twitter/forum discussions (supplement)
    remaining = limit - len(posts)
    if remaining > 0:
        posts.extend(_search_text(query, remaining))

    return posts


def _search_news(query: str, limit: int) -> list[dict]:
    """Search for recent stock news via DuckDuckGo news endpoint."""
    try:
        from duckduckgo_search import DDGS


        # Clean query for better search results
        clean_query = query.replace(".NS", "").replace(".BO", "")
        search_term = f"{clean_query} stock share price market"

        posts = []
        with DDGS() as ddgs:
            results = list(ddgs.news(search_term, max_results=limit))
            for r in results:
                title = r.get("title", "")
                body = r.get("body", "")
                text = f"{title}. {body}" if body else title

                if not text.strip():
                    continue

                posts.append({
                    "text": text,
                    "author": r.get("source", "News"),
                    "likes": 0,
                    "timestamp": r.get("date", datetime.now().isoformat()),
                    "url": r.get("url", ""),
                    "source": "twitter",  # labeled twitter for compatibility with analyzer
                })
        return posts

    except Exception as e:
        st.warning(f"⚠️ News search failed: {e}")
        return []


def _search_text(query: str, limit: int) -> list[dict]:
    """Search for stock discussions via DuckDuckGo text search."""
    if limit <= 0:
        return []

    try:
        from duckduckgo_search import DDGS

        clean_query = query.replace(".NS", "").replace(".BO", "")
        search_term = f"{clean_query} stock opinion analysis buy sell hold"

        posts = []
        with DDGS() as ddgs:
            results = list(ddgs.text(search_term, max_results=limit))
            for r in results:
                title = r.get("title", "")
                body = r.get("body", "")
                text = f"{title}. {body}" if body else title

                if not text.strip():
                    continue

                posts.append({
                    "text": text,
                    "author": _extract_source(r.get("href", "")),
                    "likes": 0,
                    "timestamp": datetime.now().isoformat(),
                    "url": r.get("href", ""),
                    "source": "twitter",  # labeled twitter for compatibility
                })
        return posts

    except Exception:
        return []


def _extract_source(url: str) -> str:
    """Extract a readable source name from URL."""
    if "twitter.com" in url or "x.com" in url:
        parts = url.replace("https://", "").replace("http://", "").split("/")
        if len(parts) >= 2 and parts[1] not in ("search", "hashtag", "i"):
            return f"@{parts[1]}"
    if "stocktwits.com" in url:
        return "StockTwits"
    if "reddit.com" in url:
        return "Reddit"
    if "moneycontrol.com" in url:
        return "MoneyControl"
    if "economictimes" in url:
        return "Economic Times"
    if "livemint.com" in url:
        return "LiveMint"
    if "yahoo.com" in url:
        return "Yahoo Finance"
    return "Web Source"
