# sentiment/twitter_scraper.py — Twitter/X post scraper (open-source, no API key)

import streamlit as st
from datetime import datetime, timedelta


def scrape_twitter(query: str, limit: int = 30) -> list[dict]:
    """Scrape tweets about a stock using open-source tools.

    Tries in order:
    1. snscrape (most reliable, scrapes public tweets without API)
    2. Nitter fallback via ntscraper
    3. Web search fallback (DuckDuckGo)

    Returns list of dicts: {text, author, likes, timestamp, url, source}
    """
    # 1) Try snscrape
    posts = _try_snscrape(query, limit)
    if posts:
        return posts

    # 2) Try ntscraper (Nitter)
    posts = _try_ntscraper(query, limit)
    if posts:
        return posts

    # 3) Web search fallback
    return _web_search_fallback(query, limit)


def _try_snscrape(query: str, limit: int) -> list[dict]:
    """Scrape tweets using snscrape (open-source, no API key needed)."""
    try:
        import snscrape.modules.twitter as sntwitter

        search_query = f"{query} stock lang:en since:{(datetime.now() - timedelta(days=7)).strftime('%Y-%m-%d')}"
        posts = []

        for i, tweet in enumerate(sntwitter.TwitterSearchScraper(search_query).get_items()):
            if i >= limit:
                break
            posts.append({
                "text": tweet.rawContent,
                "author": tweet.user.username if tweet.user else "Unknown",
                "likes": tweet.likeCount or 0,
                "timestamp": tweet.date.isoformat() if tweet.date else "",
                "url": tweet.url or "",
                "source": "twitter",
            })

        return posts
    except Exception:
        return []


def _try_ntscraper(query: str, limit: int) -> list[dict]:
    """Scrape tweets using ntscraper (Nitter instances, open-source)."""
    try:
        from ntscraper import Nitter

        scraper = Nitter()
        results = scraper.get_tweets(query, mode="term", number=limit)

        posts = []
        for tweet in results.get("tweets", []):
            posts.append({
                "text": tweet.get("text", ""),
                "author": tweet.get("user", {}).get("username", "Unknown"),
                "likes": tweet.get("stats", {}).get("likes", 0),
                "timestamp": tweet.get("date", ""),
                "url": tweet.get("link", ""),
                "source": "twitter/nitter",
            })
        return posts
    except Exception:
        return []


def _web_search_fallback(query: str, limit: int) -> list[dict]:
    """Fallback: find stock discussions via DuckDuckGo when Twitter scraping fails."""
    try:
        from duckduckgo_search import DDGS

        search_term = f"{query} stock market discussion twitter"
        posts = []

        with DDGS() as ddgs:
            results = ddgs.text(search_term, max_results=limit)
            for r in results:
                posts.append({
                    "text": f"{r.get('title', '')}. {r.get('body', '')}",
                    "author": "Web Source",
                    "likes": 0,
                    "timestamp": datetime.now().isoformat(),
                    "url": r.get("href", ""),
                    "source": "web_search",
                })
        return posts
    except Exception:
        return []
