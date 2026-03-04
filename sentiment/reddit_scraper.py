# sentiment/reddit_scraper.py — Reddit post scraper using PRAW

import streamlit as st
from datetime import datetime
from config import get_reddit_credentials, SUBREDDITS, REDDIT_POST_LIMIT


def scrape_reddit(query: str, limit: int = REDDIT_POST_LIMIT) -> list[dict]:
    """Scrape Reddit posts mentioning the given stock/query.

    Uses PRAW (Python Reddit API Wrapper) — requires Reddit API credentials.
    Falls back gracefully if PRAW is not installed or credentials are missing.

    Returns list of dicts: {text, author, score, timestamp, subreddit, url}
    """
    try:
        import praw
    except ImportError:
        st.warning("📦 `praw` not installed. Run `pip install praw` for Reddit sentiment.")
        return []

    creds = get_reddit_credentials()
    if not creds["client_id"] or not creds["client_secret"]:
        st.info("🔑 Reddit API credentials not configured. Add `REDDIT_CLIENT_ID` and `REDDIT_CLIENT_SECRET` to secrets.")
        return _fallback_reddit_search(query, limit)

    try:
        reddit = praw.Reddit(
            client_id=creds["client_id"],
            client_secret=creds["client_secret"],
            user_agent=creds["user_agent"],
        )

        posts = []
        for subreddit_name in SUBREDDITS:
            try:
                subreddit = reddit.subreddit(subreddit_name)
                for post in subreddit.search(query, sort="relevance", time_filter="month", limit=limit // len(SUBREDDITS)):
                    posts.append({
                        "text": f"{post.title}. {post.selftext[:500]}" if post.selftext else post.title,
                        "author": str(post.author) if post.author else "Unknown",
                        "score": post.score,
                        "timestamp": datetime.fromtimestamp(post.created_utc).isoformat(),
                        "subreddit": subreddit_name,
                        "url": f"https://reddit.com{post.permalink}",
                        "source": "reddit",
                    })
            except Exception:
                continue

        return sorted(posts, key=lambda x: x["score"], reverse=True)[:limit]

    except Exception as e:
        st.warning(f"Reddit scraping failed: {e}")
        return _fallback_reddit_search(query, limit)


def _fallback_reddit_search(query: str, limit: int) -> list[dict]:
    """Fallback: search Reddit via web scraping when API is unavailable."""
    try:
        import requests
        from bs4 import BeautifulSoup

        url = f"https://www.reddit.com/search.json?q={query}+stock&sort=relevance&t=month&limit={limit}"
        headers = {"User-Agent": "dalaal-ai/1.0"}
        resp = requests.get(url, headers=headers, timeout=10)

        if resp.status_code != 200:
            return []

        data = resp.json()
        posts = []
        for child in data.get("data", {}).get("children", []):
            d = child.get("data", {})
            posts.append({
                "text": f"{d.get('title', '')}. {d.get('selftext', '')[:500]}",
                "author": d.get("author", "Unknown"),
                "score": d.get("score", 0),
                "timestamp": datetime.fromtimestamp(d.get("created_utc", 0)).isoformat(),
                "subreddit": d.get("subreddit", ""),
                "url": f"https://reddit.com{d.get('permalink', '')}",
                "source": "reddit",
            })
        return posts

    except Exception:
        return []
