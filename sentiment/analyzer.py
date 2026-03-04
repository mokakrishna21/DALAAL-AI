# sentiment/analyzer.py — Dual-model sentiment: VADER (speed) + FinBERT (accuracy)
# Both models are fully open-source.

import streamlit as st
from datetime import datetime


def analyze_sentiment(posts: list[dict]) -> list[dict]:
    """Run dual sentiment analysis on a list of posts.

    - VADER: rule-based, fast, good for short text (tweets, Reddit titles)
    - FinBERT: transformer-based, trained on finance text, accurate for domain terms

    Combined score = 0.3 × VADER + 0.7 × FinBERT (finance domain weighting)
    """
    if not posts:
        return []

    texts = [p["text"] for p in posts]

    # Run VADER
    vader_scores = _vader_batch(texts)

    # Run FinBERT
    finbert_scores = _finbert_batch(texts)

    # Combine results
    for i, post in enumerate(posts):
        v = vader_scores[i] if i < len(vader_scores) else 0.0
        f = finbert_scores[i] if i < len(finbert_scores) else {"score": 0.0, "label": "neutral"}

        # Normalize FinBERT to [-1, 1] scale
        fb_score = f["score"]
        if f["label"] == "negative":
            fb_score = -fb_score
        elif f["label"] == "neutral":
            fb_score = 0.0

        # Weighted combination
        combined = 0.3 * v + 0.7 * fb_score

        post["vader_score"] = round(v, 4)
        post["finbert_score"] = round(fb_score, 4)
        post["finbert_label"] = f["label"]
        post["combined_score"] = round(combined, 4)
        post["sentiment_label"] = _classify(combined)

    return posts


def _classify(score: float) -> str:
    """Classify a combined score into a label."""
    if score >= 0.15:
        return "positive"
    elif score <= -0.15:
        return "negative"
    return "neutral"


# ──────────────────────── VADER ────────────────────────

def _vader_batch(texts: list[str]) -> list[float]:
    """Run VADER sentiment analysis. Returns list of compound scores (–1 to +1)."""
    try:
        from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
        analyzer = SentimentIntensityAnalyzer()
        return [analyzer.polarity_scores(t)["compound"] for t in texts]
    except ImportError:
        st.warning("📦 `vaderSentiment` not installed. Run `pip install vaderSentiment`.")
        return [0.0] * len(texts)
    except Exception:
        return [0.0] * len(texts)


# ──────────────────────── FinBERT (via HuggingFace Inference API) ────────────────────────

FINBERT_API_URL = "https://api-inference.huggingface.co/models/ProsusAI/finbert"


def _get_hf_api_key() -> str:
    """Get HuggingFace API key from Streamlit secrets or env."""
    import os
    try:
        return st.secrets.get("HF_API_KEY", os.environ.get("HF_API_KEY", ""))
    except Exception:
        return os.environ.get("HF_API_KEY", "")


def _finbert_batch(texts: list[str], batch_size: int = 10) -> list[dict]:
    """Run FinBERT via HuggingFace Inference API (no local model needed).

    Uses the free HF Inference API — runs on HuggingFace servers.
    Returns list of {score, label} dicts.
    """
    api_key = _get_hf_api_key()
    if not api_key:
        st.warning("🔑 HuggingFace API key not set. Add `HF_API_KEY` to secrets for FinBERT sentiment.")
        return [{"score": 0.0, "label": "neutral"}] * len(texts)

    import requests
    import time

    headers = {"Authorization": f"Bearer {api_key}"}
    results = []

    for i in range(0, len(texts), batch_size):
        batch = [t[:512] for t in texts[i : i + batch_size]]  # Truncate to ~512 chars

        for attempt in range(3):
            try:
                response = requests.post(
                    FINBERT_API_URL,
                    headers=headers,
                    json={"inputs": batch, "options": {"wait_for_model": True}},
                    timeout=30,
                )

                if response.status_code == 200:
                    data = response.json()
                    for item in data:
                        if isinstance(item, list) and item:
                            # HF returns sorted list of {label, score} per input
                            best = item[0]
                            results.append({
                                "score": best.get("score", 0.0),
                                "label": best.get("label", "neutral").lower(),
                            })
                        else:
                            results.append({"score": 0.0, "label": "neutral"})
                    break  # Success — exit retry loop

                elif response.status_code == 503:
                    # Model loading — wait and retry
                    time.sleep(3)
                    continue
                else:
                    st.warning(f"FinBERT API error {response.status_code}: {response.text[:100]}")
                    results.extend([{"score": 0.0, "label": "neutral"}] * len(batch))
                    break

            except Exception as e:
                if attempt == 2:
                    st.warning(f"FinBERT API failed after 3 retries: {e}")
                    results.extend([{"score": 0.0, "label": "neutral"}] * len(batch))
                time.sleep(2)

    return results


def get_sentiment_summary(posts: list[dict]) -> dict:
    """Compute aggregate sentiment statistics from analyzed posts."""
    if not posts:
        return {"total": 0, "positive": 0, "negative": 0, "neutral": 0,
                "avg_score": 0.0, "label": "neutral"}

    labels = [p.get("sentiment_label", "neutral") for p in posts]
    scores = [p.get("combined_score", 0.0) for p in posts]

    pos = labels.count("positive")
    neg = labels.count("negative")
    neu = labels.count("neutral")
    avg = sum(scores) / len(scores) if scores else 0.0

    return {
        "total": len(posts),
        "positive": pos,
        "negative": neg,
        "neutral": neu,
        "positive_pct": round(pos / len(posts) * 100, 1),
        "negative_pct": round(neg / len(posts) * 100, 1),
        "neutral_pct": round(neu / len(posts) * 100, 1),
        "avg_score": round(avg, 4),
        "label": _classify(avg),
    }
