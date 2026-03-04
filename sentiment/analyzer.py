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


# ──────────────────────── FinBERT ────────────────────────

@st.cache_resource(show_spinner=False)
def _load_finbert():
    """Load FinBERT model (ProsusAI/finbert) — open-source, Apache 2.0 license.
    
    Cached via st.cache_resource so the model is loaded only once per session.
    """
    try:
        from transformers import AutoTokenizer, AutoModelForSequenceClassification
        import torch

        model_name = "ProsusAI/finbert"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSequenceClassification.from_pretrained(model_name)
        model.eval()
        return tokenizer, model
    except ImportError:
        st.warning("📦 `transformers` and `torch` needed for FinBERT. Run `pip install transformers torch`.")
        return None, None
    except Exception as e:
        st.warning(f"Could not load FinBERT: {e}")
        return None, None


def _finbert_batch(texts: list[str], batch_size: int = 16) -> list[dict]:
    """Run FinBERT inference in batches.
    
    Returns list of {score, label} dicts.
    Labels: 'positive', 'negative', 'neutral'
    """
    tokenizer, model = _load_finbert()
    if tokenizer is None or model is None:
        return [{"score": 0.0, "label": "neutral"}] * len(texts)

    try:
        import torch

        labels_map = {0: "positive", 1: "negative", 2: "neutral"}
        results = []

        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            # Truncate long texts to 512 tokens
            inputs = tokenizer(
                batch, padding=True, truncation=True,
                max_length=512, return_tensors="pt",
            )

            with torch.no_grad():
                outputs = model(**inputs)
                probs = torch.nn.functional.softmax(outputs.logits, dim=-1)

            for j in range(len(batch)):
                scores = probs[j].tolist()
                max_idx = scores.index(max(scores))
                results.append({
                    "score": scores[max_idx],
                    "label": labels_map[max_idx],
                })

        return results

    except Exception as e:
        st.warning(f"FinBERT inference error: {e}")
        return [{"score": 0.0, "label": "neutral"}] * len(texts)


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
