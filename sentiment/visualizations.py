# sentiment/visualizations.py — Sentiment charts and visual summaries

import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
from datetime import datetime


def create_sentiment_gauge(avg_score: float, dark_mode: bool = False) -> go.Figure:
    """Create a sentiment gauge chart showing overall mood (–1 to +1)."""
    color = "#26a69a" if avg_score >= 0.15 else "#ef5350" if avg_score <= -0.15 else "#FF9800"

    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=avg_score,
        number={"suffix": "", "font": {"size": 36}},
        title={"text": "Overall Sentiment", "font": {"size": 18}},
        gauge={
            "axis": {"range": [-1, 1], "tickwidth": 1},
            "bar": {"color": color},
            "steps": [
                {"range": [-1, -0.15], "color": "rgba(239,83,80,0.15)"},
                {"range": [-0.15, 0.15], "color": "rgba(255,152,0,0.15)"},
                {"range": [0.15, 1], "color": "rgba(38,166,154,0.15)"},
            ],
            "threshold": {
                "line": {"color": "white", "width": 3},
                "thickness": 0.8,
                "value": avg_score,
            },
        },
    ))
    template = "plotly_dark" if dark_mode else "plotly_white"
    fig.update_layout(template=template, margin=dict(l=30, r=30, t=60, b=30))
    return fig


def create_sentiment_pie(summary: dict, dark_mode: bool = False) -> go.Figure:
    """Pie chart showing Positive / Negative / Neutral distribution."""
    labels = ["Positive", "Negative", "Neutral"]
    values = [summary.get("positive", 0), summary.get("negative", 0), summary.get("neutral", 0)]
    colors = ["#26a69a", "#ef5350", "#FF9800"]

    fig = go.Figure(go.Pie(
        labels=labels, values=values,
        marker_colors=colors, hole=0.45,
        textinfo="percent+label",
        textfont_size=13,
    ))
    template = "plotly_dark" if dark_mode else "plotly_white"
    fig.update_layout(
        title="Sentiment Distribution",
        template=template, margin=dict(l=30, r=30, t=60, b=30),
        showlegend=False,
    )
    return fig


def create_sentiment_timeline(posts: list[dict], dark_mode: bool = False) -> go.Figure:
    """Line chart of sentiment scores over time."""
    if not posts:
        return go.Figure()

    df = pd.DataFrame(posts)
    if "timestamp" not in df.columns or "combined_score" not in df.columns:
        return go.Figure()

    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df = df.dropna(subset=["timestamp"]).sort_values("timestamp")

    if df.empty:
        return go.Figure()

    # Rolling average if enough data points
    fig = go.Figure()

    # Individual points
    colors = ["#26a69a" if s >= 0.15 else "#ef5350" if s <= -0.15 else "#FF9800"
              for s in df["combined_score"]]
    fig.add_trace(go.Scatter(
        x=df["timestamp"], y=df["combined_score"],
        mode="markers", name="Posts",
        marker=dict(color=colors, size=8, opacity=0.7),
    ))

    # Trend line (rolling average)
    if len(df) >= 5:
        df["trend"] = df["combined_score"].rolling(window=min(5, len(df)), center=True).mean()
        fig.add_trace(go.Scatter(
            x=df["timestamp"], y=df["trend"],
            mode="lines", name="Trend",
            line=dict(color="#7E57C2", width=3),
        ))

    fig.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5)
    template = "plotly_dark" if dark_mode else "plotly_white"
    fig.update_layout(
        title="Sentiment Over Time",
        yaxis_title="Sentiment Score", yaxis_range=[-1.1, 1.1],
        template=template,
        margin=dict(l=50, r=20, t=60, b=40),
    )
    return fig


def create_source_comparison(posts: list[dict], dark_mode: bool = False) -> go.Figure:
    """Bar chart comparing sentiment by source (Reddit vs Twitter)."""
    if not posts:
        return go.Figure()

    df = pd.DataFrame(posts)
    if "source" not in df.columns or "combined_score" not in df.columns:
        return go.Figure()

    grouped = df.groupby("source")["combined_score"].agg(["mean", "count"]).reset_index()
    colors = ["#26a69a" if m >= 0.15 else "#ef5350" if m <= -0.15 else "#FF9800"
              for m in grouped["mean"]]

    fig = go.Figure(go.Bar(
        x=grouped["source"], y=grouped["mean"],
        text=[f"{m:.2f} ({int(c)} posts)" for m, c in zip(grouped["mean"], grouped["count"])],
        textposition="outside",
        marker_color=colors,
    ))
    template = "plotly_dark" if dark_mode else "plotly_white"
    fig.update_layout(
        title="Sentiment by Source",
        yaxis_title="Avg Sentiment", yaxis_range=[-1, 1],
        template=template,
        margin=dict(l=50, r=20, t=60, b=40),
    )
    return fig


def create_word_cloud_data(posts: list[dict]) -> dict:
    """Extract word frequencies for word cloud display.

    Returns dict of {word: frequency}.
    Uses basic tokenization — avoids heavy deps.
    """
    import re
    from collections import Counter

    stop_words = {
        "the", "a", "an", "is", "it", "to", "in", "for", "of", "and", "or",
        "on", "at", "by", "be", "as", "this", "that", "with", "from", "are",
        "was", "were", "been", "has", "have", "had", "do", "does", "did",
        "will", "would", "could", "should", "may", "might", "can", "just",
        "not", "no", "but", "if", "so", "he", "she", "they", "we", "you",
        "i", "my", "your", "its", "our", "their", "what", "which", "who",
        "how", "when", "where", "all", "each", "every", "both", "more",
        "than", "very", "too", "also", "only", "about", "up", "out", "into",
        "over", "after", "before", "between", "under", "http", "https", "www",
        "com", "reddit", "deleted", "removed",
    }

    all_text = " ".join(p.get("text", "") for p in posts).lower()
    words = re.findall(r'\b[a-z]{3,}\b', all_text)
    filtered = [w for w in words if w not in stop_words]
    return dict(Counter(filtered).most_common(80))


def display_top_posts(posts: list[dict], n: int = 10):
    """Display the top-n posts by engagement with sentiment labels."""
    import streamlit as st

    if not posts:
        st.info("No posts to display.")
        return

    sorted_posts = sorted(posts, key=lambda x: abs(x.get("combined_score", 0)), reverse=True)[:n]

    for p in sorted_posts:
        label = p.get("sentiment_label", "neutral")
        emoji = "🟢" if label == "positive" else "🔴" if label == "negative" else "🟡"
        score = p.get("combined_score", 0)
        source = p.get("source", "unknown").title()

        with st.expander(f"{emoji} [{source}] Score: {score:+.2f} — {p['text'][:80]}..."):
            st.write(p.get("text", ""))
            cols = st.columns(4)
            cols[0].metric("VADER", f"{p.get('vader_score', 0):+.3f}")
            cols[1].metric("FinBERT", f"{p.get('finbert_score', 0):+.3f}")
            cols[2].metric("Combined", f"{score:+.3f}")
            cols[3].metric("Source", source)
            if p.get("url"):
                st.markdown(f"[View original →]({p['url']})")
