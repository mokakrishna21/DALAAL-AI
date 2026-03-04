# charts/technical.py — Interactive Plotly charts with technical indicators

import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np


def _get_template(dark_mode: bool = False) -> str:
    return "plotly_dark" if dark_mode else "plotly_white"


# ──────────────────────────── Price Chart ────────────────────────────

def create_price_chart(hist: pd.DataFrame, symbol: str, dark_mode: bool = False) -> go.Figure:
    """Candlestick chart with 20/50/200 day moving averages."""
    fig = go.Figure()

    fig.add_trace(go.Candlestick(
        x=hist.index, open=hist["Open"], high=hist["High"],
        low=hist["Low"], close=hist["Close"], name="Price",
        increasing_line_color="#26a69a", decreasing_line_color="#ef5350",
    ))

    for window, color, name in [
        (20, "#FF9800", "MA 20"), (50, "#2196F3", "MA 50"), (200, "#9C27B0", "MA 200")
    ]:
        if len(hist) >= window:
            ma = hist["Close"].rolling(window=window).mean()
            fig.add_trace(go.Scatter(
                x=hist.index, y=ma, name=name,
                line=dict(color=color, width=1.5),
            ))

    fig.update_layout(
        title=f"{symbol} — Price Action",
        yaxis_title="Price", template=_get_template(dark_mode),
        xaxis_rangeslider_visible=False,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=50, r=20, t=60, b=40),
    )
    return fig


# ──────────────────────────── Volume Chart ────────────────────────────

def create_volume_chart(hist: pd.DataFrame, dark_mode: bool = False) -> go.Figure:
    """Volume bars with 20-day moving average."""
    colors = ["#26a69a" if c >= o else "#ef5350"
              for c, o in zip(hist["Close"], hist["Open"])]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=hist.index, y=hist["Volume"], name="Volume",
        marker_color=colors, opacity=0.6,
    ))
    if len(hist) >= 20:
        vol_ma = hist["Volume"].rolling(window=20).mean()
        fig.add_trace(go.Scatter(
            x=hist.index, y=vol_ma, name="20D Avg",
            line=dict(color="#FF5722", width=2),
        ))

    fig.update_layout(
        title="Trading Volume", yaxis_title="Volume",
        template=_get_template(dark_mode),
        margin=dict(l=50, r=20, t=50, b=40),
    )
    return fig


# ──────────────────────────── RSI ────────────────────────────

def compute_rsi(series: pd.Series, window: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.where(delta > 0, 0.0).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0.0)).rolling(window=window).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))


def create_rsi_chart(hist: pd.DataFrame, dark_mode: bool = False) -> go.Figure:
    rsi = compute_rsi(hist["Close"])
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=hist.index, y=rsi, name="RSI (14)",
        line=dict(color="#7E57C2", width=2),
    ))
    fig.add_hline(y=70, line_dash="dash", line_color="red", annotation_text="Overbought")
    fig.add_hline(y=30, line_dash="dash", line_color="green", annotation_text="Oversold")
    fig.update_layout(
        title="RSI (14)", yaxis_title="RSI", yaxis_range=[0, 100],
        template=_get_template(dark_mode),
        margin=dict(l=50, r=20, t=50, b=40),
    )
    return fig


# ──────────────────────────── MACD ────────────────────────────

def create_macd_chart(hist: pd.DataFrame, dark_mode: bool = False) -> go.Figure:
    ema12 = hist["Close"].ewm(span=12, adjust=False).mean()
    ema26 = hist["Close"].ewm(span=26, adjust=False).mean()
    macd_line = ema12 - ema26
    signal_line = macd_line.ewm(span=9, adjust=False).mean()
    histogram = macd_line - signal_line

    colors = ["#26a69a" if v >= 0 else "#ef5350" for v in histogram]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=hist.index, y=histogram, name="Histogram",
        marker_color=colors, opacity=0.5,
    ))
    fig.add_trace(go.Scatter(
        x=hist.index, y=macd_line, name="MACD",
        line=dict(color="#2196F3", width=2),
    ))
    fig.add_trace(go.Scatter(
        x=hist.index, y=signal_line, name="Signal",
        line=dict(color="#FF9800", width=2),
    ))
    fig.update_layout(
        title="MACD (12, 26, 9)", yaxis_title="MACD",
        template=_get_template(dark_mode),
        margin=dict(l=50, r=20, t=50, b=40),
    )
    return fig


# ──────────────────────────── Bollinger Bands ────────────────────────────

def create_bollinger_chart(hist: pd.DataFrame, dark_mode: bool = False) -> go.Figure:
    sma20 = hist["Close"].rolling(window=20).mean()
    std20 = hist["Close"].rolling(window=20).std()
    upper = sma20 + 2 * std20
    lower = sma20 - 2 * std20

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=hist.index, y=upper, name="Upper Band",
        line=dict(color="rgba(33,150,243,0.3)"), showlegend=False,
    ))
    fig.add_trace(go.Scatter(
        x=hist.index, y=lower, name="Lower Band",
        line=dict(color="rgba(33,150,243,0.3)"),
        fill="tonexty", fillcolor="rgba(33,150,243,0.08)", showlegend=False,
    ))
    fig.add_trace(go.Scatter(
        x=hist.index, y=sma20, name="SMA 20",
        line=dict(color="#2196F3", width=1.5),
    ))
    fig.add_trace(go.Scatter(
        x=hist.index, y=hist["Close"], name="Close",
        line=dict(color="#FF9800", width=2),
    ))
    fig.update_layout(
        title="Bollinger Bands (20, 2)", yaxis_title="Price",
        template=_get_template(dark_mode),
        margin=dict(l=50, r=20, t=50, b=40),
    )
    return fig


# ──────────────────────────── Helper Metrics ────────────────────────────

def compute_technical_metrics(hist: pd.DataFrame) -> dict:
    """Compute a summary dict of key technical indicators."""
    rsi = compute_rsi(hist["Close"])
    ma20 = hist["Close"].rolling(window=20).mean()
    ma50 = hist["Close"].rolling(window=50).mean()
    volatility = hist["Close"].pct_change().std() * np.sqrt(252) * 100

    daily_returns = hist["Close"].pct_change().dropna()
    sharpe = (daily_returns.mean() / daily_returns.std()) * np.sqrt(252) if daily_returns.std() != 0 else 0

    return {
        "rsi_14": round(rsi.iloc[-1], 2) if len(rsi.dropna()) > 0 else None,
        "ma_cross": "Bullish 🟢" if len(ma20.dropna()) > 0 and len(ma50.dropna()) > 0 and ma20.iloc[-1] > ma50.iloc[-1] else "Bearish 🔴",
        "volatility_pct": round(volatility, 2),
        "sharpe_ratio": round(sharpe, 2),
        "daily_change_pct": round(daily_returns.iloc[-1] * 100, 2) if len(daily_returns) > 0 else None,
    }
