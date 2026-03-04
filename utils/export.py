# utils/export.py — PDF and CSV export of analysis reports

import streamlit as st
import io
from datetime import datetime


def export_csv(data: dict, symbol: str) -> bytes:
    """Export analysis data as CSV bytes."""
    import pandas as pd

    rows = []
    # Metrics
    if "info" in data:
        info = data["info"]
        for key in ["marketCap", "trailingPE", "fiftyTwoWeekHigh", "fiftyTwoWeekLow",
                     "volume", "sector", "industry", "currency"]:
            rows.append({"Category": "Fundamental", "Metric": key, "Value": info.get(key, "N/A")})

    # Technical
    if "technicals" in data:
        for k, v in data["technicals"].items():
            rows.append({"Category": "Technical", "Metric": k, "Value": v})

    # Sentiment
    if "sentiment_summary" in data:
        for k, v in data["sentiment_summary"].items():
            rows.append({"Category": "Sentiment", "Metric": k, "Value": v})

    df = pd.DataFrame(rows)
    return df.to_csv(index=False).encode("utf-8")


def export_pdf(data: dict, symbol: str) -> bytes:
    """Export analysis as a PDF report using fpdf2 (open-source)."""
    try:
        from fpdf import FPDF
    except ImportError:
        st.warning("📦 `fpdf2` not installed. Run `pip install fpdf2` for PDF export.")
        return b""

    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()

    # Title
    pdf.set_font("Helvetica", "B", 20)
    pdf.cell(0, 15, f"DALAAL AI Report: {symbol}", ln=True, align="C")
    pdf.set_font("Helvetica", "", 10)
    pdf.cell(0, 8, f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}", ln=True, align="C")
    pdf.ln(10)

    # Fundamentals
    if "info" in data:
        _add_section(pdf, "Fundamentals", data["info"], [
            ("Market Cap", "marketCap"), ("P/E Ratio", "trailingPE"),
            ("52W High", "fiftyTwoWeekHigh"), ("52W Low", "fiftyTwoWeekLow"),
            ("Sector", "sector"), ("Industry", "industry"),
        ])

    # Technical indicators
    if "technicals" in data:
        pdf.set_font("Helvetica", "B", 14)
        pdf.cell(0, 10, "Technical Indicators", ln=True)
        pdf.set_font("Helvetica", "", 11)
        for k, v in data["technicals"].items():
            pdf.cell(0, 7, f"  {k}: {v}", ln=True)
        pdf.ln(5)

    # Sentiment
    if "sentiment_summary" in data:
        pdf.set_font("Helvetica", "B", 14)
        pdf.cell(0, 10, "Sentiment Analysis", ln=True)
        pdf.set_font("Helvetica", "", 11)
        s = data["sentiment_summary"]
        pdf.cell(0, 7, f"  Total Posts Analyzed: {s.get('total', 0)}", ln=True)
        pdf.cell(0, 7, f"  Positive: {s.get('positive_pct', 0)}%  |  Negative: {s.get('negative_pct', 0)}%  |  Neutral: {s.get('neutral_pct', 0)}%", ln=True)
        pdf.cell(0, 7, f"  Average Score: {s.get('avg_score', 0):+.3f}  ({s.get('label', 'neutral')})", ln=True)
        pdf.ln(5)

    return pdf.output()


def _add_section(pdf, title, info, fields):
    pdf.set_font("Helvetica", "B", 14)
    pdf.cell(0, 10, title, ln=True)
    pdf.set_font("Helvetica", "", 11)
    for label, key in fields:
        val = info.get(key, "N/A")
        if isinstance(val, (int, float)):
            val = f"{val:,.2f}" if isinstance(val, float) else f"{val:,}"
        pdf.cell(0, 7, f"  {label}: {val}", ln=True)
    pdf.ln(5)
