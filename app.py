# app.py — DALAAL AI: Smart Stock Intelligence Dashboard
# Main Streamlit entry point — UI orchestration only

import streamlit as st
import pandas as pd
from datetime import datetime
import importlib
import config

from config import APP_TITLE, APP_ICON, PERIOD_MAP
from data.market_data import (
    get_symbol_from_name, get_stock_data,
    get_institutional_holders, format_large_number,
)
from charts.technical import (
    create_price_chart, create_volume_chart,
    create_rsi_chart, create_macd_chart,
    create_bollinger_chart, compute_technical_metrics,
)
from agents.financial_agents import (
    initialize_agents, get_agent_for_analysis_type, ANALYSIS_TYPES,
)
from sentiment.reddit_scraper import scrape_reddit
from sentiment.twitter_scraper import scrape_twitter
from sentiment.analyzer import analyze_sentiment, get_sentiment_summary
from sentiment.visualizations import (
    create_sentiment_gauge, create_sentiment_pie,
    create_sentiment_timeline, create_source_comparison,
    create_word_cloud_data, display_top_posts,
)
from rag.document_store import DocumentStore
from rag.rag_agent import create_rag_agent, query_rag
from utils.export import export_csv, export_pdf


# ═══════════════════════════════ Page Config ═══════════════════════════════

st.set_page_config(
    page_title=APP_TITLE, page_icon=APP_ICON,
    layout="wide", initial_sidebar_state="expanded",
)


# ═══════════════════════════════ Custom CSS ═══════════════════════════════

def inject_css():
    card_bg = "#f8f9fa"
    accent = "#7c4dff"
    accent2 = "#00e5ff"

    st.markdown(f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

    .stApp {{
        font-family: 'Inter', sans-serif;
        max-width: 1500px;
        margin: 0 auto;
    }}
    .main {{ padding: 1rem 2rem; }}

    .hero-header {{
        background: linear-gradient(135deg, {accent}, {accent2});
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 2.2rem;
        font-weight: 700;
        text-align: center;
        padding: 0.5rem 0 0.3rem 0;
        letter-spacing: -0.5px;
    }}
    .hero-sub {{
        text-align: center;
        color: #888;
        font-size: 0.95rem;
        margin-bottom: 1.5rem;
    }}

    .metric-card {{
        background: {card_bg};
        border-radius: 12px;
        padding: 1rem 1.2rem;
        margin: 0.4rem 0;
        box-shadow: 0 2px 8px rgba(0,0,0,0.08);
        transition: transform 0.2s, box-shadow 0.2s;
        border: 1px solid #e8e8e8;
    }}
    .metric-card:hover {{
        transform: translateY(-3px);
        box-shadow: 0 6px 16px rgba(124,77,255,0.15);
    }}

    .news-card {{
        background: {card_bg};
        padding: 1rem 1.2rem;
        border-radius: 8px;
        margin: 8px 0;
        border-left: 4px solid {accent};
        transition: transform 0.2s;
    }}
    .news-card:hover {{ transform: translateX(5px); }}

    .sentiment-badge {{
        display: inline-block;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 0.85rem;
        font-weight: 600;
    }}
    .sentiment-positive {{ background: rgba(38,166,154,0.15); color: #26a69a; }}
    .sentiment-negative {{ background: rgba(239,83,80,0.15); color: #ef5350; }}
    .sentiment-neutral  {{ background: rgba(255,152,0,0.15); color: #FF9800; }}

    .stButton>button {{
        width: 100%;
        border-radius: 8px;
        font-weight: 600;
        transition: transform 0.1s;
    }}
    .stButton>button:active {{ transform: scale(0.98); }}

    div[data-testid="stMetricValue"] {{ font-size: 1.3rem; font-weight: 600; }}
    </style>
    """, unsafe_allow_html=True)


# ═══════════════════════════════ Session State ═══════════════════════════════

def init_session_state():
    # CACHE BUSTER: Increment this when changing agent configurations 
    # to force long-running Streamlit sessions to recreate them
    CACHE_VERSION = 5
    if st.session_state.get("cache_version") != CACHE_VERSION:
        st.session_state.clear()
        st.session_state["cache_version"] = CACHE_VERSION

    defaults = {
        "agents_initialized": False,
        "watchlist": set(),
        "analysis_history": [],
        "last_refresh": None,
        "doc_store": DocumentStore(),
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


# ═══════════════════════════════ Sidebar ═══════════════════════════════

def render_sidebar():
    with st.sidebar:
        st.markdown("## ⚙️ Settings")
        importlib.reload(config)

        st.markdown("---")
        # Global Model Exorcism (Targeted Wipe of Decommissioned Models)
        is_legacy = False
        legacy_models = ["llama-3.1-70b-versatile"]
        
        # 1. Check stored model_id
        if st.session_state.get("model_id") in legacy_models:
            is_legacy = True
            
        # 2. Direct inspection of stored agents (if initialized)
        if not is_legacy and st.session_state.get("agents_initialized"):
            # We check the most important agents
            for agent_key in ["multi_ai_agent", "fundamental_agent"]:
                if agent_key in st.session_state:
                    try:
                        agent_obj = st.session_state[agent_key]
                        if hasattr(agent_obj, "model") and hasattr(agent_obj.model, "id"):
                            model_id = str(agent_obj.model.id).lower()
                            if any(legacy in model_id for legacy in legacy_models):
                                is_legacy = True
                                break
                    except Exception:
                        pass
        
        if is_legacy:
            st.session_state.agents_initialized = False
            # Clear them explicitly just in case
            for k in ["multi_ai_agent", "fundamental_agent", "model_id"]:
                if k in st.session_state:
                    del st.session_state[k]
            st.warning("⚠️ Decommissioned model detected. Resetting agents...")
            st.rerun()

        if st.button("🔄 Force Refresh Agents"):
            st.session_state.agents_initialized = False
            if "model_id" in st.session_state:
                del st.session_state["model_id"]
            st.success("Agents reset. They will re-initialize on next use.")
            st.rerun()

        if st.button("🗑️ Clear Data Cache", width="stretch"):
            st.cache_data.clear()
            st.toast("✅ Cache cleared! You can now fetch fresh data.")

        analysis_type = st.selectbox("Analysis Type", ANALYSIS_TYPES)

        market = st.selectbox(
            "Market", ["US Market", "Indian Market (NSE)", "Indian Market (BSE)"]
        )

        st.markdown("---")

        # Watchlist
        st.markdown("### 📋 Watchlist")
        wl_input = st.text_input("Add stock", placeholder="e.g. NVIDIA, TCS")
        if st.button("➕ Add", width="stretch"):
            sym = get_symbol_from_name(wl_input)
            if sym:
                st.session_state.watchlist.add(sym)
                st.toast(f"✅ Added {sym}")

        for sym in list(st.session_state.watchlist):
            c1, c2 = st.columns([4, 1])
            c1.write(f"📌 {sym}")
            if c2.button("✕", key=f"rm_{sym}"):
                st.session_state.watchlist.discard(sym)
                st.rerun()

        st.markdown("---")
        st.markdown(
            '<p style="text-align:center;color:#888;font-size:0.8rem;">'
            'Powered by open-source models<br>LLaMA • FinBERT • VADER • MiniLM</p>',
            unsafe_allow_html=True,
        )

    return analysis_type, market


# ═══════════════════════════════ Display Helpers ═══════════════════════════════

def display_metrics(info):
    """Display key financial metrics in card grid."""
    metrics = [
        ("Market Cap", "marketCap", True),
        ("P/E Ratio", "trailingPE", False),
        ("52W High", "fiftyTwoWeekHigh", False),
        ("52W Low", "fiftyTwoWeekLow", False),
    ]
    cols = st.columns(4)
    for i, (label, key, is_money) in enumerate(metrics):
        with cols[i]:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            val = info.get(key, "N/A")
            if val != "N/A" and is_money:
                ccy = "₹" if info.get("currency") == "INR" else "$"
                val = format_large_number(val, ccy)
            elif val != "N/A" and isinstance(val, float):
                val = f"{val:.2f}"
            st.metric(label, val)
            st.markdown("</div>", unsafe_allow_html=True)


# ═══════════════════════════════ Tab: Overview ═══════════════════════════════

def tab_overview(info, hist):
    st.markdown("### 🏢 Company Overview")
    st.write(info.get("longBusinessSummary", "No description available."))

    st.markdown("### 📊 Key Metrics")
    display_metrics(info)

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("### Company Details")
        employees = info.get("fullTimeEmployees", "N/A")
        if employees != "N/A":
            employees = f"{employees:,}"
        for label, key, fmt in [
            ("Sector", "sector", None), ("Industry", "industry", None),
            ("Country", "country", None), ("Employees", "fullTimeEmployees", None),
        ]:
            st.write(f"**{label}:** {info.get(key, 'N/A') if fmt is None else fmt}")

    with c2:
        st.markdown("### Trading Info")
        for label, key in [
            ("Exchange", "exchange"), ("Currency", "currency"),
            ("Volume", "volume"), ("Avg Volume", "averageVolume"),
        ]:
            val = info.get(key, "N/A")
            if isinstance(val, (int, float)) and key in ("volume", "averageVolume"):
                val = f"{val:,}"
            st.write(f"**{label}:** {val}")

    # Institutional Holders
    inst, major = get_institutional_holders(info.get("symbol", ""))
    if inst is not None and not inst.empty:
        st.markdown("### 🏛️ Top Institutional Holders")
        st.dataframe(inst.head(10), width="stretch", hide_index=True)

    # AI-powered fundamental overview
    st.markdown("---")
    st.markdown("### 🤖 AI Fundamental Summary")
    if st.button("Generate AI Overview", key="ai_overview_btn"):
        if initialize_agents():
            with st.spinner("Fundamental Analysis Agent is analyzing..."):
                container = st.container()
                try:
                    response = st.session_state.fundamental_agent.run(
                        f"Provide a brief fundamental overview of {info.get('shortName', '')} ({info.get('symbol', '')}). "
                        f"Focus on valuation (P/E, P/B), financial health, and growth.",
                        stream=False,
                    )
                    if response and response.content:
                        container.markdown(response.content)
                except Exception as e:
                    st.error(f"⚠️ AI analysis failed: {e}")


# ═══════════════════════════════ Tab: Charts ═══════════════════════════════

def tab_charts(hist, symbol):
    st.markdown("### 📈 Price Action")
    st.plotly_chart(create_price_chart(hist, symbol, False))

    c1, c2 = st.columns(2)
    with c1:
        st.plotly_chart(create_volume_chart(hist, False))
    with c2:
        st.plotly_chart(create_rsi_chart(hist, False))

    c3, c4 = st.columns(2)
    with c3:
        st.plotly_chart(create_macd_chart(hist, False))
    with c4:
        st.plotly_chart(create_bollinger_chart(hist, False))

    # Summary metrics
    st.markdown("### 🔢 Technical Summary")
    tech = compute_technical_metrics(hist)
    cols = st.columns(5)
    labels = [
        ("RSI (14)", tech.get("rsi_14", "N/A")),
        ("MA Cross", tech.get("ma_cross", "N/A")),
        ("Volatility", f"{tech.get('volatility_pct', 0)}%"),
        ("Sharpe Ratio", tech.get("sharpe_ratio", "N/A")),
        ("Daily Chg", f"{tech.get('daily_change_pct', 0):+.2f}%"),
    ]
    for i, (l, v) in enumerate(labels):
        cols[i].metric(l, v)

    # AI-powered technical analysis
    st.markdown("---")
    st.markdown("### 🤖 AI Technical Analysis")
    if st.button("Generate AI Technical Analysis", key="ai_tech_btn"):
        if initialize_agents():
            with st.spinner("Technical Analysis Agent is analyzing charts..."):
                container = st.container()
                try:
                    response = st.session_state.technical_agent.run(
                        f"Analyze {symbol} technically. RSI is {tech.get('rsi_14', 'N/A')}, "
                        f"MA cross signal is {tech.get('ma_cross', 'N/A')}, "
                        f"annualized volatility is {tech.get('volatility_pct', 0)}%. "
                        f"Identify patterns, support/resistance, and give a trade setup.",
                        stream=False,
                    )
                    if response and response.content:
                        container.markdown(response.content)
                except Exception as e:
                    st.error(f"⚠️ AI analysis failed: {e}")


# ═══════════════════════════════ Tab: AI Analysis ═══════════════════════════════

def tab_analysis(symbol, analysis_type, doc_store):
    st.markdown("### 🤖 AI Agent Hub")
    st.caption(f"Currently selected: **{analysis_type}** — change in sidebar")

    # Show which agent will handle this
    agent_key = get_agent_for_analysis_type(analysis_type)
    agent_names = {
        "multi_ai_agent": ("Master Orchestrator", "Delegates to all 10 specialist agents"),
        "technical_agent": ("Technical Analysis Agent", "Chart patterns, indicators, support/resistance"),
        "fundamental_agent": ("Fundamental Analysis Agent", "Valuation, financials, growth, DCF"),
        "news_agent": ("News Curator Agent", "Curated news with impact ratings"),
        "sentiment_agent": ("Sentiment Intelligence Agent", "Social media, retail vs institutional mood"),
        "risk_agent": ("Risk Assessment Agent", "Multi-dimensional risk scoring, VaR"),
        "comparison_agent": ("Stock Comparison Agent", "Peer comparison across metrics"),
        "sector_agent": ("Sector & Industry Agent", "Sector trends, competitive landscape"),
        "institutional_agent": ("Institutional & Insider Agent", "Smart money flows, insider trading"),
        "report_agent": ("Report Generator Agent", "Full investment report from all agents"),
    }
    name, desc = agent_names.get(agent_key, ("AI Agent", ""))
    st.info(f"🤖 **{name}** — {desc}")

    # Display previously run analysis to survive Streamlit reruns
    if "current_analysis_output" in st.session_state and st.session_state.current_analysis_symbol == symbol:
        with st.container(border=True):
            st.markdown(st.session_state.current_analysis_output)

    if st.button("🚀 Run AI Analysis", type="primary", key="run_ai_btn", width="stretch"):
        if initialize_agents():
            container = st.container()
            try:
                if analysis_type == "Comprehensive Analysis":
                    # Custom multi-step orchestration for 8B models to avoid tool hallucination
                    st.info("📡 **Orchestrating agents...** (This may take 1-2 minutes)")
                    
                    with st.spinner("🤖 Technical Agent is analyzing chart patterns..."):
                        tech_resp = st.session_state.technical_agent.run(f"Provide a detailed technical analysis for {symbol}.", stream=False)
                    
                    with st.spinner("📊 Fundamental Agent is analyzing financials..."):
                        fund_resp = st.session_state.fundamental_agent.run(f"Provide a detailed fundamental analysis for {symbol}.", stream=False)
                        
                    with st.spinner("⚠️ Risk Agent is evaluating threats..."):
                        risk_resp = st.session_state.risk_agent.run(f"Provide a risk assessment for {symbol}.", stream=False)
                        
                    with st.spinner("📝 Report Agent is synthesizing final investment thesis..."):
                        combined_prompt = f"Create a comprehensive investment report for {symbol} based on these exact agent findings. Do NOT invent new data.\n\n"
                        combined_prompt += f"--- TECHNICAL ANALYSIS ---\n{tech_resp.content}\n\n"
                        combined_prompt += f"--- FUNDAMENTAL ANALYSIS ---\n{fund_resp.content}\n\n"
                        combined_prompt += f"--- RISK ASSESSMENT ---\n{risk_resp.content}"
                        
                        final_resp = st.session_state.report_agent.run(combined_prompt, stream=False)
                        
                    if final_resp and final_resp.content:
                        st.session_state.current_analysis_output = final_resp.content
                        st.session_state.current_analysis_symbol = symbol
                        st.rerun()
                        
                else:
                    # Normal single-agent execution
                    agent = st.session_state.get(agent_key)
                    if agent:
                        with st.spinner(f"{name} is analyzing {symbol}..."):
                            # Using phi's stream=False avoids printing intermediate tool logs when run via st.session_state
                            response = agent.run(
                                f"Provide a detailed {analysis_type.lower()} for {symbol}.",
                                stream=False,
                            )
                            if response and response.content:
                                st.session_state.current_analysis_output = response.content
                                st.session_state.current_analysis_symbol = symbol
                                st.rerun()

            except Exception as e:
                st.error(f"⚠️ AI analysis failed: {e}")
                
            st.session_state.analysis_history.append({
                "symbol": symbol,
                "timestamp": datetime.now(),
                "analysis_type": analysis_type,
                "agent": name if analysis_type != "Comprehensive Analysis" else "Agent Swarm",
            })

    # Quick-fire agent buttons
    st.markdown("---")
    st.markdown("### ⚡ Quick Agent Actions")
    qc1, qc2, qc3 = st.columns(3)
    
    # Placeholder for the quick action output so it expands full width
    quick_action_container = st.empty()
    
    with qc1:
        if st.button("⚠️ Risk Check", key="quick_risk", width="stretch"):
            if initialize_agents():
                with st.spinner("Risk Assessment Agent working..."):
                    try:
                        response = st.session_state.risk_agent.run(
                            f"Quick risk assessment for {symbol}: top 3 risks, risk score 1-10, near-term catalysts.",
                            stream=False,
                        )
                        if response and response.content:
                            with quick_action_container.container():
                                st.markdown("#### ⚠️ Risk Check Results")
                                st.markdown(response.content)
                    except Exception as e:
                        quick_action_container.error(f"⚠️ AI analysis failed: {e}")
    with qc2:
        if st.button("🏭 Sector View", key="quick_sector", width="stretch"):
            if initialize_agents():
                with st.spinner("Sector & Industry Agent working..."):
                    try:
                        response = st.session_state.sector_agent.run(
                            f"Analyze the sector and competitive position of {symbol}.",
                            stream=False,
                        )
                        if response and response.content:
                            with quick_action_container.container():
                                st.markdown("#### 🏭 Sector View Results")
                                st.markdown(response.content)
                    except Exception as e:
                        quick_action_container.error(f"⚠️ AI analysis failed: {e}")
    with qc3:
        if st.button("🔄 Peer Compare", key="quick_compare", width="stretch"):
            if initialize_agents():
                with st.spinner("Stock Comparison Agent working..."):
                    try:
                        response = st.session_state.comparison_agent.run(
                            f"Compare {symbol} against its top 3 sector peers on valuation, growth, and performance.",
                            stream=False,
                        )
                        if response and response.content:
                            with quick_action_container.container():
                                st.markdown("#### 🔄 Peer Compare Results")
                                st.markdown(response.content)
                    except Exception as e:
                        quick_action_container.error(f"⚠️ AI analysis failed: {e}")
    # RAG Query Section
    st.markdown("---")
    st.markdown("### 💬 Ask Any Agent a Question")
    custom_query = st.text_input(
        "Type a custom question for the AI",
        placeholder="e.g. What are the biggest risks for this stock? / Compare with competitors / Summarize public mood",
        key="custom_ai_input",
    )
    cq1, cq2 = st.columns([1, 1])
    with cq1:
        use_rag = st.checkbox("🧠 Include sentiment data (RAG)", value=doc_store.size > 0)
    with cq2:
        query_agent = st.selectbox("Route to agent", ANALYSIS_TYPES, key="query_agent_select")

    if custom_query and st.button("🔍 Ask", key="custom_ask_btn"):
        if initialize_agents():
            target_key = get_agent_for_analysis_type(query_agent)
            target_agent = st.session_state.get(target_key)

            # If RAG is enabled and we have sentiment data, prepend context
            full_query = f"{custom_query} about {symbol}"
            if use_rag and doc_store.size > 0:
                rag_context = query_rag(doc_store, full_query)
                full_query = (
                    f"Context from social media sentiment analysis:\n{rag_context}\n\n"
                    f"User question: {custom_query} about {symbol}"
                )

            with st.spinner(f"{agent_names.get(target_key, ('Agent',))[0]} is thinking..."):
                container = st.container()
                try:
                    response = target_agent.run(full_query, stream=False)
                    if response and response.content:
                        container.markdown(response.content)
                except Exception as e:
                    st.error(f"⚠️ AI analysis failed: {e}")


# ═══════════════════════════════ Tab: Sentiment ═══════════════════════════════

def tab_sentiment(symbol, company_name, doc_store):
    st.markdown("### 🧠 Social Sentiment Analysis")
    st.caption("Powered by VADER (speed) + FinBERT (finance accuracy) — both open-source")

    search_terms = f"{symbol} {company_name}".strip()

    if st.button("🔄 Fetch & Analyze Social Posts", type="primary", width="stretch"):
        # Scrape
        with st.spinner("📡 Scraping Reddit & Twitter/X..."):
            reddit_posts = scrape_reddit(search_terms)
            twitter_posts = scrape_twitter(search_terms)
            all_posts = reddit_posts + twitter_posts

        if not all_posts:
            st.warning("No social media posts found. Try a different stock or check API credentials.")
            return

        st.success(f"Found **{len(all_posts)}** posts ({len(reddit_posts)} Reddit, {len(twitter_posts)} Twitter/X)")

        # Analyze
        with st.spinner("🧪 Running VADER + FinBERT analysis..."):
            analyzed = analyze_sentiment(all_posts)
            summary = get_sentiment_summary(analyzed)

        # Store in RAG document store
        doc_store.clear()
        doc_store.add_documents(analyzed)
        st.session_state.doc_store = doc_store

        # Store results in session for re-rendering
        st.session_state.sentiment_results = analyzed
        st.session_state.sentiment_summary = summary

    # Render results if available
    if "sentiment_results" in st.session_state and st.session_state.sentiment_results:
        analyzed = st.session_state.sentiment_results
        summary = st.session_state.sentiment_summary

        # Overall sentiment badge
        label = summary["label"]
        badge_class = f"sentiment-{label}"
        emoji = "🟢" if label == "positive" else "🔴" if label == "negative" else "🟡"
        st.markdown(
            f'<div style="text-align:center;margin:1rem 0;">'
            f'<span class="sentiment-badge {badge_class}">{emoji} {label.upper()} — '
            f'Avg Score: {summary["avg_score"]:+.3f}</span></div>',
            unsafe_allow_html=True,
        )

        # Charts
        c1, c2 = st.columns(2)
        with c1:
            st.plotly_chart(create_sentiment_gauge(summary["avg_score"], False))
        with c2:
            st.plotly_chart(create_sentiment_pie(summary, False))

        c3, c4 = st.columns(2)
        with c3:
            st.plotly_chart(create_sentiment_timeline(analyzed, False))
        with c4:
            st.plotly_chart(create_source_comparison(analyzed, False))

        # Word cloud (text-based fallback if wordcloud lib not available)
        st.markdown("### ☁️ Key Discussion Topics")
        word_freq = create_word_cloud_data(analyzed)
        if word_freq:
            try:
                from wordcloud import WordCloud
                import matplotlib.pyplot as plt

                wc = WordCloud(
                    width=800, height=300,
                    background_color="white",
                    colormap="cool",
                    max_words=60,
                ).generate_from_frequencies(word_freq)
                fig, ax = plt.subplots(figsize=(10, 3.5))
                ax.imshow(wc, interpolation="bilinear")
                ax.axis("off")
                st.pyplot(fig)
            except ImportError:
                # Text-based fallback
                top_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)[:20]
                st.write(" • ".join(f"**{w}** ({c})" for w, c in top_words))

        # Top posts
        st.markdown("### 📝 Most Significant Posts")
        display_top_posts(analyzed, n=10)

        # AI sentiment interpretation
        st.markdown("---")
        st.markdown("### 🤖 AI Sentiment Interpretation")
        if st.button("Let AI interpret the sentiment data", key="ai_sentiment_interpret"):
            if initialize_agents():
                with st.spinner("Sentiment Intelligence Agent is analyzing social data..."):
                    sentiment_prompt = (
                        f"Analyze the social sentiment for {symbol} ({company_name}). "
                        f"Data: {summary['total']} posts analyzed — "
                        f"{summary['positive_pct']}% positive, {summary['negative_pct']}% negative, "
                        f"{summary['neutral_pct']}% neutral. Average score: {summary['avg_score']:+.3f}. "
                        f"Interpret this sentiment data, identify risks, and predict potential price impact."
                    )
                    container = st.container()
                    try:
                        response = st.session_state.sentiment_agent.run(
                            sentiment_prompt, stream=False,
                        )
                        if response and response.content:
                            container.markdown(response.content)
                    except Exception as e:
                        st.error(f"⚠️ AI analysis failed: {e}")


# ═══════════════════════════════ Tab: News ═══════════════════════════════

def tab_news(info, symbol):
    st.markdown("### 📰 Latest News")
    news = info.get("news") or info.get("companyOfficers", [])  # yfinance structure varies

    if isinstance(news, list) and news:
        for item in news[:8]:
            if isinstance(item, dict) and "title" in item:
                pub_time = ""
                if "providerPublishTime" in item:
                    try:
                        pub_time = datetime.fromtimestamp(item["providerPublishTime"]).strftime("%b %d, %Y %H:%M")
                    except Exception:
                        pub_time = ""
                st.markdown(f"""
                <div class="news-card">
                    <b>{item['title']}</b><br>
                    <small style="color:#888">{item.get('publisher', item.get('source', ''))} {'• ' + pub_time if pub_time else ''}</small>
                </div>
                """, unsafe_allow_html=True)
    else:
        st.info("No recent news from yfinance data feed.")

    # AI-powered news curation
    st.markdown("---")
    st.markdown("### 🤖 AI-Curated News & Analysis")
    if st.button("🔍 Fetch AI-Curated News", key="ai_news_btn", width="stretch"):
        if initialize_agents():
            with st.spinner("News Curator Agent is finding and analyzing news..."):
                container = st.container()
                try:
                    response = st.session_state.news_agent.run(
                        f"Find and analyze the latest news for {symbol} ({info.get('shortName', '')})."
                        f" Categorize by type, rate market impact, and summarize overall news sentiment.",
                        stream=False,
                    )
                    if response and response.content:
                        container.markdown(response.content)
                except Exception as e:
                    st.error(f"⚠️ AI news analysis failed: {e}")


# ═══════════════════════════════ Main ═══════════════════════════════

def main():
    init_session_state()
    analysis_type, market = render_sidebar()
    inject_css()

    # Hero
    st.markdown(f'<div class="hero-header">{APP_ICON} {APP_TITLE}</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="hero-sub">Real-time market data • AI insights • Social sentiment • Agentic RAG</div>',
        unsafe_allow_html=True,
    )

    # Search bar
    c1, c2 = st.columns([3, 1])
    with c1:
        stock_input = st.text_input(
            "Enter Stock Name or Symbol",
            placeholder="e.g. NVIDIA, TCS, TATAMOTORS, AAPL",
        )
    with c2:
        date_range = st.selectbox("Time Range", list(PERIOD_MAP.keys()), index=3)
        period = PERIOD_MAP[date_range]

    if st.button("🚀 Analyze", type="primary", width="stretch"):
        if not stock_input:
            st.error("Please enter a stock name or symbol.")
            return

        symbol = get_symbol_from_name(stock_input)
        if not symbol:
            return

        with st.spinner(f"Loading data for **{symbol}**..."):
            info, hist = get_stock_data(symbol, period)

        if info is None or hist is None:
            st.error(f"Could not fetch data for {symbol}. Please try again.")
            return

        # Store for tabs
        st.session_state.current_symbol = symbol
        st.session_state.current_info = info
        st.session_state.current_hist = hist

        # Clear previous sentiment
        if "sentiment_results" in st.session_state:
            del st.session_state["sentiment_results"]
            del st.session_state["sentiment_summary"]

    # Only render tabs if data is available
    if "current_symbol" not in st.session_state:
        # Landing state
        st.markdown("---")
        st.markdown(
            "<div style='text-align:center;padding:3rem;color:#888;'>"
            "<h3>Enter a stock name above to begin analysis</h3>"
            "<p>Supports US stocks (NVIDIA, AAPL) and Indian stocks (TCS, RELIANCE, TATAMOTORS)</p>"
            "</div>",
            unsafe_allow_html=True,
        )
        return

    symbol = st.session_state.current_symbol
    info = st.session_state.current_info
    hist = st.session_state.current_hist
    doc_store = st.session_state.doc_store

    # Market status
    st.markdown(
        f"<div style='text-align:center;margin:0.5rem 0;color:#888;'>"
        f"📊 <b>{symbol}</b> — {info.get('shortName', symbol)} — "
        f"{info.get('currency', 'USD')}</div>",
        unsafe_allow_html=True,
    )

    # Tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📋 Overview", "📈 Charts", "🤖 AI Analysis", "🧠 Sentiment", "📰 News",
    ])

    with tab1:
        tab_overview(info, hist)
    with tab2:
        tab_charts(hist, symbol)
    with tab3:
        tab_analysis(symbol, analysis_type, doc_store)
    with tab4:
        company_name = info.get("shortName", info.get("longName", ""))
        tab_sentiment(symbol, company_name, doc_store)
    with tab5:
        tab_news(info, symbol)

    # Export section
    st.markdown("---")
    exp_cols = st.columns([1, 1, 2])
    export_data = {
        "info": info,
        "technicals": compute_technical_metrics(hist),
        "sentiment_summary": st.session_state.get("sentiment_summary", {}),
    }
    with exp_cols[0]:
        csv_bytes = export_csv(export_data, symbol)
        st.download_button("📥 Export CSV", csv_bytes, f"{symbol}_report.csv", "text/csv")
    with exp_cols[1]:
        pdf_bytes = export_pdf(export_data, symbol)
        if pdf_bytes:
            # Explicitly cast to bytes to ensure Streamlit accepts it
            pdf_bytes = bytes(pdf_bytes)
            st.download_button("📥 Export PDF", pdf_bytes, f"{symbol}_report.pdf", "application/pdf")

    # Analysis History
    if st.session_state.analysis_history:
        st.markdown("---")
        st.markdown("### 📜 Analysis History")
        history_df = pd.DataFrame(st.session_state.analysis_history)
        history_df["timestamp"] = history_df["timestamp"].dt.strftime("%Y-%m-%d %H:%M")
        st.dataframe(history_df, width="stretch", hide_index=True)


if __name__ == "__main__":
    main()