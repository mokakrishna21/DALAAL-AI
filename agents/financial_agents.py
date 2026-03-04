# agents/financial_agents.py — Full Agentic AI Roster for DALAAL AI
# 10+ specialized agents, all powered by open-source LLaMA 3.1 via Groq
#
# Agent Architecture:
#  ┌─────────────────────────────────────────────────────────────┐
#  │                   DALAAL AI MASTER ORCHESTRATOR             │
#  │  Delegates to specialized sub-agents based on query type    │
#  ├─────────────────────────────────────────────────────────────┤
#  │                                                             │
#  │  TIER 1 — Data Agents (fetch raw information)               │
#  │  ├── 📊 Market Data Agent (yfinance prices & fundamentals)  │
#  │  ├── 🌐 Web Search Agent (DuckDuckGo + Google)             │
#  │  └── 📰 News Agent (curated financial news)                │
#  │                                                             │
#  │  TIER 2 — Analysis Agents (process & interpret)             │
#  │  ├── 📈 Technical Analysis Agent                            │
#  │  ├── 📋 Fundamental Analysis Agent                          │
#  │  ├── ⚠️  Risk Assessment Agent                              │
#  │  ├── 🧠 Sentiment Analysis Agent                            │
#  │  └── 🏛️  Institutional & Insider Agent                      │
#  │                                                             │
#  │  TIER 3 — Synthesis Agents (combine & generate)             │
#  │  ├── 🔄 Stock Comparison Agent                              │
#  │  ├── 📝 Report Generator Agent                              │
#  │  └── 🏭 Sector & Industry Agent                             │
#  │                                                             │
#  └─────────────────────────────────────────────────────────────┘

import streamlit as st
import importlib
import config
from phi.agent.agent import Agent
from phi.model.groq import Groq
from phi.tools.yfinance import YFinanceTools
from phi.tools.duckduckgo import DuckDuckGo
from phi.tools.googlesearch import GoogleSearch

import config


def _get_model():
    """Get a fresh Groq model instance."""
    return Groq(id=config.LLM_MODEL_ID, api_key=config.get_groq_api_key())


# ═══════════════════════════════════════════════════════════════════
#  TIER 1 — DATA AGENTS (fetch raw information)
# ═══════════════════════════════════════════════════════════════════

def _create_market_data_agent() -> Agent:
    """Agent #1: Market Data Agent — fetches stock prices, fundamentals, and history."""
    return Agent(
        name="Market Data Agent",
        role="Fetch real-time and historical stock market data",
        model=_get_model(),
        tools=[
            YFinanceTools(
                stock_price=True,
                company_news=True,
                analyst_recommendations=True,
                historical_prices=True,
                company_info=True,
                stock_fundamentals=True,
            )
        ],
        instructions=[
            "You are a market data specialist. Your job is to fetch accurate, real-time stock data.",
            "Always retrieve: current price, market cap, P/E, 52-week range, volume, and EPS.",
            "Include analyst recommendations and price targets when available.",
            "Format numbers clearly with commas and appropriate decimals.",
            "If data is unavailable, state 'Data not available' rather than guessing.",
        ],
        show_tool_calls=True,
        markdown=True,
    )


def _create_web_search_agent() -> Agent:
    """Agent #2: Web Search Agent — searches the web for financial news and information."""
    return Agent(
        name="Web Search Agent",
        role="Search the web for latest financial news, market events, and analyst opinions",
        model=_get_model(),
        tools=[
            GoogleSearch(fixed_language="english", fixed_max_results=5),
            DuckDuckGo(fixed_max_results=5),
        ],
        instructions=[
            "You are a financial research agent specializing in web intelligence.",
            "Search for the most recent and relevant financial news and market events.",
            "Cross-reference information from multiple sources for accuracy.",
            "Always cite your sources with URLs.",
            "Prioritize: earnings reports, analyst upgrades/downgrades, regulatory news, M&A activity.",
            "Flag any breaking news or market-moving events prominently.",
        ],
        show_tool_calls=True,
        markdown=True,
    )


def _create_news_agent() -> Agent:
    """Agent #3: News Curator Agent — curates and analyzes financial news."""
    return Agent(
        name="News Curator Agent",
        role="Curate, filter, and analyze financial news for relevance and impact",
        model=_get_model(),
        tools=[
            DuckDuckGo(fixed_max_results=8),
            GoogleSearch(fixed_language="english", fixed_max_results=5),
            YFinanceTools(company_news=True),
        ],
        instructions=[
            "You are a financial news curator and analyst.",
            "Find and organize the latest news about the specified stock or company.",
            "Categorize news by type: Earnings, Analysts, Regulatory, Product, Legal, Market Trends.",
            "Rate each news item's potential market impact: High / Medium / Low.",
            "Provide a brief analysis of how each major news item could affect the stock price.",
            "Summarize the overall news sentiment: Bullish / Bearish / Mixed.",
            "Include publication dates and sources for all news items.",
        ],
        show_tool_calls=True,
        markdown=True,
    )


# ═══════════════════════════════════════════════════════════════════
#  TIER 2 — ANALYSIS AGENTS (process & interpret)
# ═══════════════════════════════════════════════════════════════════

def _create_technical_analysis_agent() -> Agent:
    """Agent #4: Technical Analysis Agent — analyzes chart patterns and indicators."""
    return Agent(
        name="Technical Analysis Agent",
        role="Expert technical analyst interpreting chart patterns, indicators, and price action",
        model=_get_model(),
        tools=[
            YFinanceTools(stock_price=True, historical_prices=True),
        ],
        instructions=[
            "You are a senior technical analyst with 20 years of experience.",
            "Analyze the stock using these indicators: RSI, MACD, Bollinger Bands, Moving Averages (20/50/200), Volume, Stochastic.",
            "Identify chart patterns: Head & Shoulders, Double Top/Bottom, Triangles, Flags, Channels.",
            "Determine support and resistance levels based on recent price action.",
            "Provide clear entry and exit zones based on your analysis.",
            "Rate the technical setup: Strong Buy / Buy / Hold / Sell / Strong Sell.",
            "Include timeframe for your analysis (short-term, medium-term, long-term).",
            "Warn about any conflicting signals between indicators.",
        ],
        show_tool_calls=True,
        markdown=True,
    )


def _create_fundamental_analysis_agent() -> Agent:
    """Agent #5: Fundamental Analysis Agent — deep-dives into company financials."""
    return Agent(
        name="Fundamental Analysis Agent",
        role="Expert fundamental analyst evaluating company financials, valuation, and growth prospects",
        model=_get_model(),
        tools=[
            YFinanceTools(
                stock_price=True,
                stock_fundamentals=True,
                company_info=True,
                analyst_recommendations=True,
            ),
        ],
        instructions=[
            "You are a CFA-chartered fundamental analyst.",
            "Evaluate the company across these dimensions:",
            "1. VALUATION: P/E, P/B, P/S, PEG ratio, EV/EBITDA — compare to sector averages.",
            "2. PROFITABILITY: Gross margin, operating margin, net margin, ROE, ROA — trend analysis.",
            "3. GROWTH: Revenue growth YoY, EPS growth, forward estimates vs historical.",
            "4. FINANCIAL HEALTH: Debt-to-equity, current ratio, interest coverage, free cash flow.",
            "5. DIVIDEND: Yield, payout ratio, dividend growth history.",
            "Provide a fair value estimate using DCF or comparable analysis.",
            "Determine if the stock is: Undervalued / Fairly Valued / Overvalued.",
            "Highlight any red flags in the financial statements.",
        ],
        show_tool_calls=True,
        markdown=True,
    )


def _create_risk_assessment_agent() -> Agent:
    """Agent #6: Risk Assessment Agent — evaluates risks and provides risk scores."""
    return Agent(
        name="Risk Assessment Agent",
        role="Risk specialist evaluating investment risks across multiple dimensions",
        model=_get_model(),
        tools=[
            YFinanceTools(stock_price=True, stock_fundamentals=True, company_info=True),
            DuckDuckGo(fixed_max_results=5),
        ],
        instructions=[
            "You are a risk management specialist.",
            "Evaluate the stock across these risk categories:",
            "1. MARKET RISK: Beta, volatility, correlation with market indices, drawdown history.",
            "2. FINANCIAL RISK: Leverage, liquidity ratios, debt maturity, cash burn rate.",
            "3. BUSINESS RISK: Competitive position, market share trends, regulatory exposure.",
            "4. GEOPOLITICAL RISK: Geographic revenue concentration, trade policy impact, currency risk.",
            "5. ESG RISK: Environmental lawsuits, governance issues, social controversies.",
            "Provide an overall risk score: 1 (very low) to 10 (very high).",
            "Calculate Value at Risk (VaR) at 95% confidence for 1-day and 1-month horizons.",
            "Suggest hedging strategies for identified risks.",
            "Flag any near-term risk catalysts (earnings, FDA decisions, lawsuits, etc.).",
        ],
        show_tool_calls=True,
        markdown=True,
    )


def _create_sentiment_analysis_agent() -> Agent:
    """Agent #7: AI Sentiment Agent — interprets sentiment data and social media trends."""
    return Agent(
        name="Sentiment Intelligence Agent",
        role="Sentiment specialist analyzing public opinion, social media trends, and market mood",
        model=_get_model(),
        tools=[
            DuckDuckGo(fixed_max_results=8),
            GoogleSearch(fixed_language="english", fixed_max_results=5),
        ],
        instructions=[
            "You are a sentiment analysis specialist focused on financial markets.",
            "Analyze the current public sentiment around the specified stock by searching for:",
            "1. Social media discussions and trending topics.",
            "2. Retail investor sentiment on forums (Reddit, StockTwits, Twitter/X).",
            "3. Fear & Greed indicators and put/call ratios.",
            "4. Short interest and short squeeze potential.",
            "5. Institutional sentiment based on recent fund flows and 13F filings.",
            "Rate overall sentiment on a scale: Very Bearish → Bearish → Neutral → Bullish → Very Bullish.",
            "Identify any sentiment divergence (e.g., retail bullish but institutions selling).",
            "Flag potential sentiment-driven events (meme stock activity, short squeeze setups).",
            "Note if sentiment is leading or lagging the price action.",
        ],
        show_tool_calls=True,
        markdown=True,
    )


def _create_institutional_agent() -> Agent:
    """Agent #8: Institutional & Insider Agent — tracks smart money movements."""
    return Agent(
        name="Institutional & Insider Agent",
        role="Track institutional holdings, insider trades, and smart money movements",
        model=_get_model(),
        tools=[
            YFinanceTools(
                stock_price=True,
                stock_fundamentals=True,
                company_info=True,
            ),
            DuckDuckGo(fixed_max_results=5),
        ],
        instructions=[
            "You are an institutional flow analyst tracking smart money.",
            "Research and report on:",
            "1. TOP INSTITUTIONAL HOLDERS: Who owns the most shares? Any recent changes?",
            "2. INSIDER TRADING: Recent insider buys and sells — are insiders bullish or bearish?",
            "3. FUND FLOWS: Are mutual funds and ETFs adding or reducing positions?",
            "4. ACTIVIST INVESTORS: Any activist involvement or campaigns?",
            "5. SHORT INTEREST: Current short interest as % of float, days to cover.",
            "Highlight any unusual institutional activity that could signal upcoming moves.",
            "Note the ownership concentration — is it widely held or controlled by a few?",
        ],
        show_tool_calls=True,
        markdown=True,
    )


# ═══════════════════════════════════════════════════════════════════
#  TIER 3 — SYNTHESIS AGENTS (combine & generate)
# ═══════════════════════════════════════════════════════════════════

def _create_comparison_agent() -> Agent:
    """Agent #9: Comparison Agent — compares stocks against peers and benchmarks."""
    return Agent(
        name="Stock Comparison Agent",
        role="Compare stocks against sector peers and market benchmarks",
        model=_get_model(),
        tools=[
            YFinanceTools(
                stock_price=True,
                stock_fundamentals=True,
                company_info=True,
            ),
        ],
        instructions=[
            "You are a comparative analysis specialist.",
            "When given a stock, identify its top 3-5 sector peers and compare them across:",
            "1. VALUATION: P/E, P/S, EV/EBITDA relative to peers.",
            "2. GROWTH: Revenue and EPS growth vs peers.",
            "3. PROFITABILITY: Margins, ROE, ROA vs peers.",
            "4. PRICE PERFORMANCE: YTD, 1-year, 3-year returns vs peers and the market (S&P 500 or NIFTY 50).",
            "5. DIVIDEND: Yield and payout vs peers.",
            "Present results in a clear comparison table.",
            "Conclude with positioning recommendation: which peer looks most attractive and why.",
        ],
        show_tool_calls=True,
        markdown=True,
    )


def _create_sector_industry_agent() -> Agent:
    """Agent #10: Sector & Industry Agent — analyzes sector trends and positioning."""
    return Agent(
        name="Sector & Industry Agent",
        role="Analyze sector trends, industry dynamics, and competitive landscape",
        model=_get_model(),
        tools=[
            DuckDuckGo(fixed_max_results=8),
            GoogleSearch(fixed_language="english", fixed_max_results=5),
        ],
        instructions=[
            "You are a sector and industry analyst.",
            "For the stock's sector, analyze:",
            "1. SECTOR TRENDS: Is the sector in growth, maturity, or decline? What are the tailwinds and headwinds?",
            "2. MARKET DYNAMICS: Market size, growth rate, competitive intensity (use Porter's Five Forces).",
            "3. REGULATORY ENVIRONMENT: Key regulations, upcoming policy changes, compliance requirements.",
            "4. TECHNOLOGY DISRUPTION: AI, automation, or other tech trends impacting the industry.",
            "5. COMPANY POSITIONING: Where does the company sit in the competitive landscape? Market share trends.",
            "6. FUTURE OUTLOOK: 1-year and 3-year sector outlook with catalysts.",
            "Use recent data and cite sources for your sector analysis.",
        ],
        show_tool_calls=True,
        markdown=True,
    )


def _create_report_generator_agent(team: list[Agent]) -> Agent:
    """Agent #11: Report Generator Agent — synthesizes all agent outputs into a report."""
    return Agent(
        name="Report Generator Agent",
        role="Synthesize analysis from all specialist agents into a comprehensive investment report",
        model=_get_model(),
        team=team,
        instructions=[
            "You are a senior investment analyst creating a comprehensive stock report.",
            "Compile insights from ALL team members into a structured report with these sections:",
            "",
            "## Executive Summary",
            "One-paragraph verdict: Buy / Hold / Sell — with conviction level (High/Medium/Low).",
            "",
            "## Key Metrics Dashboard",
            "Table of critical numbers: Price, Market Cap, P/E, EPS, Revenue Growth, Margins.",
            "",
            "## Technical Analysis",
            "Chart patterns, indicator signals, support/resistance, trend direction.",
            "",
            "## Fundamental Analysis",
            "Valuation assessment, financial health, growth trajectory.",
            "",
            "## Risk Assessment",
            "Top 3 risks with severity ratings, mitigation factors.",
            "",
            "## Sentiment & Social Intelligence",
            "Public mood, institutional positioning, social media trends.",
            "",
            "## News & Catalysts",
            "Key upcoming events, recent developments, potential catalysts.",
            "",
            "## Competitive Position",
            "Peer comparison, market share, sector positioning.",
            "",
            "## Price Target & Outlook",
            "12-month price target with bull/base/bear scenarios.",
            "",
            "Be data-driven. Cite specific numbers. Avoid generic statements.",
            "If agents disagree, highlight the divergence and explain why.",
        ],
        show_tool_calls=True,
        markdown=True,
    )


# ═══════════════════════════════════════════════════════════════════
#  MASTER ORCHESTRATOR
# ═══════════════════════════════════════════════════════════════════

def _create_master_orchestrator(team: list[Agent]) -> Agent:
    """Agent #12: DALAAL AI Master Orchestrator — routes queries to best agents."""
    return Agent(
        name="DALAAL AI Master Orchestrator",
        role="Intelligent query router that delegates to specialized financial agents",
        model=_get_model(),
        team=team,
        instructions=[
            "You are the DALAAL AI Master Orchestrator — a superintelligent financial AI.",
            "You lead a team of specialized agents. Route each query to the MOST relevant agents.",
            "",
            "ROUTING RULES:",
            "- Technical questions → Technical Analysis Agent",
            "- Valuation/financials → Fundamental Analysis Agent",
            "- Risk concerns → Risk Assessment Agent",
            "- Public opinion/social media → Sentiment Intelligence Agent",
            "- News/events → News Curator Agent",
            "- Peer comparison → Stock Comparison Agent",
            "- Sector trends → Sector & Industry Agent",
            "- Insider/institutional → Institutional & Insider Agent",
            "- Comprehensive analysis → Delegate to ALL agents, then synthesize",
            "",
            "For comprehensive queries, combine insights from multiple agents.",
            "Always structure your response clearly with headers and sections.",
            "Highlight any CONFLICTS between agents' conclusions.",
            "End every response with a clear, actionable summary.",
        ],
        show_tool_calls=True,
        markdown=True,
    )


# ═══════════════════════════════════════════════════════════════════
#  INITIALIZATION
# ═══════════════════════════════════════════════════════════════════

def initialize_agents() -> bool:
    """Initialize the full multi-agent system (12 agents).

    Agent hierarchy:
        Master Orchestrator
        ├── Market Data Agent
        ├── Web Search Agent
        ├── News Curator Agent
        ├── Technical Analysis Agent
        ├── Fundamental Analysis Agent
        ├── Risk Assessment Agent
        ├── Sentiment Intelligence Agent
        ├── Institutional & Insider Agent
        ├── Stock Comparison Agent
        ├── Sector & Industry Agent
        └── Report Generator Agent (has access to all above)
    """
    # Force reload config to ensure we have the latest model ID
    importlib.reload(config)
    
    current_model = config.LLM_MODEL_ID
    stored_model = st.session_state.get("model_id")
    
    # Log for debugging (visible in terminal logs)
    print(f"DEBUG: initialize_agents called. Stored model: {stored_model}, Config model: {current_model}")

    if st.session_state.get("agents_initialized") and stored_model == current_model:
        # Extra safety check: peek at one agent to ensure it's not a zombie with the wrong ID
        test_agent = st.session_state.get("multi_ai_agent")
        if test_agent and hasattr(test_agent, "model") and hasattr(test_agent.model, "id"):
            if test_agent.model.id == current_model:
                return True
            else:
                print(f"DEBUG: Agent model mismatch! Stored: {test_agent.model.id}, Config: {current_model}")
        else:
            return True # Assume OK if we can't check, the logic below will fix it anyway if called

    # If we are here, we are re-initializing. Clear old agents explicitly.
    agent_keys = [
        "market_data_agent", "web_agent", "news_agent", "technical_agent", 
        "fundamental_agent", "risk_agent", "sentiment_agent", "institutional_agent",
        "comparison_agent", "sector_agent", "report_agent", "multi_ai_agent"
    ]
    for k in agent_keys:
        if k in st.session_state:
            del st.session_state[k]

    api_key = config.get_groq_api_key()
    if not api_key:
        st.error("⚠️ GROQ_API_KEY not configured. Add it to `.streamlit/secrets.toml` or environment variables.")
        return False

    try:
        # Tier 1 — Data Agents
        market_data_agent = _create_market_data_agent()
        web_search_agent = _create_web_search_agent()
        news_agent = _create_news_agent()

        # Tier 2 — Analysis Agents
        technical_agent = _create_technical_analysis_agent()
        fundamental_agent = _create_fundamental_analysis_agent()
        risk_agent = _create_risk_assessment_agent()
        sentiment_agent = _create_sentiment_analysis_agent()
        institutional_agent = _create_institutional_agent()

        # Tier 3 — Synthesis Agents
        comparison_agent = _create_comparison_agent()
        sector_agent = _create_sector_industry_agent()

        # All specialist agents
        all_specialists = [
            market_data_agent, web_search_agent, news_agent,
            technical_agent, fundamental_agent, risk_agent,
            sentiment_agent, institutional_agent,
            comparison_agent, sector_agent,
        ]

        # Report Generator (has the full team)
        report_agent = _create_report_generator_agent(all_specialists)

        # Master Orchestrator (routes to all agents including report generator)
        master = _create_master_orchestrator(all_specialists + [report_agent])

        # Store all agents in session state for direct access from tabs
        st.session_state.market_data_agent = market_data_agent
        st.session_state.web_agent = web_search_agent
        st.session_state.news_agent = news_agent
        st.session_state.technical_agent = technical_agent
        st.session_state.fundamental_agent = fundamental_agent
        st.session_state.risk_agent = risk_agent
        st.session_state.sentiment_agent = sentiment_agent
        st.session_state.institutional_agent = institutional_agent
        st.session_state.comparison_agent = comparison_agent
        st.session_state.sector_agent = sector_agent
        st.session_state.report_agent = report_agent
        st.session_state.multi_ai_agent = master
        st.session_state.model_id = current_model
        st.session_state.agents_initialized = True
        print(f"DEBUG: Agents initialized successfully with model: {current_model}")
        return True

    except Exception as e:
        st.error(f"Error initializing agents: {e}")
        return False


def get_agent_for_analysis_type(analysis_type: str) -> str:
    """Map analysis type selection to the appropriate agent key in session state."""
    mapping = {
        "Comprehensive Analysis": "multi_ai_agent",
        "Technical Analysis": "technical_agent",
        "Fundamental Analysis": "fundamental_agent",
        "News Analysis": "news_agent",
        "Sentiment Analysis": "sentiment_agent",
        "Risk Assessment": "risk_agent",
        "Peer Comparison": "comparison_agent",
        "Sector Analysis": "sector_agent",
        "Institutional Analysis": "institutional_agent",
        "Full Investment Report": "report_agent",
    }
    return mapping.get(analysis_type, "multi_ai_agent")


# All available analysis types for the sidebar
ANALYSIS_TYPES = [
    "Comprehensive Analysis",
    "Technical Analysis",
    "Fundamental Analysis",
    "Sentiment Analysis",
    "Risk Assessment",
    "News Analysis",
    "Peer Comparison",
    "Sector Analysis",
    "Institutional Analysis",
    "Full Investment Report",
]
