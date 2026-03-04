# data/market_data.py — Stock data fetching with caching & retry logic

import streamlit as st
import yfinance as yf
import time
from config import COMMON_STOCKS
from typing import Optional

try:
    from thefuzz import fuzz, process as fuzz_process
    FUZZY_AVAILABLE = True
except ImportError:
    FUZZY_AVAILABLE = False


def get_symbol_from_name(stock_name: str) -> Optional[str]:
    """Resolve a company name or ticker to a valid yfinance symbol.

    Resolution order:
    1. Exact match in COMMON_STOCKS
    2. Fuzzy match in COMMON_STOCKS (score ≥ 80)
    3. Direct yfinance lookup
    4. Try .NS (NSE) and .BO (BSE) suffixes
    """
    if not stock_name:
        return None

    cleaned = stock_name.strip().upper()

    # 1) Exact match
    if cleaned in COMMON_STOCKS:
        return COMMON_STOCKS[cleaned]

    # 2) Fuzzy match
    if FUZZY_AVAILABLE:
        match = fuzz_process.extractOne(cleaned, COMMON_STOCKS.keys(), score_cutoff=80)
        if match:
            return COMMON_STOCKS[match[0]]

    # 3) Direct lookup
    if _validate_ticker(cleaned):
        return cleaned

    # 4) Indian market fallback
    for suffix in (".NS", ".BO"):
        candidate = f"{cleaned}{suffix}"
        if _validate_ticker(candidate):
            return candidate

    st.error(f"Could not find a valid symbol for **{stock_name}**")
    return None


def _validate_ticker(symbol: str) -> bool:
    """Check if a ticker returns valid data from yfinance."""
    try:
        info = yf.Ticker(symbol).info
        return bool(info and info.get("symbol"))
    except Exception:
        return False


@st.cache_data(ttl=300, show_spinner=False)
def get_stock_data(symbol: str, period: str = "1y"):
    """Fetch stock info + OHLCV history with retry logic.

    Returns (info_dict, history_dataframe) or (None, None) on failure.
    """
    max_retries = 3

    for attempt in range(max_retries):
        try:
            # Note: Do not inject custom session; yfinance requires its own curl_cffi session
            stock = yf.Ticker(symbol)
            info = stock.info
            if not info:
                raise ValueError("Empty info response")

            hist = stock.history(period=period, interval="1d", auto_adjust=True)
            if hist.empty:
                raise ValueError("No historical data available")

            return info, hist

        except Exception as e:
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)  # exponential backoff: 1s, 2s, 4s
            else:
                st.error(f"Error fetching data for **{symbol}** after {max_retries} retries: {e}")
                return None, None


@st.cache_data(ttl=600, show_spinner=False)
def get_options_chain(symbol: str):
    """Fetch options chain for the nearest expiration date."""
    try:
        stock = yf.Ticker(symbol)
        expirations = stock.options
        if not expirations:
            return None, None, []
        calls = stock.option_chain(expirations[0]).calls
        puts = stock.option_chain(expirations[0]).puts
        return calls, puts, list(expirations)
    except Exception:
        return None, None, []


@st.cache_data(ttl=600, show_spinner=False)
def get_institutional_holders(symbol: str):
    """Get institutional + major holders data."""
    try:
        stock = yf.Ticker(symbol)
        institutional = stock.institutional_holders
        major = stock.major_holders
        return institutional, major
    except Exception:
        return None, None


def format_large_number(number, currency="$"):
    """Format large numbers into human-readable strings."""
    if number is None or number == "N/A":
        return "N/A"
    try:
        number = float(number)
    except (ValueError, TypeError):
        return "N/A"

    if number >= 1e12:
        return f"{currency}{number / 1e12:.2f}T"
    elif number >= 1e9:
        return f"{currency}{number / 1e9:.2f}B"
    elif number >= 1e6:
        return f"{currency}{number / 1e6:.2f}M"
    else:
        return f"{currency}{number:,.2f}"
