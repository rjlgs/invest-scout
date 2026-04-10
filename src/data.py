"""
Data fetching and caching for invest-scout.

Pulls OHLCV (Open, High, Low, Close, Volume) data from yfinance with
local CSV caching to avoid hammering the API on repeated scans.
"""

import os
import time
from pathlib import Path

import pandas as pd
import yaml
import yfinance as yf
from dotenv import load_dotenv

load_dotenv()

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
CACHE_DIR = Path(__file__).resolve().parent.parent / "data_cache"
CACHE_TTL_MINUTES = int(os.getenv("CACHE_TTL_MINUTES", "60"))
DATA_PERIOD = os.getenv("DATA_PERIOD", "1y")


def _ensure_cache_dir() -> None:
    """Create the cache directory if it doesn't exist."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)


def _cache_path(ticker: str, period: str) -> Path:
    """Return the local CSV path for a cached ticker."""
    return CACHE_DIR / f"{ticker.upper()}_{period}.csv"


def _cache_is_fresh(path: Path) -> bool:
    """Check whether a cached file is younger than CACHE_TTL_MINUTES."""
    if not path.exists():
        return False
    age_seconds = time.time() - path.stat().st_mtime
    return age_seconds < CACHE_TTL_MINUTES * 60


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def fetch_ohlcv(ticker: str, period: str = DATA_PERIOD,
                interval: str = "1d") -> pd.DataFrame:
    """
    Fetch OHLCV data from yfinance.

    Parameters
    ----------
    ticker : str
        Stock or ETF ticker symbol (e.g. "AAPL").
    period : str
        How far back to fetch — e.g. "1y", "6mo", "2y".
    interval : str
        Candle size — "1d" for daily (our default for swing trading).

    Returns
    -------
    pd.DataFrame
        Columns: Open, High, Low, Close, Volume with a DatetimeIndex.
    """
    tk = yf.Ticker(ticker)
    df = tk.history(period=period, interval=interval)

    if df.empty:
        raise ValueError(f"No data returned for {ticker} (period={period})")

    # yfinance sometimes includes extra columns (Dividends, Stock Splits).
    # Keep only what we need.
    keep = [c for c in ["Open", "High", "Low", "Close", "Volume"] if c in df.columns]
    return df[keep]


def get_cached_data(ticker: str, period: str = DATA_PERIOD) -> pd.DataFrame:
    """
    Return OHLCV data, serving from cache when possible.

    If the cache file exists and is fresher than CACHE_TTL_MINUTES, it is
    read directly. Otherwise fresh data is fetched and the cache is updated.
    """
    _ensure_cache_dir()
    path = _cache_path(ticker, period)

    if _cache_is_fresh(path):
        df = pd.read_csv(path, index_col=0, parse_dates=True)
        return df

    df = fetch_ohlcv(ticker, period=period)
    df.to_csv(path)
    return df


def fetch_historical(ticker: str, start: str, end: str) -> pd.DataFrame:
    """
    Fetch historical data for a specific date range (used by backtesting).

    Parameters
    ----------
    ticker : str
        Ticker symbol.
    start : str
        Start date in YYYY-MM-DD format.
    end : str
        End date in YYYY-MM-DD format.

    Returns
    -------
    pd.DataFrame
        OHLCV data for the requested range.
    """
    tk = yf.Ticker(ticker)
    df = tk.history(start=start, end=end, interval="1d")

    if df.empty:
        raise ValueError(
            f"No data returned for {ticker} ({start} to {end})"
        )

    keep = [c for c in ["Open", "High", "Low", "Close", "Volume"] if c in df.columns]
    return df[keep]


# ---------------------------------------------------------------------------
# Watchlist loading
# ---------------------------------------------------------------------------

def load_watchlist(path: str = None) -> list[dict]:
    """
    Parse the watchlist YAML file and return a list of ticker configs.

    Each entry has at minimum a ``symbol`` key. Optional per-ticker
    overrides (e.g. ``trailing_stop_pct``, ``rsi_oversold``) are merged
    with the global defaults from .env at signal-evaluation time.

    Parameters
    ----------
    path : str, optional
        Path to watchlist YAML. Defaults to ``config/watchlist.yaml``
        relative to the project root.

    Returns
    -------
    list[dict]
        E.g. [{"symbol": "AAPL"}, {"symbol": "TSLA", "trailing_stop_pct": 12.0}]
    """
    if path is None:
        path = Path(__file__).resolve().parent.parent / "config" / "watchlist.yaml"

    with open(path, "r") as f:
        data = yaml.safe_load(f)

    tickers = data.get("tickers", [])
    if not tickers:
        raise ValueError(f"No tickers found in {path}")

    return tickers
