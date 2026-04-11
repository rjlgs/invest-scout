#!/usr/bin/env python3
"""
Generate static JSON data for GitHub Pages deployment.

This script is run by GitHub Actions daily after market close.
It scans all tickers in master_tickers.yaml and generates:
  - data/signals.json       - Signal results for all tickers
  - data/tickers/{SYM}.json - Per-ticker detail (OHLCV, indicators, history)
  - data/meta.json          - Metadata (last update, available tickers)
"""

import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd

# Add project root to path for imports
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data import fetch_historical, load_watchlist
from src.signals import compute_indicators, evaluate_signals
from src.scanner import scan_watchlist

DATA_DIR = PROJECT_ROOT / "data"
TICKERS_DIR = DATA_DIR / "tickers"


def generate_ticker_detail(symbol: str, days: int = 365) -> dict | None:
    """
    Generate detailed data for a single ticker including OHLCV and indicators.

    Returns a dict with:
      - ohlcv: list of {date, open, high, low, close, volume}
      - indicators: {rsi, sma50, sma200, macd, macd_signal, macd_hist}
      - history: last 30 days of signal evaluations
    """
    try:
        # Fetch 1 year of data for charts and backtesting
        end = datetime.now()
        start = end - timedelta(days=days)

        df = fetch_historical(
            symbol,
            start.strftime("%Y-%m-%d"),
            end.strftime("%Y-%m-%d")
        )

        if df is None or df.empty:
            print(f"  [WARN] No data for {symbol}")
            return None

        # Compute indicators
        df = compute_indicators(df)

        # Build OHLCV list (most recent 252 trading days max)
        df_recent = df.tail(252)
        ohlcv = []
        for idx, row in df_recent.iterrows():
            ohlcv.append({
                "date": str(idx.date()),
                "open": round(float(row["Open"]), 2),
                "high": round(float(row["High"]), 2),
                "low": round(float(row["Low"]), 2),
                "close": round(float(row["Close"]), 2),
                "volume": int(row["Volume"]),
            })

        # Build indicators (aligned with OHLCV)
        # Include v2 indicators: ema_fast, bb_*, atr, adx, +/-DI, stoch
        def _round_col(col, ndigits):
            if col not in df_recent.columns:
                return [None] * len(df_recent)
            return [round(v, ndigits) if not pd.isna(v) else None for v in df_recent[col].tolist()]

        indicators = {
            "rsi": _round_col("rsi", 2),
            "sma50": _round_col("sma_short", 2),
            "sma200": _round_col("sma_long", 2),
            "macd": _round_col("macd", 4),
            "macd_signal": _round_col("macd_signal", 4),
            "macd_hist": _round_col("macd_hist", 4),
            # v2 indicators
            "ema20": _round_col("ema_fast", 2),
            "bb_upper": _round_col("bb_upper", 2),
            "bb_middle": _round_col("bb_middle", 2),
            "bb_lower": _round_col("bb_lower", 2),
            "bb_bandwidth": _round_col("bb_bandwidth", 4),
            "atr": _round_col("atr", 4),
            "adx": _round_col("adx", 2),
            "plus_di": _round_col("plus_di", 2),
            "minus_di": _round_col("minus_di", 2),
            "stoch_k": _round_col("stoch_k", 2),
            "stoch_d": _round_col("stoch_d", 2),
            "vol_avg_20": _round_col("vol_avg_20", 2),
        }

        # Generate signal history for last 30 trading days
        history = []
        lookback = min(30, len(df) - 200)  # Need 200 days warmup
        for i in range(lookback, 0, -1):
            window = df.iloc[:-i] if i > 0 else df
            if len(window) < 200:
                continue
            try:
                result = evaluate_signals(window, symbol)
                history.append({
                    "date": str(df.index[-i].date()),
                    "signal_type": result["signal_type"],
                    "on_ramp_score": result["on_ramp_score"],
                    "off_ramp_score": result["off_ramp_score"],
                    "signals": [s["name"] for s in result["on_ramp_signals"] + result["off_ramp_signals"]],
                })
            except Exception:
                pass

        return {
            "symbol": symbol.upper(),
            "last_updated": datetime.now(timezone.utc).isoformat(),
            "ohlcv": ohlcv,
            "indicators": indicators,
            "history": history,
        }

    except Exception as e:
        print(f"  [ERROR] {symbol}: {e}")
        return None


def main():
    print("=" * 60)
    print("Invest-Scout Static Data Generator")
    print("=" * 60)

    # Ensure directories exist
    DATA_DIR.mkdir(exist_ok=True)
    TICKERS_DIR.mkdir(exist_ok=True)

    # Load master ticker list
    master_path = PROJECT_ROOT / "config" / "master_tickers.yaml"
    tickers = load_watchlist(str(master_path))
    symbols = [t["symbol"] for t in tickers]

    print(f"\nScanning {len(symbols)} tickers...")

    # Step 1: Generate signals.json using existing scanner
    print("\n[1/3] Generating signals.json...")
    results = scan_watchlist(str(master_path))

    signals_path = DATA_DIR / "signals.json"
    with open(signals_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"  Wrote {signals_path} ({len(results)} tickers)")

    # Step 2: Generate per-ticker detail files
    print("\n[2/3] Generating ticker detail files...")
    available_tickers = []

    for i, symbol in enumerate(symbols, 1):
        print(f"  [{i}/{len(symbols)}] {symbol}...", end=" ", flush=True)
        detail = generate_ticker_detail(symbol)

        if detail:
            ticker_path = TICKERS_DIR / f"{symbol}.json"
            with open(ticker_path, "w") as f:
                json.dump(detail, f)
            available_tickers.append(symbol)
            print("OK")
        else:
            print("SKIP")

    # Step 3: Generate meta.json
    print("\n[3/3] Generating meta.json...")
    meta = {
        "last_updated": datetime.now(timezone.utc).isoformat(),
        "ticker_count": len(available_tickers),
        "tickers": sorted(available_tickers),
    }

    meta_path = DATA_DIR / "meta.json"
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"  Wrote {meta_path}")

    # Summary
    print("\n" + "=" * 60)
    print(f"Done! Generated data for {len(available_tickers)} tickers.")
    print(f"  - {signals_path}")
    print(f"  - {TICKERS_DIR}/*.json ({len(available_tickers)} files)")
    print(f"  - {meta_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
