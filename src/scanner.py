"""
Watchlist scanner for invest-scout.

Iterates through every ticker in the watchlist, fetches data, computes
indicators, evaluates signals, and writes the results to signals.json.
"""

import json
import logging
import os
import threading
import time
from pathlib import Path

from dotenv import load_dotenv

from .data import get_cached_data, load_watchlist
from .signals import (
    MAX_OFF_RAMP_SCORE,
    MAX_ON_RAMP_SCORE,
    compute_indicators,
    evaluate_signals,
)

load_dotenv()

logger = logging.getLogger("invest-scout.scanner")

SCAN_INTERVAL = int(os.getenv("SCAN_INTERVAL_MINUTES", "5"))
SIGNALS_PATH = Path(__file__).resolve().parent.parent / "signals.json"


def scan_watchlist(watchlist_path: str | None = None) -> list[dict]:
    """
    Scan every ticker in the watchlist and return signal results.

    Parameters
    ----------
    watchlist_path : str, optional
        Path to the YAML watchlist. Defaults to config/watchlist.yaml.

    Returns
    -------
    list[dict]
        One result dict per ticker (see signals.evaluate_signals).
    """
    tickers = load_watchlist(watchlist_path)
    results = []

    for entry in tickers:
        symbol = entry["symbol"]
        # Per-ticker config overrides (everything except 'symbol')
        config = {k: v for k, v in entry.items() if k != "symbol"}

        try:
            df = get_cached_data(symbol)
            df = compute_indicators(df, config if config else None)
            result = evaluate_signals(df, symbol, config if config else None)
            results.append(result)
            logger.info(
                "%s: %s (on-ramp=%d, off-ramp=%d)",
                symbol, result["signal_type"],
                result["on_ramp_score"], result["off_ramp_score"],
            )
        except Exception as e:
            logger.error("Failed to scan %s: %s", symbol, e)
            results.append({
                "ticker": symbol.upper(),
                "price": None,
                "on_ramp_signals": [],
                "off_ramp_signals": [],
                "on_ramp_score": 0,
                "off_ramp_score": 0,
                "max_on_ramp_score": MAX_ON_RAMP_SCORE,
                "max_off_ramp_score": MAX_OFF_RAMP_SCORE,
                "signal_type": "error",
                "trend_regime": "unknown",
                "error": str(e),
                "timestamp": None,
            })

    return results


def write_signals_json(results: list[dict],
                       path: str | Path = SIGNALS_PATH) -> None:
    """
    Write scan results to a JSON file.

    Overwrites the file each time so it always reflects the latest scan.
    """
    with open(path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    logger.info("Wrote %d results to %s", len(results), path)


def run_scan_loop(interval_minutes: int = SCAN_INTERVAL,
                  watchlist_path: str | None = None,
                  alert_fn=None) -> None:
    """
    Run scans repeatedly on a timer (blocking).

    Parameters
    ----------
    interval_minutes : int
        Minutes between scans.
    watchlist_path : str, optional
        Path to the watchlist YAML.
    alert_fn : callable, optional
        Called with the results list after each scan (e.g. send_email_alert).
    """
    while True:
        logger.info("Starting scan cycle...")
        results = scan_watchlist(watchlist_path)
        write_signals_json(results)

        if alert_fn:
            try:
                alert_fn(results)
            except Exception as e:
                logger.error("Alert failed: %s", e)

        logger.info("Scan complete. Next scan in %d minutes.", interval_minutes)
        time.sleep(interval_minutes * 60)


def start_background_scanner(interval_minutes: int = SCAN_INTERVAL,
                             watchlist_path: str | None = None,
                             alert_fn=None) -> threading.Thread:
    """
    Start the scan loop in a daemon thread (non-blocking).

    Returns the thread object for reference.
    """
    t = threading.Thread(
        target=run_scan_loop,
        args=(interval_minutes, watchlist_path, alert_fn),
        daemon=True,
        name="scanner",
    )
    t.start()
    logger.info("Background scanner started (interval=%dm)", interval_minutes)
    return t
