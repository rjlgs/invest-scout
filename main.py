"""
Invest-Scout — CLI entry point and Flask web dashboard.

Usage:
    python main.py                              # Start the web dashboard
    python main.py --scan                       # One-shot scan, print results
    python main.py --backtest AAPL 2024-01-01 2024-12-31  # Run a backtest

The dashboard runs at http://localhost:5000 by default and includes:
- Overview page: sortable table of all watchlist tickers with signal scores
- Ticker detail: candlestick chart, RSI, MACD, and active signal list
- Backtest page: run historical simulations via a web form
"""

import argparse
import json
import logging
import os
import sys

import plotly.graph_objects as go
from dotenv import load_dotenv
from flask import Flask, render_template, request

from src.alerts import send_email_alert
from src.backtest import (
    format_backtest_summary,
    generate_equity_curve_html,
    run_backtest,
)
from src.data import get_cached_data, load_watchlist
from src.scanner import (
    SIGNALS_PATH,
    scan_watchlist,
    start_background_scanner,
    write_signals_json,
)
from src.signals import compute_indicators, evaluate_signals

load_dotenv()

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
)
logger = logging.getLogger("invest-scout")

# ---------------------------------------------------------------------------
# Flask app
# ---------------------------------------------------------------------------
app = Flask(__name__)


def _load_latest_results() -> list[dict]:
    """Read the most recent scan results from signals.json, or run a scan."""
    if SIGNALS_PATH.exists():
        with open(SIGNALS_PATH) as f:
            return json.load(f)
    # No cached results yet — run a fresh scan
    results = scan_watchlist()
    write_signals_json(results)
    return results


# ---------------------------------------------------------------------------
# Chart generators
# ---------------------------------------------------------------------------

def generate_candlestick_chart(df, ticker: str) -> str:
    """
    Plotly candlestick chart with SMA50 and SMA200 overlaid.
    Returns an HTML div string (no full page, no plotly.js include).
    """
    fig = go.Figure()

    fig.add_trace(go.Candlestick(
        x=df.index,
        open=df["Open"],
        high=df["High"],
        low=df["Low"],
        close=df["Close"],
        name="Price",
    ))

    if "sma_short" in df.columns:
        fig.add_trace(go.Scatter(
            x=df.index, y=df["sma_short"],
            mode="lines", name="SMA 50",
            line=dict(color="#FF9800", width=1.5),
        ))

    if "sma_long" in df.columns:
        fig.add_trace(go.Scatter(
            x=df.index, y=df["sma_long"],
            mode="lines", name="SMA 200",
            line=dict(color="#9C27B0", width=1.5),
        ))

    fig.update_layout(
        title=f"{ticker} — Price & Moving Averages",
        yaxis_title="Price ($)",
        template="plotly_white",
        height=500,
        xaxis_rangeslider_visible=False,
        margin=dict(l=50, r=30, t=50, b=40),
    )

    return fig.to_html(full_html=False, include_plotlyjs=False)


def generate_rsi_chart(df) -> str:
    """RSI line chart with 30/70 threshold bands."""
    fig = go.Figure()

    if "rsi" in df.columns:
        fig.add_trace(go.Scatter(
            x=df.index, y=df["rsi"],
            mode="lines", name="RSI(14)",
            line=dict(color="#2196F3", width=1.5),
        ))

    # Overbought / oversold bands
    fig.add_hline(y=70, line_dash="dash", line_color="#F44336",
                  annotation_text="Overbought (70)")
    fig.add_hline(y=30, line_dash="dash", line_color="#4CAF50",
                  annotation_text="Oversold (30)")

    fig.update_layout(
        title="RSI (14)",
        yaxis_title="RSI",
        yaxis=dict(range=[0, 100]),
        template="plotly_white",
        height=250,
        margin=dict(l=50, r=30, t=50, b=40),
    )

    return fig.to_html(full_html=False, include_plotlyjs=False)


def generate_macd_chart(df) -> str:
    """MACD line, signal line, and histogram bar chart."""
    fig = go.Figure()

    if "macd_hist" in df.columns:
        colors = ["#4CAF50" if v >= 0 else "#F44336"
                  for v in df["macd_hist"].fillna(0)]
        fig.add_trace(go.Bar(
            x=df.index, y=df["macd_hist"],
            name="Histogram",
            marker_color=colors,
        ))

    if "macd" in df.columns:
        fig.add_trace(go.Scatter(
            x=df.index, y=df["macd"],
            mode="lines", name="MACD",
            line=dict(color="#2196F3", width=1.5),
        ))

    if "macd_signal" in df.columns:
        fig.add_trace(go.Scatter(
            x=df.index, y=df["macd_signal"],
            mode="lines", name="Signal",
            line=dict(color="#FF9800", width=1.5),
        ))

    fig.update_layout(
        title="MACD",
        yaxis_title="Value",
        template="plotly_white",
        height=250,
        margin=dict(l=50, r=30, t=50, b=40),
    )

    return fig.to_html(full_html=False, include_plotlyjs=False)


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.route("/")
def index():
    """Overview dashboard — table of all watched tickers."""
    results = _load_latest_results()
    return render_template("index.html", results=results)


@app.route("/api/scan")
def api_scan():
    """
    HTMX endpoint: returns just the table body HTML for auto-refresh.
    Triggers a fresh scan each time.
    """
    results = scan_watchlist()
    write_signals_json(results)
    return render_template("_scan_rows.html", results=results)


@app.route("/ticker/<symbol>")
def ticker_detail(symbol: str):
    """Ticker detail page with charts and signal breakdown."""
    symbol = symbol.upper()

    # Load per-ticker config if available
    tickers = load_watchlist()
    config = {}
    for entry in tickers:
        if entry["symbol"].upper() == symbol:
            config = {k: v for k, v in entry.items() if k != "symbol"}
            break

    try:
        df = get_cached_data(symbol)
        df = compute_indicators(df, config if config else None)
        result = evaluate_signals(df, symbol, config if config else None)

        # Generate charts — show last 6 months for readability
        chart_df = df.iloc[-130:] if len(df) > 130 else df
        candle_html = generate_candlestick_chart(chart_df, symbol)
        rsi_html = generate_rsi_chart(chart_df)
        macd_html = generate_macd_chart(chart_df)

        # Build recent signal history (check each of the last 30 trading days)
        history = _build_signal_history(df, symbol, config, days=30)

    except Exception as e:
        logger.error("Error loading %s: %s", symbol, e)
        return render_template("ticker.html", symbol=symbol, error=str(e))

    return render_template(
        "ticker.html",
        symbol=symbol,
        result=result,
        candle_chart=candle_html,
        rsi_chart=rsi_html,
        macd_chart=macd_html,
        history=history,
    )


def _build_signal_history(df, ticker, config, days=30):
    """Check what signals were firing on each of the last N trading days."""
    history = []
    start = max(0, len(df) - days)
    for i in range(start, len(df)):
        window = df.iloc[:i + 1]
        if len(window) < 2:
            continue
        result = evaluate_signals(window, ticker, config if config else None)
        if result["on_ramp_score"] > 0 or result["off_ramp_score"] > 0:
            history.append({
                "date": str(window.index[-1].date()),
                "signal_type": result["signal_type"],
                "on_ramp_score": result["on_ramp_score"],
                "off_ramp_score": result["off_ramp_score"],
                "signals": (
                    [s["name"] for s in result["on_ramp_signals"]] +
                    [s["name"] for s in result["off_ramp_signals"]]
                ),
            })
    return list(reversed(history))  # most recent first


@app.route("/backtest")
def backtest_page():
    """Backtest form page."""
    return render_template("backtest.html")


@app.route("/api/backtest", methods=["POST"])
def api_backtest():
    """Run a backtest and return results as HTMX partial."""
    ticker = request.form.get("ticker", "").strip().upper()
    start = request.form.get("start_date", "").strip()
    end = request.form.get("end_date", "").strip()

    if not all([ticker, start, end]):
        return "<p class='error'>Please fill in all fields.</p>"

    try:
        result = run_backtest(ticker, start, end)
        equity_html = generate_equity_curve_html(result)
        return render_template(
            "_backtest_results.html",
            result=result,
            equity_chart=equity_html,
        )
    except Exception as e:
        logger.error("Backtest error: %s", e)
        return f"<p class='error'>Backtest failed: {e}</p>"


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Invest-Scout: automated investment signal analysis"
    )
    parser.add_argument(
        "--backtest", nargs=3, metavar=("TICKER", "START", "END"),
        help="Run a backtest: --backtest AAPL 2024-01-01 2024-12-31",
    )
    parser.add_argument(
        "--scan", action="store_true",
        help="Run a one-shot scan, print results, and exit",
    )
    parser.add_argument(
        "--port", type=int, default=int(os.getenv("FLASK_PORT", "5000")),
        help="Port for the web dashboard (default: 5000)",
    )
    args = parser.parse_args()

    if args.backtest:
        ticker, start, end = args.backtest
        logger.info("Running backtest: %s %s to %s", ticker, start, end)
        result = run_backtest(ticker, start, end)
        print(format_backtest_summary(result))
        sys.exit(0)

    if args.scan:
        logger.info("Running one-shot scan...")
        results = scan_watchlist()
        write_signals_json(results)
        for r in results:
            status = r["signal_type"].upper()
            on = r["on_ramp_score"]
            off = r["off_ramp_score"]
            price = r.get("price", "N/A")
            print(f"  {r['ticker']:<6} ${price:<10} {status:<10} "
                  f"on-ramp={on} off-ramp={off}")
        print(f"\nResults written to {SIGNALS_PATH}")
        sys.exit(0)

    # Default: start the web dashboard with background scanning
    logger.info("Starting invest-scout dashboard on port %d...", args.port)
    start_background_scanner(alert_fn=send_email_alert)
    app.run(host="0.0.0.0", port=args.port, debug=False)


if __name__ == "__main__":
    main()
