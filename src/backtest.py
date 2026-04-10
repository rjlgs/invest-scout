"""
Backtesting engine for invest-scout.

Replays the signal engine on historical data to simulate what trades
would have been triggered and what the resulting P&L would look like.

This is NOT a predictor of future returns — it shows how the signals
*would have* performed on past data, which is useful for calibrating
thresholds and understanding signal behaviour.
"""

import logging

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from .data import fetch_historical
from .signals import (
    OFF_RAMP_CHECKS,
    ON_RAMP_CHECKS,
    compute_indicators,
)

logger = logging.getLogger("invest-scout.backtest")

# Minimum combined score to trigger an entry or exit
ENTRY_THRESHOLD = 2
EXIT_THRESHOLD = 2


def run_backtest(ticker: str, start_date: str, end_date: str,
                 config: dict | None = None) -> dict:
    """
    Walk-forward backtest of the signal engine on historical data.

    The simulation is simple and mechanical:
    - Start in IDLE state (no position).
    - When on-ramp score >= ENTRY_THRESHOLD, buy at close price.
    - When off-ramp score >= EXIT_THRESHOLD, sell at close price.
    - At most one position open at a time (long only, fully invested).

    Parameters
    ----------
    ticker : str
        Stock or ETF symbol.
    start_date : str
        YYYY-MM-DD start of the backtest window.
    end_date : str
        YYYY-MM-DD end of the backtest window.
    config : dict, optional
        Per-ticker config overrides.

    Returns
    -------
    dict
        {
            "ticker": str,
            "start_date": str,
            "end_date": str,
            "trades": [
                {
                    "entry_date": str,
                    "entry_price": float,
                    "exit_date": str | None,
                    "exit_price": float | None,
                    "pct_change": float | None,
                },
                ...
            ],
            "total_return_pct": float,
            "buy_hold_return_pct": float,  # buy-and-hold comparison return
            "win_rate": float,
            "num_trades": int,
            "equity_curve": list[float],   # daily portfolio value (starting at 100)
            "buy_hold_curve": list[float], # buy-and-hold curve (starting at 100)
        }
    """
    df = fetch_historical(ticker, start_date, end_date)
    df = compute_indicators(df, config)

    trades: list[dict] = []
    equity = [100.0]  # start with notional $100
    buy_hold = [100.0]  # buy-and-hold comparison curve
    in_position = False
    entry_price = 0.0
    entry_date = ""

    # We need enough history for indicators to be valid — start evaluation
    # after the first 200 rows (SMA200 warm-up period)
    warmup = 200
    if len(df) <= warmup:
        warmup = max(50, len(df) // 2)

    # Track buy-and-hold starting price
    buy_hold_start_price = float(df["Close"].iloc[warmup])

    for i in range(warmup, len(df)):
        # Evaluate signals using data up to (and including) row i
        window = df.iloc[:i + 1]

        on_score = 0
        off_score = 0
        for _, fn in ON_RAMP_CHECKS:
            fired, _ = fn(window, config)
            if fired:
                on_score += 1
        for _, fn in OFF_RAMP_CHECKS:
            fired, _ = fn(window, config)
            if fired:
                off_score += 1

        current_price = float(df["Close"].iloc[i])
        current_date = str(df.index[i].date())

        # Update buy-and-hold curve (always tracks price from start)
        buy_hold.append(100.0 * current_price / buy_hold_start_price)

        if not in_position:
            # Look for entry
            if on_score >= ENTRY_THRESHOLD:
                in_position = True
                entry_price = current_price
                entry_date = current_date
                logger.debug("BUY  %s @ %.2f on %s (score=%d)",
                             ticker, entry_price, entry_date, on_score)

            # Equity stays flat when not invested
            equity.append(equity[-1])
        else:
            # Track equity while in position
            daily_return = current_price / float(df["Close"].iloc[i - 1])
            equity.append(equity[-1] * daily_return)

            # Look for exit
            if off_score >= EXIT_THRESHOLD:
                pct = ((current_price - entry_price) / entry_price) * 100
                trades.append({
                    "entry_date": entry_date,
                    "entry_price": round(entry_price, 2),
                    "exit_date": current_date,
                    "exit_price": round(current_price, 2),
                    "pct_change": round(pct, 2),
                })
                logger.debug("SELL %s @ %.2f on %s (score=%d, pct=%.2f%%)",
                             ticker, current_price, current_date, off_score, pct)
                in_position = False

    # If still in a position at the end, close it at the last price
    if in_position:
        last_price = float(df["Close"].iloc[-1])
        last_date = str(df.index[-1].date())
        pct = ((last_price - entry_price) / entry_price) * 100
        trades.append({
            "entry_date": entry_date,
            "entry_price": round(entry_price, 2),
            "exit_date": last_date + " (open)",
            "exit_price": round(last_price, 2),
            "pct_change": round(pct, 2),
        })

    # Summary stats
    closed = [t for t in trades if t["pct_change"] is not None]
    wins = [t for t in closed if t["pct_change"] > 0]
    total_return = equity[-1] - 100.0 if equity else 0.0
    win_rate = (len(wins) / len(closed) * 100) if closed else 0.0

    # Buy-and-hold return for comparison
    buy_hold_return = buy_hold[-1] - 100.0 if buy_hold else 0.0

    return {
        "ticker": ticker.upper(),
        "start_date": start_date,
        "end_date": end_date,
        "trades": trades,
        "total_return_pct": round(total_return, 2),
        "buy_hold_return_pct": round(buy_hold_return, 2),
        "win_rate": round(win_rate, 1),
        "num_trades": len(trades),
        "equity_curve": [round(v, 2) for v in equity],
        "buy_hold_curve": [round(v, 2) for v in buy_hold],
    }


def generate_equity_curve_html(result: dict) -> str:
    """
    Generate a Plotly equity-curve chart and return it as an HTML div string.
    """
    curve = result.get("equity_curve", [])
    buy_hold = result.get("buy_hold_curve", [])
    if not curve:
        return "<p>No equity data available.</p>"

    fig = go.Figure()

    # Buy-and-hold comparison (add first so strategy line is on top)
    if buy_hold:
        fig.add_trace(go.Scatter(
            y=buy_hold,
            mode="lines",
            name="Buy & Hold",
            line=dict(color="#9E9E9E", width=2, dash="dot"),
        ))

    # Strategy equity curve
    fig.add_trace(go.Scatter(
        y=curve,
        mode="lines",
        name="Strategy",
        line=dict(color="#2196F3", width=2),
    ))

    fig.add_hline(y=100, line_dash="dash", line_color="grey",
                  annotation_text="Starting Value")

    # Build title with return comparison
    strategy_return = result.get("total_return_pct", 0)
    buy_hold_return = result.get("buy_hold_return_pct", 0)
    title = (f"{result['ticker']} Backtest: Strategy {strategy_return:+.1f}% "
             f"vs Buy & Hold {buy_hold_return:+.1f}%")

    fig.update_layout(
        title=title,
        yaxis_title="Portfolio Value ($)",
        xaxis_title="Trading Days",
        template="plotly_white",
        height=400,
        margin=dict(l=50, r=30, t=50, b=40),
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
    )

    return fig.to_html(full_html=False, include_plotlyjs=False)


def format_backtest_summary(result: dict) -> str:
    """
    Return a plain-text summary of backtest results (for CLI output).
    """
    buy_hold_ret = result.get('buy_hold_return_pct', 0)
    lines = [
        f"Backtest: {result['ticker']}  ({result['start_date']} to {result['end_date']})",
        f"{'─' * 60}",
        f"Total trades:    {result['num_trades']}",
        f"Win rate:        {result['win_rate']}%",
        f"Strategy return: {result['total_return_pct']:+.2f}%",
        f"Buy & Hold:      {buy_hold_ret:+.2f}%",
        "",
        f"{'Entry Date':<14} {'Entry $':<10} {'Exit Date':<16} {'Exit $':<10} {'Change':<8}",
        f"{'─' * 14} {'─' * 10} {'─' * 16} {'─' * 10} {'─' * 8}",
    ]
    for t in result["trades"]:
        lines.append(
            f"{t['entry_date']:<14} {t['entry_price']:<10.2f} "
            f"{str(t['exit_date']):<16} {t['exit_price']:<10.2f} "
            f"{t['pct_change']:+.2f}%"
        )

    return "\n".join(lines)
