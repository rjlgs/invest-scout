"""
Signal engine for invest-scout.

Computes technical indicators (RSI, SMA, MACD, volume averages) and
detects on-ramp (entry) and off-ramp (exit) signals for swing trading.

All indicators are calculated with plain pandas/numpy rather than an
external TA library, keeping dependencies minimal and logic transparent.
"""

import os
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

# ---------------------------------------------------------------------------
# Configurable thresholds — overridable via .env or per-ticker YAML config
# ---------------------------------------------------------------------------
DEFAULTS = {
    "rsi_period": int(os.getenv("RSI_PERIOD", "14")),
    "rsi_oversold": float(os.getenv("RSI_OVERSOLD", "30")),
    "rsi_overbought": float(os.getenv("RSI_OVERBOUGHT", "70")),
    "sma_short": int(os.getenv("SMA_SHORT", "50")),
    "sma_long": int(os.getenv("SMA_LONG", "200")),
    "macd_fast": int(os.getenv("MACD_FAST", "12")),
    "macd_slow": int(os.getenv("MACD_SLOW", "26")),
    "macd_signal": int(os.getenv("MACD_SIGNAL", "9")),
    "volume_spike_multiplier": float(os.getenv("VOLUME_SPIKE_MULTIPLIER", "2.0")),
    "trailing_stop_pct": float(os.getenv("TRAILING_STOP_PCT", "8.0")),
}


def _cfg(key: str, overrides: dict | None = None):
    """Return a config value, preferring per-ticker overrides."""
    if overrides and key in overrides:
        return type(DEFAULTS[key])(overrides[key])
    return DEFAULTS[key]


# ===================================================================
# Indicator calculations
# ===================================================================

def _rsi(series: pd.Series, period: int = 14) -> pd.Series:
    """
    Relative Strength Index.

    RSI measures the speed and magnitude of recent price changes to
    evaluate whether a stock is overbought or oversold. Values below 30
    suggest oversold conditions (potential buying opportunity), while
    values above 70 suggest overbought conditions (potential exit).
    """
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    # Wilder's smoothing (exponential moving average)
    avg_gain = gain.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()

    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))


def _sma(series: pd.Series, period: int) -> pd.Series:
    """
    Simple Moving Average.

    The SMA smooths out price noise. The 50-day SMA captures the
    medium-term trend, while the 200-day SMA captures the long-term
    trend. Crossovers between these two are classic trend-reversal signals.
    """
    return series.rolling(window=period, min_periods=period).mean()


def _macd(series: pd.Series, fast: int = 12, slow: int = 26,
          signal: int = 9) -> tuple[pd.Series, pd.Series, pd.Series]:
    """
    MACD (Moving Average Convergence Divergence).

    MACD shows the relationship between two EMAs of price. When the MACD
    histogram flips from negative to positive, momentum is shifting
    bullish; the reverse signals bearish momentum.

    Returns (macd_line, signal_line, histogram).
    """
    ema_fast = series.ewm(span=fast, adjust=False).mean()
    ema_slow = series.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram


def compute_indicators(df: pd.DataFrame,
                       config: dict | None = None) -> pd.DataFrame:
    """
    Add all technical indicator columns to the OHLCV DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain columns: Close, Open, High, Low, Volume.
    config : dict, optional
        Per-ticker config overrides.

    Returns
    -------
    pd.DataFrame
        Original DataFrame with added columns:
        sma_short, sma_long, rsi, macd, macd_signal, macd_hist, vol_avg_20
    """
    out = df.copy()

    rsi_period = _cfg("rsi_period", config)
    sma_short = _cfg("sma_short", config)
    sma_long = _cfg("sma_long", config)
    macd_fast = _cfg("macd_fast", config)
    macd_slow = _cfg("macd_slow", config)
    macd_sig = _cfg("macd_signal", config)

    out["rsi"] = _rsi(out["Close"], rsi_period)
    out["sma_short"] = _sma(out["Close"], sma_short)
    out["sma_long"] = _sma(out["Close"], sma_long)

    m_line, s_line, hist = _macd(out["Close"], macd_fast, macd_slow, macd_sig)
    out["macd"] = m_line
    out["macd_signal"] = s_line
    out["macd_hist"] = hist

    # 20-day average volume — baseline for spotting unusual activity
    out["vol_avg_20"] = out["Volume"].rolling(window=20, min_periods=20).mean()

    return out


# ===================================================================
# Individual signal checks
# Each returns (is_firing: bool, reason: str)
# ===================================================================

def check_rsi_oversold_reversal(df: pd.DataFrame,
                                config: dict | None = None) -> tuple[bool, str]:
    """
    ON-RAMP: RSI crosses below the oversold threshold then recovers above it.

    WHY this matters: When RSI dips below 30 it means selling has been
    extreme — most weak holders have already exited. A recovery back above
    30 signals that buying pressure is returning, often marking the start
    of a bounce or trend reversal. This is one of the most reliable
    mean-reversion entry signals for swing traders.
    """
    threshold = _cfg("rsi_oversold", config)
    if len(df) < 2 or pd.isna(df["rsi"].iloc[-2]) or pd.isna(df["rsi"].iloc[-1]):
        return False, ""

    prev_rsi = df["rsi"].iloc[-2]
    curr_rsi = df["rsi"].iloc[-1]

    if prev_rsi < threshold and curr_rsi >= threshold:
        return True, (
            f"RSI recovered from oversold: {prev_rsi:.1f} -> {curr_rsi:.1f} "
            f"(threshold {threshold})"
        )
    return False, ""


def check_price_reclaims_sma50(df: pd.DataFrame,
                               config: dict | None = None) -> tuple[bool, str]:
    """
    ON-RAMP: Price reclaims the short-period SMA after trading below it.

    WHY this matters: The 50-day SMA acts as a dynamic support/resistance
    level. When price falls below it, the stock is in a short-term
    downtrend. Reclaiming it from below signals that buyers have regained
    control and the medium-term trend may be turning positive again.
    """
    if len(df) < 2:
        return False, ""
    sma_col = "sma_short"
    if pd.isna(df[sma_col].iloc[-1]) or pd.isna(df[sma_col].iloc[-2]):
        return False, ""

    prev_below = df["Close"].iloc[-2] < df[sma_col].iloc[-2]
    curr_above = df["Close"].iloc[-1] >= df[sma_col].iloc[-1]

    if prev_below and curr_above:
        return True, (
            f"Price reclaimed SMA{_cfg('sma_short', config)}: "
            f"{df['Close'].iloc[-1]:.2f} crossed above {df[sma_col].iloc[-1]:.2f}"
        )
    return False, ""


def check_golden_cross(df: pd.DataFrame,
                       config: dict | None = None) -> tuple[bool, str]:
    """
    ON-RAMP: Short-period SMA crosses above long-period SMA (Golden Cross).

    WHY this matters: The golden cross is one of the most watched signals
    on Wall Street. When the 50-day SMA rises above the 200-day SMA, it
    indicates that recent price momentum has shifted decisively upward
    relative to the long-term trend. Historically this precedes sustained
    rallies and is used by institutions to confirm trend reversals.
    """
    if len(df) < 2:
        return False, ""
    if pd.isna(df["sma_short"].iloc[-1]) or pd.isna(df["sma_long"].iloc[-1]):
        return False, ""
    if pd.isna(df["sma_short"].iloc[-2]) or pd.isna(df["sma_long"].iloc[-2]):
        return False, ""

    prev_below = df["sma_short"].iloc[-2] < df["sma_long"].iloc[-2]
    curr_above = df["sma_short"].iloc[-1] >= df["sma_long"].iloc[-1]

    if prev_below and curr_above:
        return True, (
            f"Golden Cross: SMA{_cfg('sma_short', config)} crossed above "
            f"SMA{_cfg('sma_long', config)}"
        )
    return False, ""


def check_macd_bullish_flip(df: pd.DataFrame,
                            config: dict | None = None) -> tuple[bool, str]:
    """
    ON-RAMP: MACD histogram flips from negative to positive.

    WHY this matters: The MACD histogram represents the difference between
    the MACD line and its signal line. When it flips positive, short-term
    momentum is accelerating faster than medium-term momentum — a bullish
    shift. This often confirms that a price bounce has real momentum
    behind it, not just a dead-cat bounce.
    """
    if len(df) < 2:
        return False, ""
    if pd.isna(df["macd_hist"].iloc[-1]) or pd.isna(df["macd_hist"].iloc[-2]):
        return False, ""

    prev_neg = df["macd_hist"].iloc[-2] < 0
    curr_pos = df["macd_hist"].iloc[-1] >= 0

    if prev_neg and curr_pos:
        return True, (
            f"MACD histogram flipped bullish: "
            f"{df['macd_hist'].iloc[-2]:.4f} -> {df['macd_hist'].iloc[-1]:.4f}"
        )
    return False, ""


def check_volume_spike_green(df: pd.DataFrame,
                             config: dict | None = None) -> tuple[bool, str]:
    """
    ON-RAMP: Volume spike (>Nx 20-day avg) on a green (up) candle.

    WHY this matters: A large volume spike on a green candle means
    significantly more shares changed hands than usual AND the price
    closed higher than it opened. This often indicates institutional
    buying — large funds accumulating a position. Retail traders rarely
    move volume that much. When big money steps in, it tends to support
    the price going forward.
    """
    multiplier = _cfg("volume_spike_multiplier", config)
    if len(df) < 1:
        return False, ""
    if pd.isna(df["vol_avg_20"].iloc[-1]):
        return False, ""

    curr = df.iloc[-1]
    is_green = curr["Close"] > curr["Open"]
    is_spike = curr["Volume"] > multiplier * df["vol_avg_20"].iloc[-1]

    if is_green and is_spike:
        ratio = curr["Volume"] / df["vol_avg_20"].iloc[-1]
        return True, (
            f"Volume spike on green candle: {ratio:.1f}x average "
            f"(threshold {multiplier}x)"
        )
    return False, ""


def check_rsi_overbought_reversal(df: pd.DataFrame,
                                  config: dict | None = None) -> tuple[bool, str]:
    """
    OFF-RAMP: RSI crosses above the overbought threshold then drops below it.

    WHY this matters: RSI above 70 means the stock has been rising fast —
    potentially too fast. When RSI then drops back below 70, it suggests
    the buying frenzy is fading and profit-taking has begun. For swing
    traders holding a position, this is an early warning to consider
    locking in gains before a pullback accelerates.
    """
    threshold = _cfg("rsi_overbought", config)
    if len(df) < 2 or pd.isna(df["rsi"].iloc[-2]) or pd.isna(df["rsi"].iloc[-1]):
        return False, ""

    prev_rsi = df["rsi"].iloc[-2]
    curr_rsi = df["rsi"].iloc[-1]

    if prev_rsi > threshold and curr_rsi <= threshold:
        return True, (
            f"RSI dropped from overbought: {prev_rsi:.1f} -> {curr_rsi:.1f} "
            f"(threshold {threshold})"
        )
    return False, ""


def check_price_breaks_sma50(df: pd.DataFrame,
                             config: dict | None = None) -> tuple[bool, str]:
    """
    OFF-RAMP: Price breaks below the short-period SMA after trading above it.

    WHY this matters: When a stock that has been riding above its 50-day
    SMA suddenly closes below it, the medium-term trend is weakening.
    This level often acts as support, so losing it means that support has
    failed — sellers are in control. It's a warning sign to reduce
    exposure before a deeper decline.
    """
    if len(df) < 2:
        return False, ""
    sma_col = "sma_short"
    if pd.isna(df[sma_col].iloc[-1]) or pd.isna(df[sma_col].iloc[-2]):
        return False, ""

    prev_above = df["Close"].iloc[-2] > df[sma_col].iloc[-2]
    curr_below = df["Close"].iloc[-1] <= df[sma_col].iloc[-1]

    if prev_above and curr_below:
        return True, (
            f"Price broke below SMA{_cfg('sma_short', config)}: "
            f"{df['Close'].iloc[-1]:.2f} fell under {df[sma_col].iloc[-1]:.2f}"
        )
    return False, ""


def check_death_cross(df: pd.DataFrame,
                      config: dict | None = None) -> tuple[bool, str]:
    """
    OFF-RAMP: Short-period SMA crosses below long-period SMA (Death Cross).

    WHY this matters: The death cross is the inverse of the golden cross.
    When the 50-day SMA drops below the 200-day SMA, it confirms that
    recent weakness has overwhelmed the longer-term trend. Major indices
    often see extended drawdowns after a death cross. It's a signal to
    exit remaining long positions and wait on the sidelines.
    """
    if len(df) < 2:
        return False, ""
    if pd.isna(df["sma_short"].iloc[-1]) or pd.isna(df["sma_long"].iloc[-1]):
        return False, ""
    if pd.isna(df["sma_short"].iloc[-2]) or pd.isna(df["sma_long"].iloc[-2]):
        return False, ""

    prev_above = df["sma_short"].iloc[-2] > df["sma_long"].iloc[-2]
    curr_below = df["sma_short"].iloc[-1] <= df["sma_long"].iloc[-1]

    if prev_above and curr_below:
        return True, (
            f"Death Cross: SMA{_cfg('sma_short', config)} crossed below "
            f"SMA{_cfg('sma_long', config)}"
        )
    return False, ""


def check_macd_bearish_flip(df: pd.DataFrame,
                            config: dict | None = None) -> tuple[bool, str]:
    """
    OFF-RAMP: MACD histogram flips from positive to negative.

    WHY this matters: When the MACD histogram turns negative, short-term
    momentum is decelerating relative to the medium-term trend. This
    often confirms that a rally is losing steam. Exiting when this fires
    avoids holding through the initial phase of a pullback.
    """
    if len(df) < 2:
        return False, ""
    if pd.isna(df["macd_hist"].iloc[-1]) or pd.isna(df["macd_hist"].iloc[-2]):
        return False, ""

    prev_pos = df["macd_hist"].iloc[-2] >= 0
    curr_neg = df["macd_hist"].iloc[-1] < 0

    if prev_pos and curr_neg:
        return True, (
            f"MACD histogram flipped bearish: "
            f"{df['macd_hist'].iloc[-2]:.4f} -> {df['macd_hist'].iloc[-1]:.4f}"
        )
    return False, ""


def check_trailing_stop(df: pd.DataFrame,
                        config: dict | None = None) -> tuple[bool, str]:
    """
    OFF-RAMP: Price has dropped more than X% from its recent high.

    WHY this matters: A trailing stop protects accumulated gains. If a
    stock falls 8% (default) from its recent peak, the uptrend is likely
    broken. Waiting for a larger drop often means giving back most of
    the profit. The trailing stop enforces discipline by defining the
    maximum acceptable drawdown from the top.
    """
    pct = _cfg("trailing_stop_pct", config)
    lookback = 20  # rolling window for recent high

    if len(df) < lookback:
        return False, ""

    recent_high = df["Close"].iloc[-lookback:].max()
    current = df["Close"].iloc[-1]
    drawdown_pct = ((recent_high - current) / recent_high) * 100

    if drawdown_pct > pct:
        return True, (
            f"Trailing stop breached: price {current:.2f} is {drawdown_pct:.1f}% "
            f"below recent high {recent_high:.2f} (threshold {pct}%)"
        )
    return False, ""


# ===================================================================
# Aggregate evaluation
# ===================================================================

# Signal check functions grouped by type for easy iteration
ON_RAMP_CHECKS = [
    ("RSI Oversold Reversal", check_rsi_oversold_reversal),
    ("Price Reclaims SMA50", check_price_reclaims_sma50),
    ("Golden Cross", check_golden_cross),
    ("MACD Bullish Flip", check_macd_bullish_flip),
    ("Volume Spike (Green)", check_volume_spike_green),
]

OFF_RAMP_CHECKS = [
    ("RSI Overbought Reversal", check_rsi_overbought_reversal),
    ("Price Breaks SMA50", check_price_breaks_sma50),
    ("Death Cross", check_death_cross),
    ("MACD Bearish Flip", check_macd_bearish_flip),
    ("Trailing Stop Breach", check_trailing_stop),
]


def evaluate_signals(df: pd.DataFrame, ticker: str,
                     config: dict | None = None) -> dict:
    """
    Run all signal checks and return an aggregate result.

    Parameters
    ----------
    df : pd.DataFrame
        OHLCV DataFrame **with indicators already computed**.
    ticker : str
        The ticker symbol (included in the output for identification).
    config : dict, optional
        Per-ticker config overrides.

    Returns
    -------
    dict
        {
            "ticker": str,
            "price": float,
            "on_ramp_signals": [{"name": str, "reason": str}, ...],
            "off_ramp_signals": [{"name": str, "reason": str}, ...],
            "on_ramp_score": int,   # 0-5
            "off_ramp_score": int,  # 0-5
            "signal_type": "on-ramp" | "off-ramp" | "neutral",
            "timestamp": str,       # ISO 8601
        }
    """
    on_ramp = []
    for name, fn in ON_RAMP_CHECKS:
        firing, reason = fn(df, config)
        if firing:
            on_ramp.append({"name": name, "reason": reason})

    off_ramp = []
    for name, fn in OFF_RAMP_CHECKS:
        firing, reason = fn(df, config)
        if firing:
            off_ramp.append({"name": name, "reason": reason})

    on_score = len(on_ramp)
    off_score = len(off_ramp)

    if on_score > off_score:
        signal_type = "on-ramp"
    elif off_score > on_score:
        signal_type = "off-ramp"
    else:
        signal_type = "neutral"

    return {
        "ticker": ticker.upper(),
        "price": round(float(df["Close"].iloc[-1]), 2),
        "on_ramp_signals": on_ramp,
        "off_ramp_signals": off_ramp,
        "on_ramp_score": on_score,
        "off_ramp_score": off_score,
        "signal_type": signal_type,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
