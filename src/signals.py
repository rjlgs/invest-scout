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
    # --- Indicators & Signals v2 ---
    "ema_fast": int(os.getenv("EMA_FAST", "20")),
    "bb_period": int(os.getenv("BB_PERIOD", "20")),
    "bb_std": float(os.getenv("BB_STD", "2.0")),
    "bb_squeeze_lookback": int(os.getenv("BB_SQUEEZE_LOOKBACK", "120")),
    "bb_squeeze_percentile": float(os.getenv("BB_SQUEEZE_PERCENTILE", "20")),
    "atr_period": int(os.getenv("ATR_PERIOD", "14")),
    "atr_volatility_stop_mult": float(os.getenv("ATR_VOLATILITY_STOP_MULT", "3.0")),
    "adx_period": int(os.getenv("ADX_PERIOD", "14")),
    "adx_threshold": float(os.getenv("ADX_THRESHOLD", "25")),
    "adx_weak_threshold": float(os.getenv("ADX_WEAK_THRESHOLD", "20")),
    "stoch_k_period": int(os.getenv("STOCH_K_PERIOD", "14")),
    "stoch_k_smooth": int(os.getenv("STOCH_K_SMOOTH", "3")),
    "stoch_d_period": int(os.getenv("STOCH_D_PERIOD", "3")),
    "stoch_oversold": float(os.getenv("STOCH_OVERSOLD", "20")),
    "stoch_overbought": float(os.getenv("STOCH_OVERBOUGHT", "80")),
    "ema_reclaim_vol_mult": float(os.getenv("EMA_RECLAIM_VOL_MULT", "1.5")),
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


def _ema(series: pd.Series, period: int) -> pd.Series:
    """
    Exponential Moving Average — weights recent prices more heavily.

    Used as a fast trend line (EMA 20 is a widely-watched level by
    short-term traders). Reclaiming the EMA from below often marks the
    start of a short-term trend reversal.
    """
    return series.ewm(span=period, adjust=False).mean()


def _bollinger_bands(series: pd.Series, period: int = 20,
                     num_std: float = 2.0
                     ) -> tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
    """
    Bollinger Bands (upper, middle, lower) and normalized Bandwidth.

    Bollinger Bands plot N standard deviations above and below an SMA.
    They contract ("squeeze") during low-volatility consolidation and
    expand when a new trend begins — volatility mean-reverts, and a
    squeeze almost always precedes a large directional move.

    Bandwidth = (upper - lower) / middle, used to detect squeezes.

    Returns (upper, middle, lower, bandwidth).
    """
    middle = series.rolling(window=period, min_periods=period).mean()
    std = series.rolling(window=period, min_periods=period).std(ddof=0)
    upper = middle + num_std * std
    lower = middle - num_std * std
    # Guard against divide-by-zero on flat series
    bandwidth = (upper - lower) / middle.replace(0, np.nan)
    return upper, middle, lower, bandwidth


def _atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """
    Average True Range — Wilder's volatility measure.

    True Range captures the greatest of: today's H-L, |H - prevClose|,
    |L - prevClose|. ATR is the Wilder-smoothed average of TR, producing
    a robust per-bar volatility estimate that accounts for gaps.

    ATR is the foundation for adaptive stop placement — a stock that
    moves 4% on an average day needs a wider stop than one moving 0.5%.
    """
    high = df["High"]
    low = df["Low"]
    prev_close = df["Close"].shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ], axis=1).max(axis=1)
    # Wilder's smoothing — identical alpha pattern to RSI
    return tr.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()


def _adx(df: pd.DataFrame, period: int = 14
         ) -> tuple[pd.Series, pd.Series, pd.Series]:
    """
    Average Directional Index — measures trend STRENGTH, not direction.

    ADX is the single best filter for ranging vs. trending markets:
    - ADX < 20:  weak/no trend (avoid trend-following strategies)
    - ADX 20-25: emerging trend
    - ADX > 25:  strong, tradable trend
    - ADX > 40:  very strong trend (late stage)

    Combined with +DI/-DI (directional indicators), ADX tells us not
    just that a trend exists but which way it's pointing. This is the
    cornerstone of any trend-following strategy that aims to avoid
    choppy markets where most whipsaws happen.

    Returns (adx, plus_di, minus_di).
    """
    high = df["High"]
    low = df["Low"]
    close = df["Close"]

    up_move = high.diff()
    down_move = -low.diff()

    # +DM: up_move > down_move AND up_move > 0; else 0
    plus_dm = pd.Series(
        np.where((up_move > down_move) & (up_move > 0), up_move, 0.0),
        index=df.index,
    )
    # -DM: down_move > up_move AND down_move > 0; else 0
    minus_dm = pd.Series(
        np.where((down_move > up_move) & (down_move > 0), down_move, 0.0),
        index=df.index,
    )

    # True Range (same as in _atr, computed inline to avoid coupling)
    prev_close = close.shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ], axis=1).max(axis=1)

    # Wilder's smoothing on TR and ±DM
    alpha = 1 / period
    tr_smooth = tr.ewm(alpha=alpha, min_periods=period, adjust=False).mean()
    plus_dm_smooth = plus_dm.ewm(alpha=alpha, min_periods=period,
                                  adjust=False).mean()
    minus_dm_smooth = minus_dm.ewm(alpha=alpha, min_periods=period,
                                    adjust=False).mean()

    plus_di = 100 * (plus_dm_smooth / tr_smooth.replace(0, np.nan))
    minus_di = 100 * (minus_dm_smooth / tr_smooth.replace(0, np.nan))

    di_sum = (plus_di + minus_di).replace(0, np.nan)
    dx = 100 * (plus_di - minus_di).abs() / di_sum
    adx = dx.ewm(alpha=alpha, min_periods=period, adjust=False).mean()

    return adx, plus_di, minus_di


def _stochastic(df: pd.DataFrame, k_period: int = 14,
                k_smooth: int = 3, d_period: int = 3
                ) -> tuple[pd.Series, pd.Series]:
    """
    Stochastic Oscillator (%K, %D).

    Stochastic compares the current close to its recent range:
        %K_raw = 100 * (C - LowestLow) / (HighestHigh - LowestLow)
    A reading below 20 is oversold; above 80 is overbought. The signal
    line %D is an SMA of %K. Bullish crosses (%K crosses above %D) in
    oversold territory are classic swing-trade entries; bearish crosses
    in overbought territory are exits.

    Stochastic is particularly useful because it responds faster than
    RSI and confirms momentum shifts earlier.

    Returns (pct_k, pct_d).
    """
    low_min = df["Low"].rolling(window=k_period, min_periods=k_period).min()
    high_max = df["High"].rolling(window=k_period, min_periods=k_period).max()
    range_ = (high_max - low_min).replace(0, np.nan)
    raw_k = 100 * (df["Close"] - low_min) / range_
    # Standard "slow" stochastic: smooth %K with an SMA
    pct_k = raw_k.rolling(window=k_smooth, min_periods=k_smooth).mean()
    pct_d = pct_k.rolling(window=d_period, min_periods=d_period).mean()
    return pct_k, pct_d


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
        sma_short, sma_long, rsi, macd, macd_signal, macd_hist, vol_avg_20,
        ema_fast, bb_upper, bb_middle, bb_lower, bb_bandwidth,
        atr, adx, plus_di, minus_di, stoch_k, stoch_d
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

    # --- v2 indicators ---------------------------------------------------
    out["ema_fast"] = _ema(out["Close"], _cfg("ema_fast", config))

    bb_u, bb_m, bb_l, bb_bw = _bollinger_bands(
        out["Close"],
        _cfg("bb_period", config),
        _cfg("bb_std", config),
    )
    out["bb_upper"] = bb_u
    out["bb_middle"] = bb_m
    out["bb_lower"] = bb_l
    out["bb_bandwidth"] = bb_bw

    out["atr"] = _atr(out, _cfg("atr_period", config))

    adx, plus_di, minus_di = _adx(out, _cfg("adx_period", config))
    out["adx"] = adx
    out["plus_di"] = plus_di
    out["minus_di"] = minus_di

    stoch_k, stoch_d = _stochastic(
        out,
        _cfg("stoch_k_period", config),
        _cfg("stoch_k_smooth", config),
        _cfg("stoch_d_period", config),
    )
    out["stoch_k"] = stoch_k
    out["stoch_d"] = stoch_d

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
    OFF-RAMP: Price has dropped more than X% from its recent HIGH.

    WHY this matters: A trailing stop protects accumulated gains. If a
    stock falls 8% (default) from its recent peak, the uptrend is likely
    broken. Waiting for a larger drop often means giving back most of
    the profit. The trailing stop enforces discipline by defining the
    maximum acceptable drawdown from the top.

    Uses the actual High price for the rolling peak (not Close), so the
    drawdown reflects the true peak-to-trough move a holder would see.
    """
    pct = _cfg("trailing_stop_pct", config)
    lookback = 20  # rolling window for recent high

    if len(df) < lookback or "High" not in df.columns:
        return False, ""

    recent_high = df["High"].iloc[-lookback:].max()
    current = df["Close"].iloc[-1]
    drawdown_pct = ((recent_high - current) / recent_high) * 100

    if drawdown_pct > pct:
        return True, (
            f"Trailing stop breached: price {current:.2f} is {drawdown_pct:.1f}% "
            f"below recent high {recent_high:.2f} (threshold {pct}%)"
        )
    return False, ""


# ===================================================================
# v2 signal checks — volatility, trend strength, adaptive stops
# ===================================================================

def check_bb_squeeze_breakout(df: pd.DataFrame,
                              config: dict | None = None) -> tuple[bool, str]:
    """
    ON-RAMP: Bollinger Band squeeze followed by an upside breakout.

    WHY this matters: Volatility mean-reverts. When the Bollinger Bands
    contract tightly (bandwidth in the bottom percentile of recent
    history), the stock is coiled for a directional move. A close above
    the upper band after a squeeze is one of the highest-expectancy
    technical signals because it combines two independent edges:
    (1) a volatility expansion is due, and (2) the expansion has
    resolved upward. Developed and widely used by John Bollinger himself
    as the "Bollinger Band squeeze" play.
    """
    lookback = _cfg("bb_squeeze_lookback", config)
    pct_threshold = _cfg("bb_squeeze_percentile", config)

    if len(df) < lookback + 1:
        return False, ""
    if "bb_bandwidth" not in df.columns or "bb_upper" not in df.columns:
        return False, ""
    if pd.isna(df["bb_bandwidth"].iloc[-1]) or pd.isna(df["bb_upper"].iloc[-1]):
        return False, ""

    bw_window = df["bb_bandwidth"].iloc[-lookback:]
    if bw_window.isna().all():
        return False, ""
    threshold = bw_window.quantile(pct_threshold / 100.0)

    # Was the bandwidth in squeeze territory within the last 5 bars?
    recent_bw = df["bb_bandwidth"].iloc[-5:]
    was_squeezed = (recent_bw <= threshold).any()

    curr_close = df["Close"].iloc[-1]
    curr_upper = df["bb_upper"].iloc[-1]
    is_breakout = curr_close > curr_upper

    if was_squeezed and is_breakout:
        return True, (
            f"BB squeeze breakout: close {curr_close:.2f} > upper band "
            f"{curr_upper:.2f} after bandwidth compressed into bottom "
            f"{pct_threshold:.0f}% over last {lookback} bars"
        )
    return False, ""


def check_adx_bullish_trend(df: pd.DataFrame,
                            config: dict | None = None) -> tuple[bool, str]:
    """
    ON-RAMP: ADX confirms a rising bullish trend.

    WHY this matters: ADX is the gold standard for distinguishing real
    trends from chop. An ADX above 25 with +DI above -DI means there's
    a measurable, directionally-bullish trend — not a coin-flip market.
    Adding the requirement that ADX is rising filters for trends that
    are STRENGTHENING rather than fading, which dramatically improves
    entry quality. This single filter prevents most whipsaws that plague
    momentum systems in rangebound markets.
    """
    threshold = _cfg("adx_threshold", config)
    if len(df) < 2:
        return False, ""
    for col in ("adx", "plus_di", "minus_di"):
        if col not in df.columns:
            return False, ""
        if pd.isna(df[col].iloc[-1]) or pd.isna(df[col].iloc[-2]):
            return False, ""

    curr_adx = df["adx"].iloc[-1]
    prev_adx = df["adx"].iloc[-2]
    plus_di = df["plus_di"].iloc[-1]
    minus_di = df["minus_di"].iloc[-1]

    if curr_adx > threshold and plus_di > minus_di and curr_adx > prev_adx:
        return True, (
            f"ADX bullish trend: ADX {curr_adx:.1f} (rising from {prev_adx:.1f}) "
            f"with +DI {plus_di:.1f} > -DI {minus_di:.1f}"
        )
    return False, ""


def check_stochastic_bullish_cross(df: pd.DataFrame,
                                   config: dict | None = None
                                   ) -> tuple[bool, str]:
    """
    ON-RAMP: Stochastic %K crosses above %D from oversold territory.

    WHY this matters: The stochastic oscillator is more sensitive than
    RSI — it fires earlier on reversals. A bullish %K/%D cross while
    both lines are still below 20 (oversold) is a classic early-entry
    signal that captures the momentum shift before the broader market
    notices. Requiring the cross to happen IN oversold territory (not
    above 20) filters out mid-range whipsaws.
    """
    oversold = _cfg("stoch_oversold", config)
    if len(df) < 2:
        return False, ""
    for col in ("stoch_k", "stoch_d"):
        if col not in df.columns:
            return False, ""
        if pd.isna(df[col].iloc[-1]) or pd.isna(df[col].iloc[-2]):
            return False, ""

    prev_k = df["stoch_k"].iloc[-2]
    prev_d = df["stoch_d"].iloc[-2]
    curr_k = df["stoch_k"].iloc[-1]
    curr_d = df["stoch_d"].iloc[-1]

    crossed_up = prev_k <= prev_d and curr_k > curr_d
    was_oversold = prev_k < oversold and prev_d < oversold

    if crossed_up and was_oversold:
        return True, (
            f"Stochastic bullish cross in oversold zone: %K {prev_k:.1f}->{curr_k:.1f}, "
            f"%D {prev_d:.1f}->{curr_d:.1f} (oversold threshold {oversold})"
        )
    return False, ""


def check_ema20_reclaim_with_volume(df: pd.DataFrame,
                                    config: dict | None = None
                                    ) -> tuple[bool, str]:
    """
    ON-RAMP: Price reclaims the fast EMA on above-average volume.

    WHY this matters: The 20-period EMA is the dividing line between
    short-term bearish and bullish structure. A reclaim on its own is
    only moderately reliable — but paired with volume >1.5x its 20-day
    average, it indicates that the reclaim has real buying interest
    behind it rather than a low-volume drift. Volume confirmation is
    the single most important addition to any price-based signal.
    """
    vol_mult = _cfg("ema_reclaim_vol_mult", config)
    if len(df) < 2:
        return False, ""
    for col in ("ema_fast", "vol_avg_20"):
        if col not in df.columns:
            return False, ""
    if (pd.isna(df["ema_fast"].iloc[-1]) or pd.isna(df["ema_fast"].iloc[-2])
            or pd.isna(df["vol_avg_20"].iloc[-1])):
        return False, ""

    prev_below = df["Close"].iloc[-2] < df["ema_fast"].iloc[-2]
    curr_above = df["Close"].iloc[-1] >= df["ema_fast"].iloc[-1]
    vol_ok = df["Volume"].iloc[-1] > vol_mult * df["vol_avg_20"].iloc[-1]

    if prev_below and curr_above and vol_ok:
        ratio = df["Volume"].iloc[-1] / df["vol_avg_20"].iloc[-1]
        return True, (
            f"Reclaimed EMA{_cfg('ema_fast', config)} on {ratio:.1f}x avg volume: "
            f"close {df['Close'].iloc[-1]:.2f} > EMA {df['ema_fast'].iloc[-1]:.2f}"
        )
    return False, ""


def check_bb_upper_rejection(df: pd.DataFrame,
                             config: dict | None = None) -> tuple[bool, str]:
    """
    OFF-RAMP: Price pierced the upper Bollinger Band then collapsed
    back through the middle band.

    WHY this matters: A close above the upper band stretches price two
    standard deviations away from its 20-day mean — an extreme that
    rarely persists. When the very next bar closes back below the
    middle band, it confirms rejection at the extreme and a sharp
    mean-reversion move is underway. This is a high-probability exit
    for swing traders holding a position that has gone parabolic.
    """
    if len(df) < 2:
        return False, ""
    for col in ("bb_upper", "bb_middle"):
        if col not in df.columns:
            return False, ""
        if pd.isna(df[col].iloc[-1]) or pd.isna(df[col].iloc[-2]):
            return False, ""

    prev_above_upper = df["Close"].iloc[-2] > df["bb_upper"].iloc[-2]
    curr_below_middle = df["Close"].iloc[-1] < df["bb_middle"].iloc[-1]

    if prev_above_upper and curr_below_middle:
        return True, (
            f"BB upper rejection: close {df['Close'].iloc[-2]:.2f} was above upper "
            f"band {df['bb_upper'].iloc[-2]:.2f}, now {df['Close'].iloc[-1]:.2f} "
            f"< middle {df['bb_middle'].iloc[-1]:.2f}"
        )
    return False, ""


def check_adx_trend_fading(df: pd.DataFrame,
                           config: dict | None = None) -> tuple[bool, str]:
    """
    OFF-RAMP: ADX drops below the weak-trend threshold.

    WHY this matters: When ADX falls below 20, whatever trend existed
    has lost all directional momentum — the stock is now in a range or
    pulling back. For positions taken because a trend was strong,
    losing the trend is reason enough to exit before the pullback
    accelerates. This is the trend-follower's discipline check: "if
    the trend is gone, so should I be."
    """
    weak = _cfg("adx_weak_threshold", config)
    if len(df) < 2 or "adx" not in df.columns:
        return False, ""
    if pd.isna(df["adx"].iloc[-1]) or pd.isna(df["adx"].iloc[-2]):
        return False, ""

    prev_adx = df["adx"].iloc[-2]
    curr_adx = df["adx"].iloc[-1]

    if prev_adx >= weak and curr_adx < weak:
        return True, (
            f"ADX trend fading: {prev_adx:.1f} -> {curr_adx:.1f} "
            f"(below weak-trend threshold {weak})"
        )
    return False, ""


def check_stochastic_bearish_cross(df: pd.DataFrame,
                                   config: dict | None = None
                                   ) -> tuple[bool, str]:
    """
    OFF-RAMP: Stochastic %K crosses below %D from overbought territory.

    WHY this matters: Mirror of the bullish cross. When stochastic
    turns down from above 80, it's an early warning that momentum is
    reversing before the rest of the market recognizes the top.
    Requiring the cross to occur IN overbought territory filters out
    mid-range noise.
    """
    overbought = _cfg("stoch_overbought", config)
    if len(df) < 2:
        return False, ""
    for col in ("stoch_k", "stoch_d"):
        if col not in df.columns:
            return False, ""
        if pd.isna(df[col].iloc[-1]) or pd.isna(df[col].iloc[-2]):
            return False, ""

    prev_k = df["stoch_k"].iloc[-2]
    prev_d = df["stoch_d"].iloc[-2]
    curr_k = df["stoch_k"].iloc[-1]
    curr_d = df["stoch_d"].iloc[-1]

    crossed_down = prev_k >= prev_d and curr_k < curr_d
    was_overbought = prev_k > overbought and prev_d > overbought

    if crossed_down and was_overbought:
        return True, (
            f"Stochastic bearish cross in overbought zone: %K {prev_k:.1f}->{curr_k:.1f}, "
            f"%D {prev_d:.1f}->{curr_d:.1f} (overbought threshold {overbought})"
        )
    return False, ""


def check_atr_volatility_stop(df: pd.DataFrame,
                              config: dict | None = None) -> tuple[bool, str]:
    """
    OFF-RAMP: Adaptive trailing stop — price drops more than Nx ATR
    from the recent 20-bar high.

    WHY this matters: A fixed-percentage trailing stop is the wrong
    tool for the job. An 8% stop is far too tight for TSLA (which can
    move 5% on a normal day) and far too loose for KO (which rarely
    moves 1%). ATR-based stops scale with each stock's own volatility,
    so the "noise vs. real pullback" decision is made on the stock's
    own terms. 3x ATR is the Chandelier Exit default popularized by
    Chuck LeBeau and is the most widely-used adaptive stop in
    professional trend-following systems.
    """
    mult = _cfg("atr_volatility_stop_mult", config)
    lookback = 20
    if len(df) < lookback:
        return False, ""
    if "atr" not in df.columns or pd.isna(df["atr"].iloc[-1]):
        return False, ""
    if "High" not in df.columns:
        return False, ""

    recent_high = df["High"].iloc[-lookback:].max()
    current = df["Close"].iloc[-1]
    atr = df["atr"].iloc[-1]
    drop = recent_high - current
    stop_distance = mult * atr

    if drop > stop_distance:
        return True, (
            f"ATR volatility stop: price {current:.2f} dropped {drop:.2f} from "
            f"recent high {recent_high:.2f} (>{mult}x ATR = {stop_distance:.2f})"
        )
    return False, ""


# ===================================================================
# Aggregate evaluation — weighted scoring + trend regime
# ===================================================================

# Signal check functions with weights. Weights reflect each signal's
# historical reliability and rarity:
#   3 = high-conviction, rare (Golden/Death Cross)
#   2 = medium reliability, well-established
#   1 = noisy or supportive-only
#
# Total maximum on-ramp score: 15
# Total maximum off-ramp score: 16
ON_RAMP_CHECKS = [
    ("Golden Cross",              check_golden_cross,             3),
    ("RSI Oversold Reversal",     check_rsi_oversold_reversal,    2),
    ("Price Reclaims SMA50",      check_price_reclaims_sma50,     2),
    ("BB Squeeze Breakout",       check_bb_squeeze_breakout,      2),
    ("ADX Bullish Trend",         check_adx_bullish_trend,        2),
    ("MACD Bullish Flip",         check_macd_bullish_flip,        1),
    ("Volume Spike (Green)",      check_volume_spike_green,       1),
    ("Stochastic Bullish Cross",  check_stochastic_bullish_cross, 1),
    ("EMA20 Reclaim + Volume",    check_ema20_reclaim_with_volume, 1),
]

OFF_RAMP_CHECKS = [
    ("Death Cross",               check_death_cross,              3),
    ("RSI Overbought Reversal",   check_rsi_overbought_reversal,  2),
    ("Price Breaks SMA50",        check_price_breaks_sma50,       2),
    ("BB Upper Band Rejection",   check_bb_upper_rejection,       2),
    ("ATR Volatility Stop",       check_atr_volatility_stop,      2),
    ("ADX Trend Fading",          check_adx_trend_fading,         2),
    ("MACD Bearish Flip",         check_macd_bearish_flip,        1),
    ("Stochastic Bearish Cross",  check_stochastic_bearish_cross, 1),
    ("Trailing Stop Breach",      check_trailing_stop,            1),
]

MAX_ON_RAMP_SCORE = sum(w for _, _, w in ON_RAMP_CHECKS)    # 15
MAX_OFF_RAMP_SCORE = sum(w for _, _, w in OFF_RAMP_CHECKS)  # 16


def compute_trend_regime(df: pd.DataFrame) -> str:
    """
    Classify the current market regime for this ticker.

    - "bullish":      Close > SMA200 AND SMA50 > SMA200 (uptrend)
    - "bearish":      Close < SMA200 AND SMA50 < SMA200 (downtrend)
    - "transitional": mixed / unclear

    WHY this matters: The single biggest source of underperformance in
    mechanical signal systems is taking long entries in bearish
    regimes — every oversold bounce looks like a bottom, until the
    next leg down. Using a simple regime filter (price above both
    moving averages) cuts out the majority of those losing trades.
    """
    if len(df) < 1:
        return "transitional"
    if "sma_short" not in df.columns or "sma_long" not in df.columns:
        return "transitional"
    close = df["Close"].iloc[-1]
    sma_s = df["sma_short"].iloc[-1]
    sma_l = df["sma_long"].iloc[-1]
    if pd.isna(sma_s) or pd.isna(sma_l) or pd.isna(close):
        return "transitional"

    if close > sma_l and sma_s > sma_l:
        return "bullish"
    if close < sma_l and sma_s < sma_l:
        return "bearish"
    return "transitional"


def evaluate_signals(df: pd.DataFrame, ticker: str,
                     config: dict | None = None) -> dict:
    """
    Run all signal checks and return an aggregate result.

    Uses weighted scoring — each signal contributes its weight (1-3)
    rather than a simple count, so rare high-conviction signals
    (Golden/Death Cross) matter more than noisy common ones (MACD
    histogram flips).

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
            "on_ramp_signals": [{"name": str, "reason": str, "weight": int}, ...],
            "off_ramp_signals": [{"name": str, "reason": str, "weight": int}, ...],
            "on_ramp_score": int,        # weighted, 0 to MAX_ON_RAMP_SCORE
            "off_ramp_score": int,       # weighted, 0 to MAX_OFF_RAMP_SCORE
            "max_on_ramp_score": int,    # 15 (for UI bars)
            "max_off_ramp_score": int,   # 16 (for UI bars)
            "signal_type": "on-ramp" | "off-ramp" | "neutral",
            "trend_regime": "bullish" | "bearish" | "transitional",
            "timestamp": str,            # ISO 8601
        }
    """
    on_ramp: list[dict] = []
    on_score = 0
    for name, fn, weight in ON_RAMP_CHECKS:
        firing, reason = fn(df, config)
        if firing:
            on_ramp.append({"name": name, "reason": reason, "weight": weight})
            on_score += weight

    off_ramp: list[dict] = []
    off_score = 0
    for name, fn, weight in OFF_RAMP_CHECKS:
        firing, reason = fn(df, config)
        if firing:
            off_ramp.append({"name": name, "reason": reason, "weight": weight})
            off_score += weight

    if on_score > off_score:
        signal_type = "on-ramp"
    elif off_score > on_score:
        signal_type = "off-ramp"
    else:
        signal_type = "neutral"

    trend_regime = compute_trend_regime(df)

    return {
        "ticker": ticker.upper(),
        "price": round(float(df["Close"].iloc[-1]), 2),
        "on_ramp_signals": on_ramp,
        "off_ramp_signals": off_ramp,
        "on_ramp_score": on_score,
        "off_ramp_score": off_score,
        "max_on_ramp_score": MAX_ON_RAMP_SCORE,
        "max_off_ramp_score": MAX_OFF_RAMP_SCORE,
        "signal_type": signal_type,
        "trend_regime": trend_regime,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
