"""
Unit tests for the signal engine.

Each test constructs a synthetic DataFrame that isolates a single signal
condition, then verifies that the signal fires (or doesn't).
"""

import numpy as np
import pandas as pd
import pytest

from src.signals import (
    MAX_OFF_RAMP_SCORE,
    MAX_ON_RAMP_SCORE,
    check_adx_bullish_trend,
    check_adx_trend_fading,
    check_atr_volatility_stop,
    check_bb_squeeze_breakout,
    check_bb_upper_rejection,
    check_death_cross,
    check_ema20_reclaim_with_volume,
    check_golden_cross,
    check_macd_bearish_flip,
    check_macd_bullish_flip,
    check_price_breaks_sma50,
    check_price_reclaims_sma50,
    check_rsi_overbought_reversal,
    check_rsi_oversold_reversal,
    check_stochastic_bearish_cross,
    check_stochastic_bullish_cross,
    check_trailing_stop,
    check_volume_spike_green,
    compute_indicators,
    compute_trend_regime,
    evaluate_signals,
)


# ---------------------------------------------------------------------------
# Helpers to build synthetic DataFrames
# ---------------------------------------------------------------------------

def _make_df(**columns) -> pd.DataFrame:
    """Shorthand to build a DataFrame from column keyword args."""
    idx = pd.date_range("2024-01-01", periods=len(list(columns.values())[0]))
    return pd.DataFrame(columns, index=idx)


def _minimal_ohlcv(close_prices: list, volumes: list | None = None,
                   opens: list | None = None) -> pd.DataFrame:
    """
    Build a minimal OHLCV DataFrame from close prices.
    Open defaults to close, High = close+1, Low = close-1.
    """
    n = len(close_prices)
    c = np.array(close_prices, dtype=float)
    o = np.array(opens, dtype=float) if opens else c.copy()
    h = c + 1
    lo = c - 1
    v = np.array(volumes, dtype=float) if volumes else np.full(n, 1_000_000.0)
    idx = pd.date_range("2024-01-01", periods=n)
    return pd.DataFrame(
        {"Open": o, "High": h, "Low": lo, "Close": c, "Volume": v},
        index=idx,
    )


# ===================================================================
# ON-RAMP signal tests
# ===================================================================

class TestRSIOversoldReversal:
    def test_fires_on_recovery(self):
        """RSI goes from 25 -> 32 => signal should fire."""
        df = _make_df(
            Close=[100.0, 101.0],
            rsi=[25.0, 32.0],
        )
        fired, reason = check_rsi_oversold_reversal(df)
        assert fired is True
        assert "oversold" in reason.lower()

    def test_no_signal_when_rsi_stays_above(self):
        """RSI never dips below 30 => no signal."""
        df = _make_df(
            Close=[100.0, 101.0],
            rsi=[40.0, 42.0],
        )
        fired, _ = check_rsi_oversold_reversal(df)
        assert fired is False

    def test_no_signal_when_rsi_stays_below(self):
        """RSI goes from 25 -> 28 (still below 30) => no signal."""
        df = _make_df(
            Close=[100.0, 99.0],
            rsi=[25.0, 28.0],
        )
        fired, _ = check_rsi_oversold_reversal(df)
        assert fired is False

    def test_custom_threshold(self):
        """Per-ticker override: threshold=25, RSI 24->26 should fire."""
        df = _make_df(Close=[100.0, 101.0], rsi=[24.0, 26.0])
        fired, _ = check_rsi_oversold_reversal(df, {"rsi_oversold": 25})
        assert fired is True


class TestPriceReclaimsSMA50:
    def test_fires_on_reclaim(self):
        """Price was below SMA50 yesterday, above today => signal."""
        df = _make_df(
            Close=[98.0, 102.0],
            sma_short=[100.0, 100.0],
        )
        fired, reason = check_price_reclaims_sma50(df)
        assert fired is True
        assert "reclaimed" in reason.lower()

    def test_no_signal_when_above_both_days(self):
        """Price was above SMA50 on both days => no signal."""
        df = _make_df(
            Close=[102.0, 103.0],
            sma_short=[100.0, 100.0],
        )
        fired, _ = check_price_reclaims_sma50(df)
        assert fired is False


class TestGoldenCross:
    def test_fires_on_crossover(self):
        """SMA50 crosses above SMA200 => golden cross."""
        df = _make_df(
            Close=[100.0, 101.0],
            sma_short=[99.0, 101.0],
            sma_long=[100.0, 100.0],
        )
        fired, reason = check_golden_cross(df)
        assert fired is True
        assert "golden" in reason.lower()

    def test_no_signal_when_already_above(self):
        """SMA50 was already above SMA200 => no new signal."""
        df = _make_df(
            Close=[100.0, 101.0],
            sma_short=[101.0, 102.0],
            sma_long=[100.0, 100.0],
        )
        fired, _ = check_golden_cross(df)
        assert fired is False


class TestMACDBullishFlip:
    def test_fires_on_flip(self):
        """MACD histogram goes from -0.5 to +0.2 => bullish flip."""
        df = _make_df(
            Close=[100.0, 101.0],
            macd_hist=[-0.5, 0.2],
        )
        fired, reason = check_macd_bullish_flip(df)
        assert fired is True
        assert "bullish" in reason.lower()

    def test_no_signal_when_already_positive(self):
        """Histogram was already positive => no new signal."""
        df = _make_df(
            Close=[100.0, 101.0],
            macd_hist=[0.3, 0.5],
        )
        fired, _ = check_macd_bullish_flip(df)
        assert fired is False


class TestVolumeSpikeGreen:
    def test_fires_on_spike(self):
        """Volume = 3x avg on green candle => signal."""
        df = _make_df(
            Open=[100.0],
            Close=[105.0],
            High=[106.0],
            Low=[99.0],
            Volume=[3_000_000.0],
            vol_avg_20=[1_000_000.0],
        )
        fired, reason = check_volume_spike_green(df)
        assert fired is True
        assert "spike" in reason.lower()

    def test_no_signal_on_red_candle(self):
        """Volume spike but close < open (red candle) => no signal."""
        df = _make_df(
            Open=[105.0],
            Close=[100.0],
            High=[106.0],
            Low=[99.0],
            Volume=[3_000_000.0],
            vol_avg_20=[1_000_000.0],
        )
        fired, _ = check_volume_spike_green(df)
        assert fired is False

    def test_no_signal_normal_volume(self):
        """Green candle but volume is only 1.5x avg => no signal."""
        df = _make_df(
            Open=[100.0],
            Close=[105.0],
            High=[106.0],
            Low=[99.0],
            Volume=[1_500_000.0],
            vol_avg_20=[1_000_000.0],
        )
        fired, _ = check_volume_spike_green(df)
        assert fired is False


# ===================================================================
# OFF-RAMP signal tests
# ===================================================================

class TestRSIOverboughtReversal:
    def test_fires_on_drop(self):
        """RSI goes from 75 -> 68 => overbought reversal."""
        df = _make_df(
            Close=[100.0, 99.0],
            rsi=[75.0, 68.0],
        )
        fired, reason = check_rsi_overbought_reversal(df)
        assert fired is True
        assert "overbought" in reason.lower()

    def test_no_signal_when_rsi_stays_below(self):
        """RSI never goes above 70 => no signal."""
        df = _make_df(
            Close=[100.0, 99.0],
            rsi=[60.0, 58.0],
        )
        fired, _ = check_rsi_overbought_reversal(df)
        assert fired is False


class TestPriceBreaksSMA50:
    def test_fires_on_break(self):
        """Price was above SMA50 yesterday, below today => signal."""
        df = _make_df(
            Close=[102.0, 98.0],
            sma_short=[100.0, 100.0],
        )
        fired, reason = check_price_breaks_sma50(df)
        assert fired is True
        assert "broke below" in reason.lower()

    def test_no_signal_when_below_both_days(self):
        """Price was below SMA50 on both days => no signal."""
        df = _make_df(
            Close=[98.0, 97.0],
            sma_short=[100.0, 100.0],
        )
        fired, _ = check_price_breaks_sma50(df)
        assert fired is False


class TestDeathCross:
    def test_fires_on_crossover(self):
        """SMA50 crosses below SMA200 => death cross."""
        df = _make_df(
            Close=[100.0, 99.0],
            sma_short=[101.0, 99.0],
            sma_long=[100.0, 100.0],
        )
        fired, reason = check_death_cross(df)
        assert fired is True
        assert "death" in reason.lower()

    def test_no_signal_when_already_below(self):
        """SMA50 was already below SMA200 => no new signal."""
        df = _make_df(
            Close=[100.0, 99.0],
            sma_short=[98.0, 97.0],
            sma_long=[100.0, 100.0],
        )
        fired, _ = check_death_cross(df)
        assert fired is False


class TestMACDBearishFlip:
    def test_fires_on_flip(self):
        """MACD histogram goes from +0.3 to -0.1 => bearish flip."""
        df = _make_df(
            Close=[100.0, 99.0],
            macd_hist=[0.3, -0.1],
        )
        fired, reason = check_macd_bearish_flip(df)
        assert fired is True
        assert "bearish" in reason.lower()

    def test_no_signal_when_already_negative(self):
        """Histogram was already negative => no new signal."""
        df = _make_df(
            Close=[100.0, 99.0],
            macd_hist=[-0.2, -0.5],
        )
        fired, _ = check_macd_bearish_flip(df)
        assert fired is False


class TestTrailingStop:
    def test_fires_on_breach(self):
        """Price drops >8% from rolling high => trailing stop fires."""
        # Build 20 bars where the high was 100, then current is 90 (10% drop)
        closes = [100.0] * 19 + [90.0]
        highs = [100.0] * 19 + [90.0]
        df = _make_df(Close=closes, High=highs)
        fired, reason = check_trailing_stop(df)
        assert fired is True
        assert "trailing stop" in reason.lower()

    def test_no_signal_within_threshold(self):
        """Price drops 5% from high (below 8% threshold) => no signal."""
        closes = [100.0] * 19 + [95.0]
        highs = [100.0] * 19 + [95.0]
        df = _make_df(Close=closes, High=highs)
        fired, _ = check_trailing_stop(df)
        assert fired is False

    def test_custom_threshold(self):
        """With trailing_stop_pct=4, a 5% drop should fire."""
        closes = [100.0] * 19 + [95.0]
        highs = [100.0] * 19 + [95.0]
        df = _make_df(Close=closes, High=highs)
        fired, _ = check_trailing_stop(df, {"trailing_stop_pct": 4.0})
        assert fired is True

    def test_uses_high_not_close(self):
        """The rolling high must come from the High column, not Close.

        If the intraday High = 110 but Close = 100 throughout, a later 100->92
        drop should still measure against 110 (16% drawdown), firing the stop.
        """
        closes = [100.0] * 19 + [92.0]
        highs = [110.0] * 19 + [92.0]
        df = _make_df(Close=closes, High=highs)
        fired, reason = check_trailing_stop(df)
        assert fired is True
        # The reason should reference the actual High of 110, not 100
        assert "110" in reason


# ===================================================================
# Integration: compute_indicators and evaluate_signals
# ===================================================================

class TestComputeIndicators:
    def test_adds_all_columns(self):
        """compute_indicators should add v1 + v2 indicator columns."""
        # Need enough data for SMA200 (200+ rows)
        n = 250
        np.random.seed(42)
        prices = 100 + np.cumsum(np.random.randn(n) * 0.5)
        df = _minimal_ohlcv(prices.tolist())

        result = compute_indicators(df)

        expected_cols = [
            # v1
            "rsi", "sma_short", "sma_long", "macd",
            "macd_signal", "macd_hist", "vol_avg_20",
            # v2
            "ema_fast", "bb_upper", "bb_middle", "bb_lower", "bb_bandwidth",
            "atr", "adx", "plus_di", "minus_di", "stoch_k", "stoch_d",
        ]
        for col in expected_cols:
            assert col in result.columns, f"Missing column: {col}"

    def test_rsi_range(self):
        """RSI must be between 0 and 100."""
        n = 250
        np.random.seed(7)
        prices = 100 + np.cumsum(np.random.randn(n) * 0.5)
        df = compute_indicators(_minimal_ohlcv(prices.tolist()))
        valid = df["rsi"].dropna()
        assert (valid >= 0).all() and (valid <= 100).all()

    def test_adx_range(self):
        """ADX, +DI, -DI must all be between 0 and 100."""
        n = 250
        np.random.seed(11)
        prices = 100 + np.cumsum(np.random.randn(n) * 0.5)
        df = compute_indicators(_minimal_ohlcv(prices.tolist()))
        for col in ("adx", "plus_di", "minus_di"):
            valid = df[col].dropna()
            assert (valid >= 0).all() and (valid <= 100).all(), f"{col} out of range"

    def test_stochastic_range(self):
        """Stochastic %K and %D must be between 0 and 100."""
        n = 250
        np.random.seed(13)
        prices = 100 + np.cumsum(np.random.randn(n) * 0.5)
        df = compute_indicators(_minimal_ohlcv(prices.tolist()))
        for col in ("stoch_k", "stoch_d"):
            valid = df[col].dropna()
            assert (valid >= 0).all() and (valid <= 100).all(), f"{col} out of range"

    def test_atr_positive(self):
        """ATR must be strictly positive (volatility cannot be negative)."""
        n = 250
        np.random.seed(17)
        prices = 100 + np.cumsum(np.random.randn(n) * 0.5)
        df = compute_indicators(_minimal_ohlcv(prices.tolist()))
        valid = df["atr"].dropna()
        assert (valid > 0).all()

    def test_bollinger_ordering(self):
        """Upper band >= middle >= lower at every valid row."""
        n = 250
        np.random.seed(19)
        prices = 100 + np.cumsum(np.random.randn(n) * 0.5)
        df = compute_indicators(_minimal_ohlcv(prices.tolist()))
        valid = df[["bb_upper", "bb_middle", "bb_lower"]].dropna()
        assert (valid["bb_upper"] >= valid["bb_middle"]).all()
        assert (valid["bb_middle"] >= valid["bb_lower"]).all()


# ===================================================================
# v2 ON-RAMP signal tests
# ===================================================================

class TestBBSqueezeBreakout:
    def _make_squeeze_df(self, bandwidth_values, close, upper):
        """Helper to build a df with the full bandwidth history + current bar."""
        n = len(bandwidth_values)
        df = pd.DataFrame({
            "Close": [100.0] * (n - 1) + [close],
            "bb_upper": [105.0] * (n - 1) + [upper],
            "bb_middle": [100.0] * n,
            "bb_lower": [95.0] * n,
            "bb_bandwidth": bandwidth_values,
        }, index=pd.date_range("2024-01-01", periods=n))
        return df

    def test_fires_on_breakout(self):
        """Bandwidth mostly wide, but squeezed in last 5 bars, then breakout."""
        # 116 wide bandwidth values, then 5 squeeze values, then a breakout bar
        bws = [0.10] * 115 + [0.02] * 5 + [0.03]  # 121 total
        df = self._make_squeeze_df(bws, close=106.0, upper=104.0)
        fired, reason = check_bb_squeeze_breakout(df)
        assert fired is True
        assert "squeeze" in reason.lower()

    def test_no_signal_when_not_squeezed(self):
        """Bandwidth varies but never compresses near its bottom => no squeeze."""
        # Low percentile is ~0.05, but the last 5 bars are wide (>=0.15).
        bws = [0.05 + (i % 10) * 0.01 for i in range(116)] + [0.15] * 5
        df = self._make_squeeze_df(bws, close=106.0, upper=104.0)
        fired, _ = check_bb_squeeze_breakout(df)
        assert fired is False

    def test_no_signal_when_no_breakout(self):
        """Bandwidth squeezed but price doesn't close above upper band."""
        bws = [0.10] * 115 + [0.02] * 5 + [0.03]
        df = self._make_squeeze_df(bws, close=102.0, upper=104.0)
        fired, _ = check_bb_squeeze_breakout(df)
        assert fired is False


class TestADXBullishTrend:
    def test_fires_on_rising_adx(self):
        """ADX > 25, +DI > -DI, and ADX rising => signal."""
        df = _make_df(
            Close=[100.0, 101.0],
            adx=[26.0, 28.0],
            plus_di=[30.0, 32.0],
            minus_di=[15.0, 14.0],
        )
        fired, reason = check_adx_bullish_trend(df)
        assert fired is True
        assert "bullish" in reason.lower()

    def test_no_signal_when_adx_falling(self):
        """ADX is falling => no signal even if > 25."""
        df = _make_df(
            Close=[100.0, 101.0],
            adx=[30.0, 28.0],
            plus_di=[30.0, 32.0],
            minus_di=[15.0, 14.0],
        )
        fired, _ = check_adx_bullish_trend(df)
        assert fired is False

    def test_no_signal_when_minus_di_higher(self):
        """-DI above +DI => bearish directional dominance => no signal."""
        df = _make_df(
            Close=[100.0, 101.0],
            adx=[26.0, 28.0],
            plus_di=[15.0, 16.0],
            minus_di=[30.0, 32.0],
        )
        fired, _ = check_adx_bullish_trend(df)
        assert fired is False

    def test_no_signal_below_threshold(self):
        """ADX < 25 => no trend strength => no signal."""
        df = _make_df(
            Close=[100.0, 101.0],
            adx=[15.0, 18.0],
            plus_di=[30.0, 32.0],
            minus_di=[15.0, 14.0],
        )
        fired, _ = check_adx_bullish_trend(df)
        assert fired is False


class TestStochasticBullishCross:
    def test_fires_on_cross_from_oversold(self):
        """%K crosses above %D, both were below 20."""
        df = _make_df(
            Close=[100.0, 101.0],
            stoch_k=[15.0, 22.0],
            stoch_d=[18.0, 19.0],
        )
        fired, reason = check_stochastic_bullish_cross(df)
        assert fired is True
        assert "bullish cross" in reason.lower()

    def test_no_signal_if_not_oversold(self):
        """Cross happens but above 20 => no signal."""
        df = _make_df(
            Close=[100.0, 101.0],
            stoch_k=[40.0, 55.0],
            stoch_d=[50.0, 52.0],
        )
        fired, _ = check_stochastic_bullish_cross(df)
        assert fired is False

    def test_no_signal_without_cross(self):
        """%K never crosses %D => no signal."""
        df = _make_df(
            Close=[100.0, 101.0],
            stoch_k=[15.0, 18.0],
            stoch_d=[10.0, 12.0],
        )
        fired, _ = check_stochastic_bullish_cross(df)
        assert fired is False


class TestEMA20ReclaimWithVolume:
    def test_fires_on_reclaim_with_volume(self):
        """Close reclaims EMA20 with volume > 1.5x avg."""
        df = _make_df(
            Close=[98.0, 102.0],
            Volume=[1_000_000.0, 2_500_000.0],
            ema_fast=[100.0, 100.0],
            vol_avg_20=[1_000_000.0, 1_000_000.0],
        )
        fired, reason = check_ema20_reclaim_with_volume(df)
        assert fired is True
        assert "reclaimed" in reason.lower()

    def test_no_signal_on_low_volume(self):
        """Reclaim without volume confirmation => no signal."""
        df = _make_df(
            Close=[98.0, 102.0],
            Volume=[1_000_000.0, 1_100_000.0],
            ema_fast=[100.0, 100.0],
            vol_avg_20=[1_000_000.0, 1_000_000.0],
        )
        fired, _ = check_ema20_reclaim_with_volume(df)
        assert fired is False

    def test_no_signal_without_reclaim(self):
        """Volume spike but price doesn't cross EMA20 => no signal."""
        df = _make_df(
            Close=[95.0, 96.0],
            Volume=[1_000_000.0, 3_000_000.0],
            ema_fast=[100.0, 100.0],
            vol_avg_20=[1_000_000.0, 1_000_000.0],
        )
        fired, _ = check_ema20_reclaim_with_volume(df)
        assert fired is False


# ===================================================================
# v2 OFF-RAMP signal tests
# ===================================================================

class TestBBUpperRejection:
    def test_fires_on_rejection(self):
        """Prev close above upper band, current close below middle."""
        df = _make_df(
            Close=[106.0, 99.0],
            bb_upper=[105.0, 105.0],
            bb_middle=[100.0, 100.0],
        )
        fired, reason = check_bb_upper_rejection(df)
        assert fired is True
        assert "rejection" in reason.lower()

    def test_no_signal_when_above_middle(self):
        """Current close still above middle => no rejection."""
        df = _make_df(
            Close=[106.0, 103.0],
            bb_upper=[105.0, 105.0],
            bb_middle=[100.0, 100.0],
        )
        fired, _ = check_bb_upper_rejection(df)
        assert fired is False


class TestADXTrendFading:
    def test_fires_on_fade(self):
        """ADX drops from 22 to 18 (below weak threshold of 20)."""
        df = _make_df(
            Close=[100.0, 99.0],
            adx=[22.0, 18.0],
        )
        fired, reason = check_adx_trend_fading(df)
        assert fired is True
        assert "fading" in reason.lower()

    def test_no_signal_when_still_strong(self):
        """ADX stays above 20 => no fade."""
        df = _make_df(
            Close=[100.0, 99.0],
            adx=[25.0, 22.0],
        )
        fired, _ = check_adx_trend_fading(df)
        assert fired is False


class TestStochasticBearishCross:
    def test_fires_on_cross_from_overbought(self):
        """%K crosses below %D, both were above 80."""
        df = _make_df(
            Close=[100.0, 99.0],
            stoch_k=[85.0, 78.0],
            stoch_d=[82.0, 81.0],
        )
        fired, reason = check_stochastic_bearish_cross(df)
        assert fired is True
        assert "bearish cross" in reason.lower()

    def test_no_signal_below_overbought(self):
        """Cross happens in mid-range => no signal."""
        df = _make_df(
            Close=[100.0, 99.0],
            stoch_k=[50.0, 45.0],
            stoch_d=[48.0, 48.0],
        )
        fired, _ = check_stochastic_bearish_cross(df)
        assert fired is False


class TestATRVolatilityStop:
    def test_fires_on_stop(self):
        """Price drops more than 3x ATR from recent 20-bar high."""
        # 20 bars: high stays at 100, ATR = 1.0, current price = 95 (5 > 3*1)
        highs = [100.0] * 19 + [95.0]
        closes = [100.0] * 19 + [95.0]
        atr_vals = [1.0] * 20
        df = _make_df(Close=closes, High=highs, atr=atr_vals)
        fired, reason = check_atr_volatility_stop(df)
        assert fired is True
        assert "atr" in reason.lower()

    def test_no_signal_small_drop(self):
        """Drop is only 2x ATR => within normal volatility."""
        highs = [100.0] * 19 + [98.0]
        closes = [100.0] * 19 + [98.0]
        atr_vals = [1.0] * 20
        df = _make_df(Close=closes, High=highs, atr=atr_vals)
        fired, _ = check_atr_volatility_stop(df)
        assert fired is False

    def test_scales_with_volatility(self):
        """High-ATR stock tolerates a larger drop."""
        # Same 5-point drop but ATR is 3 => only 1.67x, should NOT fire
        highs = [100.0] * 19 + [95.0]
        closes = [100.0] * 19 + [95.0]
        atr_vals = [3.0] * 20
        df = _make_df(Close=closes, High=highs, atr=atr_vals)
        fired, _ = check_atr_volatility_stop(df)
        assert fired is False


# ===================================================================
# Trend regime and weighted scoring
# ===================================================================

class TestTrendRegime:
    def test_bullish(self):
        """Close > SMA200 and SMA50 > SMA200 => bullish."""
        df = _make_df(
            Close=[110.0],
            sma_short=[105.0],
            sma_long=[100.0],
        )
        assert compute_trend_regime(df) == "bullish"

    def test_bearish(self):
        """Close < SMA200 and SMA50 < SMA200 => bearish."""
        df = _make_df(
            Close=[90.0],
            sma_short=[95.0],
            sma_long=[100.0],
        )
        assert compute_trend_regime(df) == "bearish"

    def test_transitional_mixed(self):
        """Close above SMA200 but SMA50 below => transitional."""
        df = _make_df(
            Close=[101.0],
            sma_short=[99.0],
            sma_long=[100.0],
        )
        assert compute_trend_regime(df) == "transitional"

    def test_transitional_when_nan(self):
        """If SMAs are NaN (insufficient history) => transitional."""
        df = pd.DataFrame({
            "Close": [100.0],
            "sma_short": [np.nan],
            "sma_long": [np.nan],
        }, index=pd.date_range("2024-01-01", periods=1))
        assert compute_trend_regime(df) == "transitional"


class TestEvaluateSignals:
    def test_scoring(self):
        """Weighted score: RSI oversold (w=2) + MACD bullish flip (w=1) => 3."""
        df = _make_df(
            Open=[100.0, 100.0],
            High=[101.0, 102.0],
            Low=[99.0, 99.0],
            Close=[100.0, 101.0],
            Volume=[1_000_000.0, 1_000_000.0],
            rsi=[25.0, 32.0],
            sma_short=[105.0, 105.0],  # price below => no reclaim
            sma_long=[110.0, 110.0],
            macd=[0.0, 0.0],
            macd_signal=[0.0, 0.0],
            macd_hist=[-0.5, 0.2],
            vol_avg_20=[1_000_000.0, 1_000_000.0],
        )

        result = evaluate_signals(df, "TEST")
        assert result["ticker"] == "TEST"
        # Weighted: RSI oversold reversal (2) + MACD bullish flip (1) = 3
        assert result["on_ramp_score"] == 3
        assert result["off_ramp_score"] == 0
        assert result["signal_type"] == "on-ramp"
        assert len(result["on_ramp_signals"]) == 2
        # Each signal carries its weight
        weights = {s["name"]: s["weight"] for s in result["on_ramp_signals"]}
        assert weights["RSI Oversold Reversal"] == 2
        assert weights["MACD Bullish Flip"] == 1
        # New output keys
        assert result["max_on_ramp_score"] == MAX_ON_RAMP_SCORE
        assert result["max_off_ramp_score"] == MAX_OFF_RAMP_SCORE
        assert "trend_regime" in result

    def test_neutral_when_balanced(self):
        """No crossovers, no signals firing => neutral."""
        df = _make_df(
            Open=[100.0, 100.0],
            High=[101.0, 101.0],
            Low=[99.0, 99.0],
            Close=[100.0, 100.0],
            Volume=[1_000_000.0, 1_000_000.0],
            rsi=[50.0, 50.0],
            sma_short=[100.0, 100.0],
            sma_long=[95.0, 95.0],
            macd=[0.1, 0.1],
            macd_signal=[0.0, 0.0],
            macd_hist=[0.1, 0.1],
            vol_avg_20=[1_000_000.0, 1_000_000.0],
        )
        result = evaluate_signals(df, "FLAT")
        assert result["signal_type"] == "neutral"
        assert result["on_ramp_score"] == 0
        assert result["off_ramp_score"] == 0


class TestWeightedScoring:
    def test_max_scores(self):
        """Sanity check on the max-score constants."""
        assert MAX_ON_RAMP_SCORE == 15
        assert MAX_OFF_RAMP_SCORE == 16

    def test_golden_cross_alone_scores_3(self):
        """Golden Cross is weight 3, so it alone produces an on-ramp score of 3."""
        # Use enough rows to avoid BB squeeze breakout check running on short data
        df = _make_df(
            Open=[100.0, 100.0],
            High=[101.0, 102.0],
            Low=[99.0, 99.0],
            Close=[100.0, 101.0],
            Volume=[1_000_000.0, 1_000_000.0],
            rsi=[50.0, 50.0],
            sma_short=[99.0, 101.0],
            sma_long=[100.0, 100.0],
            macd=[0.0, 0.0],
            macd_signal=[0.0, 0.0],
            macd_hist=[0.1, 0.1],
            vol_avg_20=[1_000_000.0, 1_000_000.0],
        )
        result = evaluate_signals(df, "TEST")
        assert result["on_ramp_score"] == 3
        assert any(s["name"] == "Golden Cross" and s["weight"] == 3
                   for s in result["on_ramp_signals"])

    def test_death_cross_alone_scores_3(self):
        """Death Cross is weight 3, so it alone produces an off-ramp score of 3."""
        df = _make_df(
            Open=[100.0, 100.0],
            High=[101.0, 102.0],
            Low=[99.0, 99.0],
            Close=[100.0, 99.0],
            Volume=[1_000_000.0, 1_000_000.0],
            rsi=[50.0, 50.0],
            sma_short=[101.0, 99.0],
            sma_long=[100.0, 100.0],
            macd=[0.0, 0.0],
            macd_signal=[0.0, 0.0],
            macd_hist=[-0.1, -0.1],
            vol_avg_20=[1_000_000.0, 1_000_000.0],
        )
        result = evaluate_signals(df, "TEST")
        assert result["off_ramp_score"] == 3
        assert any(s["name"] == "Death Cross" and s["weight"] == 3
                   for s in result["off_ramp_signals"])
