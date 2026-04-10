"""
Unit tests for the signal engine.

Each test constructs a synthetic DataFrame that isolates a single signal
condition, then verifies that the signal fires (or doesn't).
"""

import numpy as np
import pandas as pd
import pytest

from src.signals import (
    check_death_cross,
    check_golden_cross,
    check_macd_bearish_flip,
    check_macd_bullish_flip,
    check_price_breaks_sma50,
    check_price_reclaims_sma50,
    check_rsi_overbought_reversal,
    check_rsi_oversold_reversal,
    check_trailing_stop,
    check_volume_spike_green,
    compute_indicators,
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
        df = _make_df(Close=closes)
        fired, reason = check_trailing_stop(df)
        assert fired is True
        assert "trailing stop" in reason.lower()

    def test_no_signal_within_threshold(self):
        """Price drops 5% from high (below 8% threshold) => no signal."""
        closes = [100.0] * 19 + [95.0]
        df = _make_df(Close=closes)
        fired, _ = check_trailing_stop(df)
        assert fired is False

    def test_custom_threshold(self):
        """With trailing_stop_pct=4, a 5% drop should fire."""
        closes = [100.0] * 19 + [95.0]
        df = _make_df(Close=closes)
        fired, _ = check_trailing_stop(df, {"trailing_stop_pct": 4.0})
        assert fired is True


# ===================================================================
# Integration: compute_indicators and evaluate_signals
# ===================================================================

class TestComputeIndicators:
    def test_adds_all_columns(self):
        """compute_indicators should add rsi, sma_short, sma_long, macd, etc."""
        # Need enough data for SMA200 (200+ rows)
        n = 250
        np.random.seed(42)
        prices = 100 + np.cumsum(np.random.randn(n) * 0.5)
        df = _minimal_ohlcv(prices.tolist())

        result = compute_indicators(df)

        expected_cols = ["rsi", "sma_short", "sma_long", "macd",
                         "macd_signal", "macd_hist", "vol_avg_20"]
        for col in expected_cols:
            assert col in result.columns, f"Missing column: {col}"

        # RSI should be between 0 and 100 where it's not NaN
        valid_rsi = result["rsi"].dropna()
        assert (valid_rsi >= 0).all() and (valid_rsi <= 100).all()


class TestEvaluateSignals:
    def test_scoring(self):
        """Evaluate signals returns correct scores and type."""
        # Build a df that triggers exactly 2 on-ramp signals:
        # RSI oversold reversal + MACD bullish flip
        df = _make_df(
            Open=[100.0, 100.0],
            High=[101.0, 102.0],
            Low=[99.0, 99.0],
            Close=[100.0, 101.0],
            Volume=[1_000_000.0, 1_000_000.0],
            rsi=[25.0, 32.0],
            sma_short=[105.0, 105.0],  # price below SMA => no reclaim
            sma_long=[110.0, 110.0],
            macd=[0.0, 0.0],
            macd_signal=[0.0, 0.0],
            macd_hist=[-0.5, 0.2],
            vol_avg_20=[1_000_000.0, 1_000_000.0],
        )

        result = evaluate_signals(df, "TEST")
        assert result["ticker"] == "TEST"
        assert result["on_ramp_score"] == 2
        assert result["off_ramp_score"] == 0
        assert result["signal_type"] == "on-ramp"
        assert len(result["on_ramp_signals"]) == 2

    def test_neutral_when_balanced(self):
        """Equal on-ramp and off-ramp scores => neutral."""
        # No crossovers, no signals firing
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
