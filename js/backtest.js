/**
 * Client-side backtesting engine (v2 — weighted scoring + regime filter)
 *
 * Mirrors src/signals.py and src/backtest.py. All changes to the Python
 * signal engine MUST be reflected here or the in-browser backtester will
 * diverge from server-side results.
 */

// ---------------------------------------------------------------------
// Thresholds & weights
// ---------------------------------------------------------------------

// Weighted scoring: max on-ramp = 15, max off-ramp = 16. A threshold of
// 4 means two medium signals (2+2), or one high-conviction plus one
// supportive (3+1), or four low-weight supportive signals.
const ENTRY_THRESHOLD = 4;
const EXIT_THRESHOLD = 4;

const MAX_ON_RAMP_SCORE = 15;
const MAX_OFF_RAMP_SCORE = 16;

// Signal weights — must match src/signals.py
const WEIGHTS = {
    // on-ramp
    golden_cross: 3,
    rsi_oversold_reversal: 2,
    price_reclaims_sma50: 2,
    bb_squeeze_breakout: 2,
    adx_bullish_trend: 2,
    macd_bullish_flip: 1,
    volume_spike_green: 1,
    stochastic_bullish_cross: 1,
    ema20_reclaim_with_volume: 1,
    // off-ramp
    death_cross: 3,
    rsi_overbought_reversal: 2,
    price_breaks_sma50: 2,
    bb_upper_rejection: 2,
    atr_volatility_stop: 2,
    adx_trend_fading: 2,
    macd_bearish_flip: 1,
    stochastic_bearish_cross: 1,
    trailing_stop: 1,
};

// Default config — must match DEFAULTS in src/signals.py
const DEFAULT_CONFIG = {
    rsi_oversold: 30,
    rsi_overbought: 70,
    trailing_stop_pct: 8.0,
    volume_spike_multiplier: 2.0,
    // v2
    bb_squeeze_lookback: 120,
    bb_squeeze_percentile: 20,
    adx_threshold: 25,
    adx_weak_threshold: 20,
    stoch_oversold: 20,
    stoch_overbought: 80,
    ema_reclaim_vol_mult: 1.5,
    atr_volatility_stop_mult: 3.0,
};

// ---------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------

function isNum(x) {
    return x !== null && x !== undefined && !Number.isNaN(x);
}

function lastN(arr, n) {
    return arr.slice(Math.max(0, arr.length - n));
}

// ---------------------------------------------------------------------
// ON-RAMP checks
// ---------------------------------------------------------------------

function checkRSIOversoldReversal(rsi, config) {
    const threshold = config.rsi_oversold ?? DEFAULT_CONFIG.rsi_oversold;
    if (rsi.length < 2) return false;
    const prev = rsi[rsi.length - 2];
    const curr = rsi[rsi.length - 1];
    return isNum(prev) && isNum(curr) && prev < threshold && curr >= threshold;
}

function checkPriceReclaimsSMA50(closes, sma50) {
    if (closes.length < 2 || sma50.length < 2) return false;
    const currClose = closes[closes.length - 1];
    const prevClose = closes[closes.length - 2];
    const currSMA = sma50[sma50.length - 1];
    const prevSMA = sma50[sma50.length - 2];
    return isNum(prevSMA) && isNum(currSMA) &&
           prevClose < prevSMA && currClose >= currSMA;
}

function checkGoldenCross(sma50, sma200) {
    if (sma50.length < 2 || sma200.length < 2) return false;
    const currShort = sma50[sma50.length - 1];
    const prevShort = sma50[sma50.length - 2];
    const currLong = sma200[sma200.length - 1];
    const prevLong = sma200[sma200.length - 2];
    return isNum(prevShort) && isNum(prevLong) && isNum(currShort) && isNum(currLong) &&
           prevShort < prevLong && currShort >= currLong;
}

function checkMACDBullishFlip(macdHist) {
    if (macdHist.length < 2) return false;
    const prev = macdHist[macdHist.length - 2];
    const curr = macdHist[macdHist.length - 1];
    return isNum(prev) && isNum(curr) && prev < 0 && curr >= 0;
}

function checkVolumeSpikeGreen(volumes, closes, opens, config) {
    const multiplier = config.volume_spike_multiplier ?? DEFAULT_CONFIG.volume_spike_multiplier;
    if (volumes.length < 21 || closes.length < 1) return false;
    const currVol = volumes[volumes.length - 1];
    const avgVol = volumes.slice(-21, -1).reduce((a, b) => a + b, 0) / 20;
    // Green candle = close > open (same as Python)
    const isGreen = closes[closes.length - 1] > opens[opens.length - 1];
    return isGreen && currVol > avgVol * multiplier;
}

function checkBBSqueezeBreakout(closes, bbUpper, bbBandwidth, config) {
    const lookback = config.bb_squeeze_lookback ?? DEFAULT_CONFIG.bb_squeeze_lookback;
    const pctThreshold = config.bb_squeeze_percentile ?? DEFAULT_CONFIG.bb_squeeze_percentile;
    if (closes.length < lookback + 1) return false;

    const currClose = closes[closes.length - 1];
    const currUpper = bbUpper[bbUpper.length - 1];
    if (!isNum(currClose) || !isNum(currUpper)) return false;

    // Quantile of the last `lookback` bandwidth values (ignore nulls)
    const window = bbBandwidth.slice(-lookback).filter(isNum);
    if (window.length === 0) return false;
    const sorted = [...window].sort((a, b) => a - b);
    const idx = Math.max(0, Math.floor((pctThreshold / 100) * (sorted.length - 1)));
    const threshold = sorted[idx];

    // Was bandwidth in squeeze territory within the last 5 bars?
    const recent = bbBandwidth.slice(-5).filter(isNum);
    const wasSqueezed = recent.some(v => v <= threshold);
    const isBreakout = currClose > currUpper;
    return wasSqueezed && isBreakout;
}

function checkADXBullishTrend(adx, plusDI, minusDI, config) {
    const threshold = config.adx_threshold ?? DEFAULT_CONFIG.adx_threshold;
    if (adx.length < 2 || plusDI.length < 1 || minusDI.length < 1) return false;
    const currAdx = adx[adx.length - 1];
    const prevAdx = adx[adx.length - 2];
    const pdi = plusDI[plusDI.length - 1];
    const mdi = minusDI[minusDI.length - 1];
    if (!isNum(currAdx) || !isNum(prevAdx) || !isNum(pdi) || !isNum(mdi)) return false;
    return currAdx > threshold && pdi > mdi && currAdx > prevAdx;
}

function checkStochasticBullishCross(stochK, stochD, config) {
    const oversold = config.stoch_oversold ?? DEFAULT_CONFIG.stoch_oversold;
    if (stochK.length < 2 || stochD.length < 2) return false;
    const prevK = stochK[stochK.length - 2];
    const prevD = stochD[stochD.length - 2];
    const currK = stochK[stochK.length - 1];
    const currD = stochD[stochD.length - 1];
    if (!isNum(prevK) || !isNum(prevD) || !isNum(currK) || !isNum(currD)) return false;
    const crossedUp = prevK <= prevD && currK > currD;
    const wasOversold = prevK < oversold && prevD < oversold;
    return crossedUp && wasOversold;
}

function checkEMA20ReclaimWithVolume(closes, emaFast, volumes, volAvg20, config) {
    const mult = config.ema_reclaim_vol_mult ?? DEFAULT_CONFIG.ema_reclaim_vol_mult;
    if (closes.length < 2 || emaFast.length < 2 || volumes.length < 1) return false;
    const prevEma = emaFast[emaFast.length - 2];
    const currEma = emaFast[emaFast.length - 1];
    const prevClose = closes[closes.length - 2];
    const currClose = closes[closes.length - 1];
    const currVol = volumes[volumes.length - 1];
    const avgVol = volAvg20[volAvg20.length - 1];
    if (!isNum(prevEma) || !isNum(currEma) || !isNum(avgVol)) return false;
    const prevBelow = prevClose < prevEma;
    const currAbove = currClose >= currEma;
    const volOk = currVol > mult * avgVol;
    return prevBelow && currAbove && volOk;
}

// ---------------------------------------------------------------------
// OFF-RAMP checks
// ---------------------------------------------------------------------

function checkRSIOverboughtReversal(rsi, config) {
    const threshold = config.rsi_overbought ?? DEFAULT_CONFIG.rsi_overbought;
    if (rsi.length < 2) return false;
    const prev = rsi[rsi.length - 2];
    const curr = rsi[rsi.length - 1];
    return isNum(prev) && isNum(curr) && prev > threshold && curr <= threshold;
}

function checkPriceBreaksSMA50(closes, sma50) {
    if (closes.length < 2 || sma50.length < 2) return false;
    const currClose = closes[closes.length - 1];
    const prevClose = closes[closes.length - 2];
    const currSMA = sma50[sma50.length - 1];
    const prevSMA = sma50[sma50.length - 2];
    return isNum(prevSMA) && isNum(currSMA) &&
           prevClose > prevSMA && currClose <= currSMA;
}

function checkDeathCross(sma50, sma200) {
    if (sma50.length < 2 || sma200.length < 2) return false;
    const currShort = sma50[sma50.length - 1];
    const prevShort = sma50[sma50.length - 2];
    const currLong = sma200[sma200.length - 1];
    const prevLong = sma200[sma200.length - 2];
    return isNum(prevShort) && isNum(prevLong) && isNum(currShort) && isNum(currLong) &&
           prevShort > prevLong && currShort <= currLong;
}

function checkMACDBearishFlip(macdHist) {
    if (macdHist.length < 2) return false;
    const prev = macdHist[macdHist.length - 2];
    const curr = macdHist[macdHist.length - 1];
    return isNum(prev) && isNum(curr) && prev >= 0 && curr < 0;
}

function checkTrailingStop(highs, closes, config) {
    const pct = config.trailing_stop_pct ?? DEFAULT_CONFIG.trailing_stop_pct;
    if (highs.length < 20 || closes.length < 1) return false;
    // Use HIGH prices for the rolling peak — not close — matching Python.
    const recentHigh = Math.max(...highs.slice(-20));
    const current = closes[closes.length - 1];
    const drawdown = ((recentHigh - current) / recentHigh) * 100;
    return drawdown > pct;
}

function checkBBUpperRejection(closes, bbUpper, bbMiddle) {
    if (closes.length < 2 || bbUpper.length < 2 || bbMiddle.length < 2) return false;
    const prevClose = closes[closes.length - 2];
    const currClose = closes[closes.length - 1];
    const prevUpper = bbUpper[bbUpper.length - 2];
    const currMiddle = bbMiddle[bbMiddle.length - 1];
    if (!isNum(prevUpper) || !isNum(currMiddle)) return false;
    return prevClose > prevUpper && currClose < currMiddle;
}

function checkADXTrendFading(adx, config) {
    const weak = config.adx_weak_threshold ?? DEFAULT_CONFIG.adx_weak_threshold;
    if (adx.length < 2) return false;
    const prev = adx[adx.length - 2];
    const curr = adx[adx.length - 1];
    return isNum(prev) && isNum(curr) && prev >= weak && curr < weak;
}

function checkStochasticBearishCross(stochK, stochD, config) {
    const overbought = config.stoch_overbought ?? DEFAULT_CONFIG.stoch_overbought;
    if (stochK.length < 2 || stochD.length < 2) return false;
    const prevK = stochK[stochK.length - 2];
    const prevD = stochD[stochD.length - 2];
    const currK = stochK[stochK.length - 1];
    const currD = stochD[stochD.length - 1];
    if (!isNum(prevK) || !isNum(prevD) || !isNum(currK) || !isNum(currD)) return false;
    const crossedDown = prevK >= prevD && currK < currD;
    const wasOverbought = prevK > overbought && prevD > overbought;
    return crossedDown && wasOverbought;
}

function checkATRVolatilityStop(highs, closes, atr, config) {
    const mult = config.atr_volatility_stop_mult ?? DEFAULT_CONFIG.atr_volatility_stop_mult;
    const lookback = 20;
    if (highs.length < lookback || closes.length < 1 || atr.length < 1) return false;
    const currAtr = atr[atr.length - 1];
    if (!isNum(currAtr)) return false;
    const recentHigh = Math.max(...highs.slice(-lookback));
    const current = closes[closes.length - 1];
    const drop = recentHigh - current;
    return drop > mult * currAtr;
}

// ---------------------------------------------------------------------
// Trend regime
// ---------------------------------------------------------------------

function getTrendRegime(close, sma50, sma200) {
    if (!isNum(close) || !isNum(sma50) || !isNum(sma200)) return "transitional";
    if (close > sma200 && sma50 > sma200) return "bullish";
    if (close < sma200 && sma50 < sma200) return "bearish";
    return "transitional";
}

// ---------------------------------------------------------------------
// Weighted score aggregation
// ---------------------------------------------------------------------

/**
 * Extract all indicator arrays (sliced to current index) from the
 * ticker data blob. Returns nulls for any v2 indicators missing from
 * the JSON so we can gracefully degrade on stale pre-v2 data files.
 */
function getIndicatorSlices(data, endIdx) {
    const ind = data.indicators || {};
    const ohlcv = data.ohlcv.slice(0, endIdx + 1);
    const get = (key) => {
        const arr = ind[key];
        if (!arr) return new Array(ohlcv.length).fill(null);
        return arr.slice(0, endIdx + 1);
    };
    return {
        closes:   ohlcv.map(d => d.close),
        opens:    ohlcv.map(d => d.open),
        highs:    ohlcv.map(d => d.high),
        lows:     ohlcv.map(d => d.low),
        volumes:  ohlcv.map(d => d.volume),
        rsi:         get("rsi"),
        sma50:       get("sma50"),
        sma200:      get("sma200"),
        macdHist:    get("macd_hist"),
        emaFast:     get("ema_fast"),
        bbUpper:     get("bb_upper"),
        bbMiddle:    get("bb_middle"),
        bbLower:     get("bb_lower"),
        bbBandwidth: get("bb_bandwidth"),
        atr:         get("atr"),
        adx:         get("adx"),
        plusDI:      get("plus_di"),
        minusDI:     get("minus_di"),
        stochK:      get("stoch_k"),
        stochD:      get("stoch_d"),
        volAvg20:    get("vol_avg_20"),
    };
}

function calcOnRampScore(data, endIdx, config) {
    const s = getIndicatorSlices(data, endIdx);
    let score = 0;
    if (checkGoldenCross(s.sma50, s.sma200))                              score += WEIGHTS.golden_cross;
    if (checkRSIOversoldReversal(s.rsi, config))                          score += WEIGHTS.rsi_oversold_reversal;
    if (checkPriceReclaimsSMA50(s.closes, s.sma50))                       score += WEIGHTS.price_reclaims_sma50;
    if (checkBBSqueezeBreakout(s.closes, s.bbUpper, s.bbBandwidth, config)) score += WEIGHTS.bb_squeeze_breakout;
    if (checkADXBullishTrend(s.adx, s.plusDI, s.minusDI, config))         score += WEIGHTS.adx_bullish_trend;
    if (checkMACDBullishFlip(s.macdHist))                                 score += WEIGHTS.macd_bullish_flip;
    if (checkVolumeSpikeGreen(s.volumes, s.closes, s.opens, config))      score += WEIGHTS.volume_spike_green;
    if (checkStochasticBullishCross(s.stochK, s.stochD, config))          score += WEIGHTS.stochastic_bullish_cross;
    if (checkEMA20ReclaimWithVolume(s.closes, s.emaFast, s.volumes, s.volAvg20, config)) score += WEIGHTS.ema20_reclaim_with_volume;
    return score;
}

function calcOffRampScore(data, endIdx, config) {
    const s = getIndicatorSlices(data, endIdx);
    let score = 0;
    if (checkDeathCross(s.sma50, s.sma200))                               score += WEIGHTS.death_cross;
    if (checkRSIOverboughtReversal(s.rsi, config))                        score += WEIGHTS.rsi_overbought_reversal;
    if (checkPriceBreaksSMA50(s.closes, s.sma50))                         score += WEIGHTS.price_breaks_sma50;
    if (checkBBUpperRejection(s.closes, s.bbUpper, s.bbMiddle))           score += WEIGHTS.bb_upper_rejection;
    if (checkATRVolatilityStop(s.highs, s.closes, s.atr, config))         score += WEIGHTS.atr_volatility_stop;
    if (checkADXTrendFading(s.adx, config))                               score += WEIGHTS.adx_trend_fading;
    if (checkMACDBearishFlip(s.macdHist))                                 score += WEIGHTS.macd_bearish_flip;
    if (checkStochasticBearishCross(s.stochK, s.stochD, config))          score += WEIGHTS.stochastic_bearish_cross;
    if (checkTrailingStop(s.highs, s.closes, config))                     score += WEIGHTS.trailing_stop;
    return score;
}

function getRegimeAt(data, endIdx) {
    const ohlcv = data.ohlcv;
    if (endIdx >= ohlcv.length) return "transitional";
    const close = ohlcv[endIdx].close;
    const sma50 = data.indicators?.sma50?.[endIdx];
    const sma200 = data.indicators?.sma200?.[endIdx];
    return getTrendRegime(close, sma50, sma200);
}

// ---------------------------------------------------------------------
// Backtest runner
// ---------------------------------------------------------------------

/**
 * Run backtest on ticker data.
 * @param {Object} data - Ticker data from JSON (ohlcv, indicators)
 * @param {string} startDate - YYYY-MM-DD
 * @param {string} endDate - YYYY-MM-DD
 * @param {Object} config - Optional config overrides
 */
function runBacktest(data, startDate, endDate, config = {}) {
    const cfg = { ...DEFAULT_CONFIG, ...config };

    const startIdx = data.ohlcv.findIndex(d => d.date >= startDate);
    const endIdxRaw = data.ohlcv.findIndex(d => d.date > endDate);
    const actualEndIdx = endIdxRaw === -1 ? data.ohlcv.length : endIdxRaw;

    if (startIdx === -1 || startIdx >= actualEndIdx) {
        return {
            ticker: data.symbol,
            start_date: startDate,
            end_date: endDate,
            trades: [],
            total_return_pct: 0,
            buy_hold_return_pct: 0,
            win_rate: 0,
            num_trades: 0,
            equity_curve: [100],
            buy_hold_curve: [100],
            max_on_ramp_score: MAX_ON_RAMP_SCORE,
            max_off_ramp_score: MAX_OFF_RAMP_SCORE,
            error: "No data in date range",
        };
    }

    const warmup = Math.min(200, Math.max(50, Math.floor((actualEndIdx - startIdx) / 2)));
    const evalStartIdx = startIdx + warmup;

    if (evalStartIdx >= actualEndIdx) {
        return {
            ticker: data.symbol,
            start_date: startDate,
            end_date: endDate,
            trades: [],
            total_return_pct: 0,
            buy_hold_return_pct: 0,
            win_rate: 0,
            num_trades: 0,
            equity_curve: [100],
            buy_hold_curve: [100],
            max_on_ramp_score: MAX_ON_RAMP_SCORE,
            max_off_ramp_score: MAX_OFF_RAMP_SCORE,
            error: "Not enough data for warmup period",
        };
    }

    const trades = [];
    const equity = [100.0];
    const buyHold = [100.0];
    let inPosition = false;
    let entryPrice = 0;
    let entryDate = "";

    const buyHoldStartPrice = data.ohlcv[evalStartIdx].close;

    for (let i = evalStartIdx; i < actualEndIdx; i++) {
        const onScore = calcOnRampScore(data, i, cfg);
        const offScore = calcOffRampScore(data, i, cfg);
        const regime = getRegimeAt(data, i);

        const currentPrice = data.ohlcv[i].close;
        const currentDate = data.ohlcv[i].date;
        const prevPrice = i > 0 ? data.ohlcv[i - 1].close : currentPrice;

        buyHold.push(100.0 * currentPrice / buyHoldStartPrice);

        if (!inPosition) {
            // Suppress entries in bearish regime — the single biggest
            // improvement over the v1 engine.
            if (onScore >= ENTRY_THRESHOLD && regime !== "bearish") {
                inPosition = true;
                entryPrice = currentPrice;
                entryDate = currentDate;
            }
            equity.push(equity[equity.length - 1]);
        } else {
            const dailyReturn = currentPrice / prevPrice;
            equity.push(equity[equity.length - 1] * dailyReturn);

            if (offScore >= EXIT_THRESHOLD) {
                const pctChange = ((currentPrice - entryPrice) / entryPrice) * 100;
                trades.push({
                    entry_date: entryDate,
                    entry_price: Math.round(entryPrice * 100) / 100,
                    exit_date: currentDate,
                    exit_price: Math.round(currentPrice * 100) / 100,
                    pct_change: Math.round(pctChange * 100) / 100,
                });
                inPosition = false;
            }
        }
    }

    // Close any open position at end
    if (inPosition) {
        const lastPrice = data.ohlcv[actualEndIdx - 1].close;
        const lastDate = data.ohlcv[actualEndIdx - 1].date;
        const pctChange = ((lastPrice - entryPrice) / entryPrice) * 100;
        trades.push({
            entry_date: entryDate,
            entry_price: Math.round(entryPrice * 100) / 100,
            exit_date: lastDate + " (open)",
            exit_price: Math.round(lastPrice * 100) / 100,
            pct_change: Math.round(pctChange * 100) / 100,
        });
    }

    const closedTrades = trades.filter(t => t.pct_change !== null);
    const wins = closedTrades.filter(t => t.pct_change > 0);
    const totalReturn = equity[equity.length - 1] - 100.0;
    const winRate = closedTrades.length > 0 ? (wins.length / closedTrades.length) * 100 : 0;
    const buyHoldReturn = buyHold[buyHold.length - 1] - 100.0;

    return {
        ticker: data.symbol,
        start_date: startDate,
        end_date: endDate,
        trades: trades,
        total_return_pct: Math.round(totalReturn * 100) / 100,
        buy_hold_return_pct: Math.round(buyHoldReturn * 100) / 100,
        win_rate: Math.round(winRate * 10) / 10,
        num_trades: trades.length,
        equity_curve: equity.map(v => Math.round(v * 100) / 100),
        buy_hold_curve: buyHold.map(v => Math.round(v * 100) / 100),
        max_on_ramp_score: MAX_ON_RAMP_SCORE,
        max_off_ramp_score: MAX_OFF_RAMP_SCORE,
    };
}

// ---------------------------------------------------------------------
// Results rendering
// ---------------------------------------------------------------------

function renderBacktestResults(result, containerId) {
    const container = document.getElementById(containerId);
    if (!container) return;

    if (result.error) {
        container.innerHTML = `<div class="error">${result.error}</div>`;
        return;
    }

    const diff = result.total_return_pct - result.buy_hold_return_pct;
    const diffColor = diff >= 0 ? 'var(--green)' : 'var(--red)';
    const diffBg = diff >= 0 ? '#e8f5e9' : '#ffebee';

    let html = `
        <div class="card">
            <h3>${result.ticker} &mdash; ${result.start_date} to ${result.end_date}</h3>
            <h4 style="margin:0.5rem 0 1rem;color:#666;font-weight:500">Strategy Comparison</h4>

            <div style="display:grid;grid-template-columns:1fr 1fr;gap:1.5rem;margin-bottom:1rem">
                <div style="background:#f8fafc;border-radius:8px;padding:1rem;border-left:4px solid #2196F3">
                    <div style="font-size:0.9rem;font-weight:600;color:#2196F3;margin-bottom:0.75rem">Signal Engine</div>
                    <div style="display:flex;flex-direction:column;gap:0.5rem">
                        <div style="display:flex;justify-content:space-between;align-items:baseline">
                            <span style="font-size:0.8rem;color:#666">Return</span>
                            <span style="font-size:1.4rem;font-weight:700;color:${result.total_return_pct >= 0 ? 'var(--green)' : 'var(--red)'}">
                                ${result.total_return_pct >= 0 ? '+' : ''}${result.total_return_pct.toFixed(2)}%
                            </span>
                        </div>
                        <div style="display:flex;justify-content:space-between;align-items:baseline">
                            <span style="font-size:0.8rem;color:#666">Trades</span>
                            <span style="font-size:1rem;font-weight:600">${result.num_trades}</span>
                        </div>
                        <div style="display:flex;justify-content:space-between;align-items:baseline">
                            <span style="font-size:0.8rem;color:#666">Win Rate</span>
                            <span style="font-size:1rem;font-weight:600">${result.win_rate}%</span>
                        </div>
                    </div>
                </div>

                <div style="background:#f8fafc;border-radius:8px;padding:1rem;border-left:4px solid #9E9E9E">
                    <div style="font-size:0.9rem;font-weight:600;color:#666;margin-bottom:0.75rem">Buy & Hold</div>
                    <div style="display:flex;flex-direction:column;gap:0.5rem">
                        <div style="display:flex;justify-content:space-between;align-items:baseline">
                            <span style="font-size:0.8rem;color:#666">Return</span>
                            <span style="font-size:1.4rem;font-weight:700;color:${result.buy_hold_return_pct >= 0 ? 'var(--green)' : 'var(--red)'}">
                                ${result.buy_hold_return_pct >= 0 ? '+' : ''}${result.buy_hold_return_pct.toFixed(2)}%
                            </span>
                        </div>
                        <div style="display:flex;justify-content:space-between;align-items:baseline">
                            <span style="font-size:0.8rem;color:#666">Trades</span>
                            <span style="font-size:1rem;font-weight:600">1</span>
                        </div>
                        <div style="display:flex;justify-content:space-between;align-items:baseline">
                            <span style="font-size:0.8rem;color:#666">Win Rate</span>
                            <span style="font-size:1rem;font-weight:600">${result.buy_hold_return_pct >= 0 ? '100' : '0'}%</span>
                        </div>
                    </div>
                </div>
            </div>

            <div style="text-align:center;padding:0.5rem;background:${diffBg};border-radius:6px">
                <span style="font-size:0.85rem;color:${diffColor};font-weight:600">
                    Signal Engine ${diff >= 0 ? '+' : ''}${diff.toFixed(2)}% vs Buy & Hold
                </span>
            </div>
        </div>

        <div class="card chart-container">
            <div id="equity-chart"></div>
        </div>
    `;

    // Add trades table if there are trades
    if (result.trades.length > 0) {
        html += `
            <div class="card">
                <h3>Trade History</h3>
                <table>
                    <thead>
                        <tr>
                            <th>Entry Date</th>
                            <th>Entry $</th>
                            <th>Exit Date</th>
                            <th>Exit $</th>
                            <th>Change</th>
                        </tr>
                    </thead>
                    <tbody>
                        ${result.trades.map(t => `
                            <tr>
                                <td>${t.entry_date}</td>
                                <td>$${t.entry_price.toFixed(2)}</td>
                                <td>${t.exit_date}</td>
                                <td>$${t.exit_price.toFixed(2)}</td>
                                <td style="color:${t.pct_change >= 0 ? 'var(--green)' : 'var(--red)'}">
                                    ${t.pct_change >= 0 ? '+' : ''}${t.pct_change.toFixed(2)}%
                                </td>
                            </tr>
                        `).join('')}
                    </tbody>
                </table>
            </div>
        `;
    }

    container.innerHTML = html;

    // Render equity curve chart
    Charts.renderEquityCurve('equity-chart', result);
}

// Export for global access
window.Backtest = {
    run: runBacktest,
    render: renderBacktestResults,
    MAX_ON_RAMP_SCORE,
    MAX_OFF_RAMP_SCORE,
};
