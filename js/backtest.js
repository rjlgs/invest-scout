/**
 * Client-side backtesting engine
 * Ported from src/backtest.py
 */

// Signal thresholds
const ENTRY_THRESHOLD = 2;
const EXIT_THRESHOLD = 2;

// Default config values
const DEFAULT_CONFIG = {
    rsi_oversold: 30,
    rsi_overbought: 70,
    trailing_stop_pct: 8.0,
    volume_spike_multiplier: 2.0,
};

/**
 * Check RSI Oversold Reversal (ON-RAMP)
 * RSI crosses below oversold threshold then recovers above it
 */
function checkRSIOversoldReversal(rsi, config) {
    const threshold = config.rsi_oversold || DEFAULT_CONFIG.rsi_oversold;
    if (rsi.length < 3) return false;

    const curr = rsi[rsi.length - 1];
    const prev = rsi[rsi.length - 2];
    const prev2 = rsi[rsi.length - 3];

    // Was below threshold, now above
    return prev2 !== null && prev !== null && curr !== null &&
           prev < threshold && curr >= threshold;
}

/**
 * Check Price Reclaims SMA50 (ON-RAMP)
 * Price crosses above SMA50 after being below
 */
function checkPriceReclaimsSMA50(closes, sma50) {
    if (closes.length < 2 || sma50.length < 2) return false;

    const currClose = closes[closes.length - 1];
    const prevClose = closes[closes.length - 2];
    const currSMA = sma50[sma50.length - 1];
    const prevSMA = sma50[sma50.length - 2];

    return prevSMA !== null && currSMA !== null &&
           prevClose < prevSMA && currClose >= currSMA;
}

/**
 * Check Golden Cross (ON-RAMP)
 * SMA50 crosses above SMA200
 */
function checkGoldenCross(sma50, sma200) {
    if (sma50.length < 2 || sma200.length < 2) return false;

    const currShort = sma50[sma50.length - 1];
    const prevShort = sma50[sma50.length - 2];
    const currLong = sma200[sma200.length - 1];
    const prevLong = sma200[sma200.length - 2];

    return prevShort !== null && prevLong !== null &&
           currShort !== null && currLong !== null &&
           prevShort <= prevLong && currShort > currLong;
}

/**
 * Check MACD Bullish Flip (ON-RAMP)
 * MACD histogram flips from negative to positive
 */
function checkMACDBullishFlip(macdHist) {
    if (macdHist.length < 2) return false;

    const curr = macdHist[macdHist.length - 1];
    const prev = macdHist[macdHist.length - 2];

    return prev !== null && curr !== null &&
           prev < 0 && curr >= 0;
}

/**
 * Check Volume Spike Green (ON-RAMP)
 * Volume > multiplier * 20-day average on up day
 */
function checkVolumeSpikeGreen(volumes, closes, config) {
    const multiplier = config.volume_spike_multiplier || DEFAULT_CONFIG.volume_spike_multiplier;
    if (volumes.length < 21 || closes.length < 2) return false;

    const currVol = volumes[volumes.length - 1];
    const avgVol = volumes.slice(-21, -1).reduce((a, b) => a + b, 0) / 20;
    const currClose = closes[closes.length - 1];
    const prevClose = closes[closes.length - 2];

    return currVol > avgVol * multiplier && currClose > prevClose;
}

/**
 * Check RSI Overbought Reversal (OFF-RAMP)
 * RSI crosses above overbought threshold then drops below
 */
function checkRSIOverboughtReversal(rsi, config) {
    const threshold = config.rsi_overbought || DEFAULT_CONFIG.rsi_overbought;
    if (rsi.length < 3) return false;

    const curr = rsi[rsi.length - 1];
    const prev = rsi[rsi.length - 2];

    return prev !== null && curr !== null &&
           prev > threshold && curr <= threshold;
}

/**
 * Check Price Breaks SMA50 (OFF-RAMP)
 * Price crosses below SMA50 after being above
 */
function checkPriceBreaksSMA50(closes, sma50) {
    if (closes.length < 2 || sma50.length < 2) return false;

    const currClose = closes[closes.length - 1];
    const prevClose = closes[closes.length - 2];
    const currSMA = sma50[sma50.length - 1];
    const prevSMA = sma50[sma50.length - 2];

    return prevSMA !== null && currSMA !== null &&
           prevClose > prevSMA && currClose <= currSMA;
}

/**
 * Check Death Cross (OFF-RAMP)
 * SMA50 crosses below SMA200
 */
function checkDeathCross(sma50, sma200) {
    if (sma50.length < 2 || sma200.length < 2) return false;

    const currShort = sma50[sma50.length - 1];
    const prevShort = sma50[sma50.length - 2];
    const currLong = sma200[sma200.length - 1];
    const prevLong = sma200[sma200.length - 2];

    return prevShort !== null && prevLong !== null &&
           currShort !== null && currLong !== null &&
           prevShort >= prevLong && currShort < currLong;
}

/**
 * Check MACD Bearish Flip (OFF-RAMP)
 * MACD histogram flips from positive to negative
 */
function checkMACDBearishFlip(macdHist) {
    if (macdHist.length < 2) return false;

    const curr = macdHist[macdHist.length - 1];
    const prev = macdHist[macdHist.length - 2];

    return prev !== null && curr !== null &&
           prev > 0 && curr <= 0;
}

/**
 * Check Trailing Stop (OFF-RAMP)
 * Price dropped more than X% from recent high
 */
function checkTrailingStop(closes, config) {
    const threshold = config.trailing_stop_pct || DEFAULT_CONFIG.trailing_stop_pct;
    if (closes.length < 20) return false;

    const recentHigh = Math.max(...closes.slice(-20));
    const currPrice = closes[closes.length - 1];
    const dropPct = ((recentHigh - currPrice) / recentHigh) * 100;

    return dropPct >= threshold;
}

/**
 * Calculate on-ramp score at a given index
 */
function calcOnRampScore(data, endIdx, config) {
    const closes = data.ohlcv.slice(0, endIdx + 1).map(d => d.close);
    const volumes = data.ohlcv.slice(0, endIdx + 1).map(d => d.volume);
    const rsi = data.indicators.rsi.slice(0, endIdx + 1);
    const sma50 = data.indicators.sma50.slice(0, endIdx + 1);
    const sma200 = data.indicators.sma200.slice(0, endIdx + 1);
    const macdHist = data.indicators.macd_hist.slice(0, endIdx + 1);

    let score = 0;
    if (checkRSIOversoldReversal(rsi, config)) score++;
    if (checkPriceReclaimsSMA50(closes, sma50)) score++;
    if (checkGoldenCross(sma50, sma200)) score++;
    if (checkMACDBullishFlip(macdHist)) score++;
    if (checkVolumeSpikeGreen(volumes, closes, config)) score++;

    return score;
}

/**
 * Calculate off-ramp score at a given index
 */
function calcOffRampScore(data, endIdx, config) {
    const closes = data.ohlcv.slice(0, endIdx + 1).map(d => d.close);
    const rsi = data.indicators.rsi.slice(0, endIdx + 1);
    const sma50 = data.indicators.sma50.slice(0, endIdx + 1);
    const sma200 = data.indicators.sma200.slice(0, endIdx + 1);
    const macdHist = data.indicators.macd_hist.slice(0, endIdx + 1);

    let score = 0;
    if (checkRSIOverboughtReversal(rsi, config)) score++;
    if (checkPriceBreaksSMA50(closes, sma50)) score++;
    if (checkDeathCross(sma50, sma200)) score++;
    if (checkMACDBearishFlip(macdHist)) score++;
    if (checkTrailingStop(closes, config)) score++;

    return score;
}

/**
 * Run backtest on ticker data
 * @param {Object} data - Ticker data from JSON (ohlcv, indicators)
 * @param {string} startDate - YYYY-MM-DD start date
 * @param {string} endDate - YYYY-MM-DD end date
 * @param {Object} config - Optional config overrides
 * @returns {Object} Backtest results
 */
function runBacktest(data, startDate, endDate, config = {}) {
    const cfg = { ...DEFAULT_CONFIG, ...config };

    // Filter data to date range
    const startIdx = data.ohlcv.findIndex(d => d.date >= startDate);
    const endIdx = data.ohlcv.findIndex(d => d.date > endDate);
    const actualEndIdx = endIdx === -1 ? data.ohlcv.length : endIdx;

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
            error: 'No data in date range',
        };
    }

    // Warmup period - need at least 200 data points or half of data
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
            error: 'Not enough data for warmup period',
        };
    }

    const trades = [];
    const equity = [100.0];
    const buyHold = [100.0];
    let inPosition = false;
    let entryPrice = 0;
    let entryDate = '';

    const buyHoldStartPrice = data.ohlcv[evalStartIdx].close;

    for (let i = evalStartIdx; i < actualEndIdx; i++) {
        const onScore = calcOnRampScore(data, i, cfg);
        const offScore = calcOffRampScore(data, i, cfg);

        const currentPrice = data.ohlcv[i].close;
        const currentDate = data.ohlcv[i].date;
        const prevPrice = i > 0 ? data.ohlcv[i - 1].close : currentPrice;

        // Update buy-and-hold curve
        buyHold.push(100.0 * currentPrice / buyHoldStartPrice);

        if (!inPosition) {
            // Look for entry
            if (onScore >= ENTRY_THRESHOLD) {
                inPosition = true;
                entryPrice = currentPrice;
                entryDate = currentDate;
            }
            // Equity stays flat when not invested
            equity.push(equity[equity.length - 1]);
        } else {
            // Track equity while in position
            const dailyReturn = currentPrice / prevPrice;
            equity.push(equity[equity.length - 1] * dailyReturn);

            // Look for exit
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
            exit_date: lastDate + ' (open)',
            exit_price: Math.round(lastPrice * 100) / 100,
            pct_change: Math.round(pctChange * 100) / 100,
        });
    }

    // Calculate summary stats
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
    };
}

/**
 * Render backtest results to the page
 */
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
};
