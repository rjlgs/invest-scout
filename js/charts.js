/**
 * Chart rendering using Plotly.js
 */

/**
 * Load ticker detail data
 * @param {string} symbol - Ticker symbol
 * @returns {Promise<Object>} Ticker data including OHLCV and indicators
 */
async function loadTickerData(symbol) {
    try {
        const response = await fetch(`data/tickers/${symbol}.json`);
        if (!response.ok) throw new Error('Ticker not found');
        return await response.json();
    } catch (error) {
        console.error(`Error loading ${symbol}:`, error);
        return null;
    }
}

/**
 * Render candlestick chart with moving averages
 * @param {string} containerId - DOM element ID
 * @param {Object} data - Ticker data object
 */
function renderCandlestickChart(containerId, data) {
    const { ohlcv, indicators, symbol } = data;

    const dates = ohlcv.map(d => d.date);

    const traces = [
        // Candlestick
        {
            type: 'candlestick',
            x: dates,
            open: ohlcv.map(d => d.open),
            high: ohlcv.map(d => d.high),
            low: ohlcv.map(d => d.low),
            close: ohlcv.map(d => d.close),
            name: 'Price',
            increasing: { line: { color: '#4CAF50' } },
            decreasing: { line: { color: '#F44336' } },
        },
        // SMA 50
        {
            type: 'scatter',
            mode: 'lines',
            x: dates,
            y: indicators.sma50,
            name: 'SMA 50',
            line: { color: '#2196F3', width: 1.5 },
        },
        // SMA 200
        {
            type: 'scatter',
            mode: 'lines',
            x: dates,
            y: indicators.sma200,
            name: 'SMA 200',
            line: { color: '#FF9800', width: 1.5 },
        },
    ];

    const layout = {
        title: `${symbol} — Price & Moving Averages`,
        yaxis: { title: 'Price ($)' },
        xaxis: { rangeslider: { visible: false } },
        template: 'plotly_white',
        height: 450,
        margin: { l: 50, r: 30, t: 50, b: 40 },
        legend: { orientation: 'h', y: 1.1 },
    };

    Plotly.newPlot(containerId, traces, layout, { responsive: true });
}

/**
 * Render RSI chart
 * @param {string} containerId - DOM element ID
 * @param {Object} data - Ticker data object
 */
function renderRSIChart(containerId, data) {
    const { ohlcv, indicators, symbol } = data;
    const dates = ohlcv.map(d => d.date);

    const traces = [
        // RSI line
        {
            type: 'scatter',
            mode: 'lines',
            x: dates,
            y: indicators.rsi,
            name: 'RSI',
            line: { color: '#9C27B0', width: 2 },
        },
    ];

    // Add overbought/oversold zones
    const shapes = [
        // Overbought zone (70-100)
        {
            type: 'rect',
            xref: 'paper',
            yref: 'y',
            x0: 0,
            x1: 1,
            y0: 70,
            y1: 100,
            fillcolor: 'rgba(244, 67, 54, 0.1)',
            line: { width: 0 },
        },
        // Oversold zone (0-30)
        {
            type: 'rect',
            xref: 'paper',
            yref: 'y',
            x0: 0,
            x1: 1,
            y0: 0,
            y1: 30,
            fillcolor: 'rgba(76, 175, 80, 0.1)',
            line: { width: 0 },
        },
        // 70 line
        {
            type: 'line',
            xref: 'paper',
            yref: 'y',
            x0: 0,
            x1: 1,
            y0: 70,
            y1: 70,
            line: { color: '#F44336', width: 1, dash: 'dash' },
        },
        // 30 line
        {
            type: 'line',
            xref: 'paper',
            yref: 'y',
            x0: 0,
            x1: 1,
            y0: 30,
            y1: 30,
            line: { color: '#4CAF50', width: 1, dash: 'dash' },
        },
    ];

    const layout = {
        title: 'RSI (14)',
        yaxis: { title: 'RSI', range: [0, 100] },
        template: 'plotly_white',
        height: 250,
        margin: { l: 50, r: 30, t: 40, b: 40 },
        shapes: shapes,
    };

    Plotly.newPlot(containerId, traces, layout, { responsive: true });
}

/**
 * Render MACD chart
 * @param {string} containerId - DOM element ID
 * @param {Object} data - Ticker data object
 */
function renderMACDChart(containerId, data) {
    const { ohlcv, indicators, symbol } = data;
    const dates = ohlcv.map(d => d.date);

    // Color histogram bars based on value
    const histColors = indicators.macd_hist.map(v =>
        v >= 0 ? 'rgba(76, 175, 80, 0.7)' : 'rgba(244, 67, 54, 0.7)'
    );

    const traces = [
        // Histogram
        {
            type: 'bar',
            x: dates,
            y: indicators.macd_hist,
            name: 'Histogram',
            marker: { color: histColors },
        },
        // MACD line
        {
            type: 'scatter',
            mode: 'lines',
            x: dates,
            y: indicators.macd,
            name: 'MACD',
            line: { color: '#2196F3', width: 1.5 },
        },
        // Signal line
        {
            type: 'scatter',
            mode: 'lines',
            x: dates,
            y: indicators.macd_signal,
            name: 'Signal',
            line: { color: '#FF9800', width: 1.5 },
        },
    ];

    const layout = {
        title: 'MACD (12, 26, 9)',
        yaxis: { title: 'MACD' },
        template: 'plotly_white',
        height: 250,
        margin: { l: 50, r: 30, t: 40, b: 40 },
        legend: { orientation: 'h', y: 1.15 },
        barmode: 'relative',
    };

    Plotly.newPlot(containerId, traces, layout, { responsive: true });
}

/**
 * Render equity curve chart for backtest results
 * @param {string} containerId - DOM element ID
 * @param {Object} result - Backtest result object
 */
function renderEquityCurve(containerId, result) {
    const traces = [];

    // Buy & hold curve (if available)
    if (result.buy_hold_curve && result.buy_hold_curve.length > 0) {
        traces.push({
            type: 'scatter',
            mode: 'lines',
            y: result.buy_hold_curve,
            name: 'Buy & Hold',
            line: { color: '#9E9E9E', width: 2, dash: 'dot' },
        });
    }

    // Strategy equity curve
    traces.push({
        type: 'scatter',
        mode: 'lines',
        y: result.equity_curve,
        name: 'Strategy',
        line: { color: '#2196F3', width: 2 },
    });

    // Starting value reference line
    const shapes = [{
        type: 'line',
        xref: 'paper',
        yref: 'y',
        x0: 0,
        x1: 1,
        y0: 100,
        y1: 100,
        line: { color: 'grey', width: 1, dash: 'dash' },
    }];

    const strategyReturn = result.total_return_pct || 0;
    const buyHoldReturn = result.buy_hold_return_pct || 0;

    const layout = {
        title: `${result.ticker} Backtest: Strategy ${strategyReturn >= 0 ? '+' : ''}${strategyReturn.toFixed(1)}% vs Buy & Hold ${buyHoldReturn >= 0 ? '+' : ''}${buyHoldReturn.toFixed(1)}%`,
        yaxis: { title: 'Portfolio Value ($)' },
        xaxis: { title: 'Trading Days' },
        template: 'plotly_white',
        height: 400,
        margin: { l: 50, r: 30, t: 50, b: 40 },
        legend: { yanchor: 'top', y: 0.99, xanchor: 'left', x: 0.01 },
        shapes: shapes,
    };

    Plotly.newPlot(containerId, traces, layout, { responsive: true });
}

// Export for global access
window.Charts = {
    loadTickerData,
    renderCandlestickChart,
    renderRSIChart,
    renderMACDChart,
    renderEquityCurve,
};
