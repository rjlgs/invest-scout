# Invest-Scout

Automated investment signal analysis for stocks and ETFs. Identifies potential entry ("on-ramp") and exit ("off-ramp") opportunities for swing trading on days-to-weeks timeframes.

> **DISCLAIMER:** This tool is for **informational and educational purposes only**. It is NOT financial advice. Past performance of signals does not guarantee future results. Always do your own research and consult a qualified financial advisor before making investment decisions. The authors assume no liability for financial losses incurred using this software. **This tool never executes trades** — it only generates signals.

---

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. (Optional) Copy and configure environment variables
cp .env.example .env

# 3. Start the web dashboard
python main.py

# Dashboard runs at http://localhost:5000
```

## CLI Usage

```bash
# Start the web dashboard (default)
python main.py

# Run a one-shot scan and print results
python main.py --scan

# Run a backtest
python main.py --backtest AAPL 2024-01-01 2024-12-31

# Use a custom port
python main.py --port 8080
```

## How It Works

Invest-Scout watches a list of stock/ETF tickers, fetches their daily price data, computes technical indicators, and checks for 10 specific signal conditions (5 entry, 5 exit). Each ticker gets a score from 0 to 5 for both on-ramp and off-ramp signals.

---

## Signals Explained

### On-Ramp Signals (Potential Entry)

These signals suggest it might be a good time to **buy or add to a position**:

#### 1. RSI Oversold Reversal
**What it does:** Flags when the RSI (Relative Strength Index) dips below 30 then climbs back above it.

**In plain English:** The stock has been beaten down heavily — most people who wanted to sell have already sold. When RSI climbs back above 30, it means buyers are starting to step back in. Think of it like a rubber band that's been stretched too far — it tends to snap back.

#### 2. Price Reclaims 50-Day Moving Average
**What it does:** Flags when the stock's price closes above its 50-day average after being below it.

**In plain English:** The 50-day moving average is like a "trend line" that smooths out daily noise. When a stock falls below it, it's in a short-term downtrend. Climbing back above it is like clearing a hurdle — it signals the downtrend may be over and buyers are back in control.

#### 3. Golden Cross
**What it does:** Flags when the 50-day moving average crosses above the 200-day moving average.

**In plain English:** This is one of the most famous signals on Wall Street. The 50-day average represents the recent trend, and the 200-day represents the long-term trend. When the short-term trend overtakes the long-term one, it suggests a major shift upward. Think of it as the stock switching from "going downhill" to "going uphill."

#### 4. MACD Bullish Flip
**What it does:** Flags when the MACD histogram turns from negative to positive.

**In plain English:** MACD measures momentum — how fast the price is moving. When the histogram flips positive, it means upward momentum is accelerating. It's like a car that was slowing down and has now started speeding up again.

#### 5. Volume Spike on Green Candle
**What it does:** Flags when trading volume is more than 2x the 20-day average AND the stock closed higher than it opened.

**In plain English:** Normally a certain amount of shares trade each day. When volume suddenly doubles or triples on a day where the price went UP, it usually means big institutional investors (mutual funds, hedge funds) are buying. When the "smart money" buys aggressively, it often signals confidence in the stock's future.

---

### Off-Ramp Signals (Potential Exit)

These signals suggest it might be a good time to **sell or reduce a position**:

#### 6. RSI Overbought Reversal
**What it does:** Flags when RSI rises above 70 then drops back below it.

**In plain English:** RSI above 70 means the stock has been rising very fast — possibly too fast. When it drops back below 70, the buying frenzy is cooling off. It's like a party that's winding down — early leavers often avoid the messy cleanup. This is a good time to consider taking profits.

#### 7. Price Breaks Below 50-Day Moving Average
**What it does:** Flags when the stock closes below its 50-day average after being above it.

**In plain English:** The opposite of Signal #2. If the stock was riding above its trend line and suddenly breaks below it, the medium-term trend is weakening. It's like a stock dropping below a "floor" that was holding it up.

#### 8. Death Cross
**What it does:** Flags when the 50-day moving average crosses below the 200-day moving average.

**In plain English:** The opposite of the golden cross. When the recent trend drops below the long-term trend, it's a bearish signal suggesting extended weakness ahead. Markets often see significant declines after a death cross.

#### 9. MACD Bearish Flip
**What it does:** Flags when the MACD histogram turns from positive to negative.

**In plain English:** Momentum has shifted from accelerating upward to decelerating. The rally is losing steam. Like a ball thrown in the air that has started coming back down — gravity (selling pressure) is winning.

#### 10. Trailing Stop Breach
**What it does:** Flags when the current price is more than 8% below its recent high (last 20 trading days).

**In plain English:** This is a safety net. If you bought a stock and it rose nicely, but now it's dropped 8% from its peak, the uptrend may be broken. The trailing stop enforces discipline — it says "I'll accept small dips, but if it drops this much, I'm out." The 8% threshold is configurable.

---

## Scoring

Each ticker is scored on two dimensions:

- **On-Ramp Score (0-5):** How many of the 5 entry signals are currently firing
- **Off-Ramp Score (0-5):** How many of the 5 exit signals are currently firing

The overall signal type is determined by which score is higher:
- **On-Ramp:** More entry signals than exit signals
- **Off-Ramp:** More exit signals than entry signals
- **Neutral:** Equal scores (or both zero)

A score of 3+ is considered significant.

---

## Configuration

### Environment Variables (`.env`)

Copy `.env.example` to `.env` and adjust:

| Variable | Default | Description |
|----------|---------|-------------|
| `CACHE_TTL_MINUTES` | 60 | How long to cache market data |
| `RSI_PERIOD` | 14 | RSI calculation period |
| `RSI_OVERSOLD` | 30 | RSI oversold threshold |
| `RSI_OVERBOUGHT` | 70 | RSI overbought threshold |
| `SMA_SHORT` | 50 | Short moving average period |
| `SMA_LONG` | 200 | Long moving average period |
| `MACD_FAST` | 12 | MACD fast EMA period |
| `MACD_SLOW` | 26 | MACD slow EMA period |
| `MACD_SIGNAL` | 9 | MACD signal line period |
| `VOLUME_SPIKE_MULTIPLIER` | 2.0 | Volume spike threshold multiplier |
| `TRAILING_STOP_PCT` | 8.0 | Trailing stop percentage |
| `FLASK_PORT` | 5000 | Web dashboard port |
| `SCAN_INTERVAL_MINUTES` | 5 | Background scan interval |
| `EMAIL_ENABLED` | false | Enable email alerts |

### Watchlist (`config/watchlist.yaml`)

```yaml
tickers:
  - symbol: AAPL
  - symbol: TSLA
    trailing_stop_pct: 12.0    # Override: wider stop for volatile stocks
    rsi_oversold: 25           # Override: TSLA tends to dip harder
  - symbol: SPY
```

---

## Web Dashboard

The dashboard runs locally and provides:

- **Overview page** (`/`): Sortable table of all tickers with color-coded signal scores. Auto-refreshes every 5 minutes.
- **Ticker detail** (`/ticker/AAPL`): Candlestick chart with SMA overlays, RSI and MACD subcharts, active signal breakdown, and 30-day signal history.
- **Backtest** (`/backtest`): Run historical simulations with an interactive form. Shows trade log, summary stats, and equity curve.

---

## Project Structure

```
invest-scout/
├── main.py              # CLI entry point + Flask web server
├── config/
│   └── watchlist.yaml   # Ticker watchlist
├── src/
│   ├── data.py          # Data fetching & caching (yfinance)
│   ├── signals.py       # Indicator calculations & signal detection
│   ├── scanner.py       # Watchlist scanner loop
│   ├── alerts.py        # Email alerts (SMTP)
│   └── backtest.py      # Backtesting engine
├── templates/           # Jinja2 HTML templates (Flask + HTMX)
├── tests/
│   └── test_signals.py  # Signal logic unit tests
├── .env.example         # Configuration template
├── requirements.txt     # Python dependencies
└── README.md
```

## Running Tests

```bash
pytest tests/ -v
```

---

## Tech Stack

- **Python 3.11+** — core language
- **pandas / numpy** — data manipulation and indicator calculations
- **yfinance** — market data from Yahoo Finance
- **Flask** — lightweight web server
- **HTMX** — dashboard interactivity without a JS framework
- **Plotly** — interactive charts (candlestick, RSI, MACD, equity curves)
- **PyYAML** — watchlist configuration
- **python-dotenv** — environment variable management
