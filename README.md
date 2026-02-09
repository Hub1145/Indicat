# Pro-Trader Ultimate TA-as-a-Service API

A high-performance, enterprise-grade Technical Analysis API built with Python, FastAPI, and industry-standard financial libraries. This service offloads the complex mathematical "heavy lifting" of technical analysis, providing over 250+ indicators, candlestick patterns, and institutional trading concepts.

## 🚀 Key Features

- **Monetization Ready**: Integrated middleware for RapidAPI proxy verification and tiered access control.
- **Dual Modes of Analysis**:
  - **Market Fetching**: Automatic data retrieval for Crypto (Binance/Kraken via CCXT), Stocks, and Forex (yfinance).
  - **Data Upload**: Accept direct OHLCV JSON uploads with automatic trimming (Max 2000 candles).
- **Massive Indicator Set**:
  - **250+ Indicators**: Full integration of TA-Lib and the 'ta' library.
  - **Clean API**: All indicator names are normalized (e.g., `RSI`, `EMA200`, `Bollinger_High`) and library-agnostic.
  - **Discovery**: `GET /indicators` provides full descriptive names and shorthand codes for all features.
- **Institutional & Advanced Logic**:
  - **SMC/ICT Strategy**: Built-in detection for Fair Value Gaps (FVG), Order Blocks (OB), and Market Structure Shifts (MSS).
  - **Divergence Engine**: Automated detection of Bullish and Bearish divergences (RSI/MACD).
  - **Volume Profile**: Point of Control (POC), Value Area High (VAH), and Value Area Low (VAL).
  - **Market Dynamics**: Automatic session detection (Tokyo/London/NY) and Support/Resistance level clustering.
- **Pro Trading Tools**:
  - **Multi-Timeframe Analysis (MTF)**: Analyze assets across multiple intervals in one request.
  - **Correlation Matrix**: Deep cross-asset correlation analysis.
  - **Options Analytics**: Black-Scholes Greeks Calculator (Delta, Gamma, Theta, Vega).
  - **Heatmap Data**: Aggregated metrics for multiple symbols simultaneously.
- **Performance & Stability**:
  - **Vectorized Signals**: Full historical Buy/Sell marking supported via `include_history` flag.
  - **Caching**: 60-second disk-based caching (`diskcache`) for rate-limit protection.
  - **Robust JSON**: Recursive NaN/Inf cleaning ensures 100% stability.

## 🛠 Tech Stack

- **Framework**: [FastAPI](https://fastapi.tiangolo.com/)
- **Core Math**: [TA-Lib](https://mrjbq7.github.io/ta-lib/) (C-wrapper)
- **Trend/Volatility**: [ta](https://github.com/bukosabino/ta)
- **Market Data**: [CCXT](https://github.com/ccxt/ccxt), [yfinance](https://github.com/ranaroussi/yfinance)
- **Infrastructure**: Docker, Diskcache, Pydantic, Gunicorn/Uvicorn

## ⚙️ Installation

### 1. Prerequisites (TA-Lib C-Library)
**Ubuntu/Linux:**
```bash
wget http://prdownloads.sourceforge.net/ta-lib/ta-lib-0.4.0-src.tar.gz
tar -xzf ta-lib-0.4.0-src.tar.gz
cd ta-lib/ && ./configure --prefix=/usr && make && sudo make install
```

### 2. Deployment (Docker - Recommended)
```bash
docker-compose up --build -d
```

## 💰 Tiered Pricing Strategy

| Tier | Price | Features |
| :--- | :--- | :--- |
| **Basic** | Free | 1d timeframe, top 10 indicators, 50 req/day |
| **Pro** | $25/mo | All timeframes, all 250+ indicators, 5,000 req/mo |
| **Ultra** | $80/mo | Everything + SMC/ICT + MTF + Confluence Score |
| **Mega** | $150/mo | Unlimited + Priority Support + Webhook support |

## 📖 API Documentation

Visit `http://127.0.0.1:8000/docs` for full interactive documentation.

### Core Endpoints
- `GET /indicators`: Categorized list of all supported logic.
- `POST /analyze/market`: Live exchange analysis.
- `POST /analyze/upload`: User-data analysis.
- `POST /analyze/mtf`: Multi-timeframe analysis.
- `POST /confluence-score`: Unified Buy/Sell sentiment (-100 to +100).
- `POST /analyze/correlation`: Cross-asset coefficients.
- `POST /options/greeks`: Options pricing analytics.

## 📝 License
MIT
