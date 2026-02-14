# Pro-Trader Ultimate TA-as-a-Service API

A high-performance, commercial-grade Technical Analysis API built with FastAPI and TA-Lib. Designed for trading bot developers, fintech startups, and institutional-grade algorithmic analysis. This platform offloads the "math heavy lifting" and complex strategy logic into a scalable microservice.

## 🚀 Enterprise-Grade Features

### 1. Advanced Indicator Math Engine
- **250+ Indicators**: Full integration of **TA-Lib (C-Core)** for maximum speed, plus the `ta` library for high-level trend/momentum analysis.
- **Dynamic Discovery**: The `GET /indicators` endpoint allows clients to explore all 150+ TA-Lib functions and custom strategy models programmatically.

### 2. Smart Money Concepts (SMC) & ICT Toolkit
Professional-grade port of high-end strategy concepts:
- **Market Structure**: Automatic detection of Market Structure Shifts (MSS) and Break of Structure (BOS).
- **Liquidity Imbalances**: Real-time identification of Fair Value Gaps (FVG) and Balanced Price Ranges (BPR).
- **Institutional Zones**: Detection of Order Blocks and Breaker Blocks with mitigation tracking.
- **Liquidity Pools**: Clustering of pivot points to identify major buy-side and sell-side liquidity.
- **Volume Imbalance**: Detection of gaps between candle bodies.
- **Opening Gaps**: New Week Opening Gaps (NWOG) and New Day Opening Gaps (NDOG).

### 3. Predictive Forecasting & Modeling
- **Adaptive Harmonic Forecast**: A LuxAlgo-inspired model that uses Periodogram-based cycle detection and Least Squares fitting to project future price paths.
- **MTF MACD Forecast**: Statistical forecasting using a memory of relative price movements within higher timeframe trend segments.

### 4. Specialized Strategy Ports
- **Trend-Pro + Z**: Advanced trend-following system using HMA phase compensation and ATR-based trailing scores.
- **UT Bot Alerts**: High-sensitivity buy/sell signal engine using recursive ATR trailing stops.
- **Squeeze Momentum [LazyBear]**: BB/KC squeeze detection with linear regression momentum visualization.
- **Z-Score Predictive Zones**: LuxAlgo port for volatility-normalized overbought/oversold identification.

### 5. Institutional Risk & Portfolio Tools
- **Options Greeks**: Black-Scholes engine for calculating Delta and other greeks.
- **Portfolio Metrics**: Asset correlation matrices, Value at Risk (VaR 95), and Portfolio Beta.
- **Market Dynamics**: Multi-Timeframe (MTF) analysis, horizontal S/R level clustering, and VWAP Volume Profiles.

### 6. High-Performance Infrastructure
- **TradingView Integration**: `GET /analyze/chart` generates interactive Lightweight Charts with intelligent oscillator pane separation.
- **Distributed Ready**: Docker-compose orchestration with **PostgreSQL 15** and **Redis 7**.
- **Tiered Monetization**: Built-in support for Free, Pro, and Enterprise tiers with Redis-backed rate limiting.
- **Parallel Execution**: CPU-bound calculations are parallelized using `ProcessPoolExecutor`.
- **Fast Delivery**: Optimized with `orjson` and GZip compression for millisecond response times.

## 🛠 Tech Stack

- **FastAPI**: Asynchronous Python web framework.
- **TA-Lib**: Industry-standard C-based financial math library.
- **yfinance & Binance**: Unified data fetching for Stocks, Forex, and Crypto.
- **SQLAlchemy/PostgreSQL**: Request logging, user management, and alert persistence.
- **Redis**: Rate limiting and real-time state management.
- **SciPy/NumPy**: Advanced signal processing and matrix mathematics.

## 🚦 Quick Start

### 1. Prerequisites
- Docker and Docker Compose.
- (Optional) TA-Lib C-library if running without Docker.

### 2. Deployment
```bash
# Clone and start the infrastructure
docker-compose up --build -d
```
The API will be available at `http://localhost:8000`.

### 3. Environment Variables
- `RAPIDAPI_PROXY_SECRET`: Secret for validating RapidAPI proxy headers.
- `DATABASE_URL`: Connection string for PostgreSQL.
- `REDIS_URL`: Connection string for Redis.

## 📖 API Usage Guide

### Full Market Analysis
`POST /analyze/market`
```json
{
  "provider": "crypto",
  "symbol": "BTC/USDT",
  "timeframe": "1h",
  "indicators": ["RSI", "MACD", "EMA200"],
  "include_history": false
}
```

### Interactive Charting
Open in your browser:
`http://localhost:8000/analyze/chart?symbol=ETH/USDT&timeframe=1d&indicators=RSI,EMA20,EMA50`

### Confluence Scoring
`POST /confluence-score`
Returns a consolidated sentiment score from -100 to +100 based on multiple indicators and market structure.

## 💰 Monetization Strategy

The platform is designed to be hosted on **RapidAPI**. It includes middleware to verify the `X-RapidAPI-Proxy-Secret`.
- **Free Tier**: 10 RPM (Requests Per Minute)
- **Pro Tier**: 100 RPM
- **Enterprise Tier**: 1000 RPM

---
© 2024 Pro-Trader API | Built for the next generation of algorithmic traders.
