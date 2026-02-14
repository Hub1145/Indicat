# Pro-Trader Strategy Intelligence & Institutional Risk Platform

A production-grade Technical Analysis Microservice built with FastAPI and TA-Lib. This platform is not just an indicator provider—it is a comprehensive **Strategy Intelligence Engine** designed to reduce bot development time by 50% while improving statistical performance.

## 🚀 Key Differentiators (Why This Wins)

### 1. Market Regime Detection Engine
Stop trading trend-following strategies in ranging markets.
- **Dynamic Classification**: Automatically identifies if the market is **Trending Bullish/Bearish**, **Ranging (High/Low Volatility)**, or in a **Volatility Squeeze**.
- **Metrics**: ADX Trend Strength, BBWidth Volatility Scores, and Relative ATR percentage.

### 2. AI Strategy Discovery & Optimizer
Don't guess which indicators to use.
- **Automated Optimization**: Scans multiple strategy archetypes (EMA Cross, RSI Mean Reversion, BBands Mean Reversion) for the current market regime.
- **Expectancy Ranking**: Returns the top 3 performing rule-sets based on historical expectancy and win rate.
- **Contextual Recommendation**: Provides data-driven advice on whether to favor trend-following or mean-reversion tools.

### 3. Smart Liquidity Map (Institutional Level)
Trade like the "Big Money."
- **Liquidity Density**: Uses Gaussian distribution to calculate price density around pivot clusters.
- **Sweep Probability**: Assigns a probability score (0-100%) for potential stop-hunts and liquidity sweeps.
- **SMC Detection**: Built-in Fair Value Gaps (FVG), Order Blocks (OB), Breakers, and Market Structure Shifts (MSS/BOS).

### 4. Walk-Forward Backtesting & Monte Carlo Engine
Validate your edge with institutional-grade statistics.
- **Walk-Forward Validation**: Splits data into segments to test strategy consistency across different time periods.
- **Monte Carlo Simulations**: Runs 1000 randomized trade sequence shuffles to determine the probability distribution of outcomes and risk of ruin.
- **Stats**: Win Rate, Profit Factor, Sharpe Ratio, and Max Drawdown.

### 5. Execution & Risk Layer API
Plug-and-play risk management for bot developers.
- **Position Sizing**: `/risk/position-size` calculates exact units and notional value based on account balance, risk percentage, and stop-loss distance.
- **Portfolio Allocator**: `/risk/allocator` distributes capital across multiple assets using **Inverse Volatility Weighting** (lower volatility = higher allocation).

## 📊 Comprehensive Indicator Suite

- **250+ Indicators**: Full **TA-Lib (C-Core)** integration for 150+ standard functions, plus the `ta` library.
- **Specialized Strategy Ports**:
    - **Trend-Pro + Z**: Advanced HMA phase-compensated trend tracking.
    - **UT Bot Alerts**: High-sensitivity ATR-trailing signals.
    - **Squeeze Momentum [LazyBear]**: The definitive BB/KC squeeze indicator.
    - **VWAP Volume Profile**: 50-bin price distribution analysis with Positive/Negative POC.
    - **Adaptive Harmonic Forecast**: Periodogram-based cycle detection and Least Squares path projection.

## 🛠 Tech Stack & Infrastructure

- **FastAPI**: Non-blocking asynchronous API framework.
- **TA-Lib**: Industry-standard high-performance C math engine.
- **PostgreSQL 15 & Redis 7**: Managed persistence for users, requests, and alerts with tiered rate limiting.
- **ProcessPoolExecutor**: Parallelizes CPU-bound calculations for millisecond responses.
- **orjson & GZip**: Ultra-fast serialization and payload compression.
- **TradingView Integration**: `GET /analyze/chart` renders interactive charts with intelligent oscillator pane separation.

## 🚦 Quick Start

### Deployment (Docker Compose)
```bash
docker-compose up --build -d
```
Access the API at `http://localhost:8000` and Swagger docs at `/docs`.

### Environment Configuration
- `RAPIDAPI_PROXY_SECRET`: For RapidAPI gateway integration.
- `DATABASE_URL`: PostgreSQL connection string.
- `REDIS_URL`: Redis connection string.

## 💰 Monetization & Tiering
Configured for RapidAPI hosting with tiered RPM limits:
- **Free**: 10 RPM (Market testing)
- **Pro**: 100 RPM (Professional botting)
- **Enterprise**: 1000 RPM (Institutional grade)

## 🖥 Resource Requirements
Estimated for production deployment (100 concurrent users):
- **RAM**: ~1 GB
- **Storage**: ~2 GB
- **CPU**: 2+ Cores (Recommended for parallel calculation)

---
© 2024 Pro-Trader API | Powering the next generation of algorithmic intelligence.
