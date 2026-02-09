# Pro-Trader Ultimate TA-as-a-Service API

A high-performance, commercial-grade Technical Analysis API built with FastAPI and TA-Lib. Designed for trading bot developers and institutional-grade analysis.

## 🚀 Key Features

- **250+ Indicators**: Full TA-Lib suite plus `pandas-ta` and `ta` library integration.
- **SMC/ICT Toolkit**: Automatic detection of Fair Value Gaps (FVG), Order Blocks (OB), and Market Structure Shifts (MSS).
- **Price Action Engine**: SciPy-powered detection of Head & Shoulders, Triangles, Double Tops/Bottoms.
- **LuxAlgo Customs**: Ported logic for Z-Score Predictive Zones and Market Structure Break toolkit.
- **Institutional Tools**: Multi-Timeframe (MTF) analysis, Asset Correlation Matrices, and Options Greeks (Black-Scholes).
- **SaaS Ready**: RapidAPI middleware, Disk Caching, and automatic data normalization.

## 🛠 Tech Stack

- **FastAPI**: Asynchronous API framework.
- **TA-Lib**: Industry-standard C-based math engine.
- **CCXT & yfinance**: Live data fetching for Crypto, Stocks, and Forex.
- **SciPy**: Advanced signal processing for pattern recognition.

## 🚦 Quick Start

### 1. Installation (Local)
```bash
# Requires TA-Lib C-library
pip install -r requirements.txt
python main.py
```

### 2. Docker Deployment
```bash
docker-compose up --build -d
```

## 📖 API Documentation

Once running, visit `http://localhost:8000/docs` for the interactive Swagger UI.

### Key Endpoints:
- `POST /analyze/market`: Full analysis of a specific asset.
- `POST /analyze/upload`: Analyze your own OHLCV JSON data.
- `GET /indicators`: List all available indicators and patterns.
- `POST /confluence-score`: Get a consolidated sentiment score (-100 to +100).
- `POST /analyze/mtf`: Multi-timeframe trend analysis.

## 💰 Monetization

This API is pre-configured for **RapidAPI**.
- Set `RAPIDAPI_PROXY_SECRET` in your environment.
- The middleware automatically validates requests coming through the RapidAPI proxy.

---
Built for speed, accuracy, and profitability.
