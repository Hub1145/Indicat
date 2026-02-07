# Pro-Trader Ultimate TA-as-a-Service API

A high-performance, enterprise-grade Technical Analysis API built with Python, FastAPI, and industry-standard financial libraries. This service offloads the complex mathematical "heavy lifting" of technical analysis, providing over 200 indicators, candlestick patterns, and advanced price action detection.

## 🚀 Key Features

- **Monetization Ready**: Integrated middleware for RapidAPI proxy verification and tiered access control.
- **Dual Modes of Analysis**:
  - **Market Fetching**: Automatic data retrieval for Crypto (Binance/Kraken via CCXT), Stocks, and Forex (yfinance).
  - **Data Upload**: Accept direct OHLCV JSON uploads for custom datasets or backtesting.
- **Massive Indicator Set**:
  - **200+ Indicators**: Full integration of TA-Lib and the 'ta' (Bukosabino) library.
  - **Deduplication**: Intelligently prefers high-performance C-based TA-Lib functions over overlaps.
  - **Selection & Filtering**: Request specific indicators to optimize bandwidth and speed.
- **Historical Analysis**: Toggle full historical data marking (Buy/Sell signals on every candle) for backtesting and charting.
- **Clean API**: All indicator names are normalized (e.g., `RSI`, `EMA200`) and library-agnostic.
- **Advanced Pattern Recognition**:
  - **Candlestick Patterns**: Over 60 TA-Lib patterns (Hammer, Doji, Engulfing, etc.) with human-readable sentiment mapping.
  - **Price Action Detection**: Native detection of complex chart patterns using high-precision pivot point analysis.
  - **SMC/ICT Strategy**: Built-in detection for Smart Money Concepts like Fair Value Gaps (FVG), Order Blocks (OB), and Market Structure Shifts (MSS).
- **Intelligent Logic**:
  - **Summary Signals**: Aggregate "Buy/Sell/Hold" advice based on RSI, BBands, MACD, and Trend crossovers.
  - **Trend Analysis**: Built-in 200 EMA and ADX trend strength evaluation.
- **Advanced Confluence Engine**: A unified scoring endpoint that aggregates multiple indicators into a single actionable sentiment score (-100 to +100).
- **Performance & Stability**:
  - **Caching**: 60-second disk-based caching (`diskcache`) to prevent redundant external API calls and rate-limiting.
  - **Robust JSON**: Recursive NaN/Inf cleaning ensures 100% JSON compatibility.
  - **Optimized**: Heavy math performed using Numpy arrays.

## 📊 Supported Patterns

### Price Action Patterns (Advanced)
The API identifies complex chart structures using advanced pivot point analysis:
- **Head and Shoulders**: Detects potential trend reversals.
- **Double Top / Double Bottom**: Identifies critical reversal support and resistance.
- **Symmetrical Triangle**: Detects consolidation periods.
- **Ascending Triangle**: Bullish continuation structure.
- **Descending Triangle**: Bearish continuation structure.

### SMC/ICT Strategy (Institutional)
Detects Smart Money Concepts used by professional traders:
- **Fair Value Gaps (FVG)**: Identifies price imbalances and liquidity gaps.
- **Order Blocks (OB)**: Locates institutional buying and selling zones.
- **Market Structure Shift (MSS)**: Detects changes in trend character.

### Candlestick Patterns (TA-Lib)
Includes all 60+ industry-standard candlestick patterns:
- **Hammer / Inverted Hammer**
- **Bullish / Bearish Engulfing**
- **Morning Star / Evening Star**
- **Doji / Dragonfly Doji / Gravestone Doji**
- **Three White Soldiers / Three Black Crows**
- **Shooting Star**
- *And 50+ more...*

## 💰 Tiered Pricing Strategy

| Tier | Price | Features | Limits |
| :--- | :--- | :--- | :--- |
| **Basic** | Free | 1d timeframe, top 10 indicators | 50 req/day |
| **Pro** | $25/mo | All timeframes, all 200+ indicators | 5,000 req/mo |
| **Ultra** | $80/mo | Everything + Pattern Scanning + Trend Checks + Confluence Score | 50,000 req/mo |
| **Mega** | $150/mo | Unlimited requests + Priority Webhook Support | Unlimited |

## 🛠 Tech Stack

- **Framework**: [FastAPI](https://fastapi.tiangolo.com/)
- **Core Math**: [TA-Lib](https://mrjbq7.github.io/ta-lib/) (C-wrapper)
- **Trend/Volatility**: [ta](https://github.com/bukosabino/ta)
- **Market Data**: [CCXT](https://github.com/ccxt/ccxt), [yfinance](https://github.com/ranaroussi/yfinance)
- **Analytics**: Pandas, SciPy, Numpy
- **Infrastructure**: Diskcache, Pydantic, Gunicorn/Uvicorn

## ⚙️ Installation

### 1. Prerequisites (TA-Lib C-Library)
The TA-Lib Python wrapper requires the underlying C-library to be installed.

**Ubuntu/Linux:**
```bash
wget http://prdownloads.sourceforge.net/ta-lib/ta-lib-0.4.0-src.tar.gz
tar -xzf ta-lib-0.4.0-src.tar.gz
cd ta-lib/
./configure --prefix=/usr
make
sudo make install
```

**MacOS:**
```bash
brew install ta-lib
```

### 2. Python Environment
```bash
pip install fastapi uvicorn pandas TA-Lib ta ccxt yfinance diskcache scipy requests
```

## 📖 API Documentation

Once the server is running, visit `http://127.0.0.1:8000/docs` for the interactive Swagger documentation.

### Core Endpoints

#### `GET /indicators`
Returns a categorized list of every indicator, candlestick pattern, and strategy concept supported by the API.

#### `POST /analyze/market`
Fetches and analyzes live data from exchanges.
- **Provider**: `crypto`, `stock`, `forex`
- **Timeframes**: `15m`, `4h`, `1d`
- **Exchange**: `binance` (default), `kraken`, etc.
- **Indicators**: (Optional) List of specific indicators to return (e.g., `["RSI", "EMA200"]`).
- **include_history**: (Optional bool) If `true`, returns signals for the entire 200-candle dataset.

#### `POST /analyze/upload`
Analyzes a provided list of OHLCV data.
- **Data**: List of candles (Max 2000 per request).
- **indicators**: (Optional) List of specific indicators to return.
- **include_history**: (Optional bool) If `true`, returns analysis for the full history instead of just the latest candle.

#### `POST /scan-patterns`
Performs a full scan of the dataset to identify every occurrence of specific candlestick patterns.

#### `POST /is-trend-bullish`
A simplified endpoint that returns a boolean indicating if the current trend is bullish based on EMA 200 and ADX.

#### `POST /confluence-score`
The "Secret Sauce" endpoint. Aggregates RSI, MACD, BBands, and Price Action into a single -100 (Strong Sell) to +100 (Strong Buy) score.

### Example Usage (Python)

```python
import requests

payload = {
    "provider": "crypto",
    "symbol": "BTC/USDT",
    "timeframe": "1d",
    "indicators": ["talib_RSI", "volatility_bbh"]
}

response = requests.post("http://127.0.0.1:8000/analyze/market", json=payload)
print(response.json())
```

## 🚢 Deployment

### Using Docker (Recommended)
The project includes a production-ready Dockerfile that handles the complex TA-Lib C-library installation automatically.

1. **Build and Run:**
```bash
docker-compose up --build -d
```

### Manual Deployment
For production, it is recommended to use Gunicorn with Uvicorn workers to handle high concurrency:

```bash
gunicorn -w 4 -k uvicorn.workers.UvicornWorker main:app --bind 0.0.0.0:8000
```

## 📝 License
MIT
