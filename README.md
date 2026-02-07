# Pro-Trader Ultimate TA-as-a-Service API

A high-performance, production-ready Technical Analysis API built with Python, FastAPI, and industry-standard financial libraries. This service offloads the complex mathematical "heavy lifting" of technical analysis, providing over 200 indicators, candlestick patterns, and advanced price action detection.

## 🚀 Key Features

- **Dual Modes of Analysis**:
  - **Market Fetching**: Automatic data retrieval for Crypto (Binance/Kraken via CCXT), Stocks, and Forex (yfinance).
  - **Data Upload**: Accept direct OHLCV JSON uploads for custom datasets or backtesting.
- **Massive Indicator Set**:
  - **200+ Indicators**: Full integration of TA-Lib and the 'ta' (Bukosabino) library.
  - **Deduplication**: Intelligently prefers high-performance C-based TA-Lib functions over overlaps.
  - **Selection & Filtering**: Request specific indicators to optimize bandwidth and speed.
- **Advanced Pattern Recognition**:
  - **Candlestick Patterns**: Over 60 TA-Lib patterns (Hammer, Doji, Engulfing, etc.) with human-readable sentiment mapping.
  - **Price Action Detection**: Native detection of complex chart patterns (Head and Shoulders, Double Top/Bottom, Triangles) using pivot point analysis.
- **Intelligent Logic**:
  - **Summary Signals**: Aggregate "Buy/Sell/Hold" advice based on RSI, BBands, MACD, and Trend crossovers.
  - **Trend Analysis**: Built-in 200 EMA and ADX trend strength evaluation.
- **Performance & Stability**:
  - **Caching**: 60-second disk-based caching (`diskcache`) to prevent redundant external API calls and rate-limiting.
  - **Robust JSON**: Recursive NaN/Inf cleaning ensures 100% JSON compatibility.
  - **Optimized**: Heavy math performed using Numpy arrays.

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

#### `POST /analyze/market`
Fetches and analyzes live data from exchanges.
- **Provider**: `crypto`, `stock`, `forex`
- **Timeframes**: `15m`, `4h`, `1d`
- **Exchange**: `binance` (default), `kraken`, etc.
- **Indicators**: (Optional) List of specific indicators to return.

#### `POST /analyze/upload`
Analyzes a provided list of OHLCV data.
- **Data**: List of candles (open, high, low, close, volume).

#### `POST /scan-patterns`
Performs a full scan of the dataset to identify every occurrence of specific candlestick patterns.

#### `POST /is-trend-bullish`
A simplified endpoint that returns a boolean indicating if the current trend is bullish based on EMA 200 and ADX.

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

For production, it is recommended to use Gunicorn with Uvicorn workers to handle high concurrency:

```bash
gunicorn -w 4 -k uvicorn.workers.UvicornWorker main:app
```

## 📝 License
MIT
