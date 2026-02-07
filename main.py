import pandas as pd
import numpy as np
import talib
import ccxt
import yfinance as yf
from ta.trend import IchimokuIndicator
from scipy.signal import argrelextrema
from fastapi import FastAPI, HTTPException, Depends
from pydantic import BaseModel
from typing import List, Optional, Literal
import diskcache
import time
import asyncio

app = FastAPI(title="Master Trader TA API")

# --- Caching Setup ---
# Cache results for 60 seconds
cache = diskcache.Cache("./cache")

# --- Models ---

class Candle(BaseModel):
    timestamp: Optional[str] = None
    open: float
    high: float
    low: float
    close: float
    volume: float

class UploadRequest(BaseModel):
    data: List[Candle]

class MarketRequest(BaseModel):
    provider: Literal["crypto", "stock", "forex"]
    symbol: str
    timeframe: Literal["15m", "4h", "1d"]

# --- Helpers ---

def clean_dict(d):
    """Recursively replaces NaN with None for JSON compatibility."""
    if isinstance(d, dict):
        return {k: clean_dict(v) for k, v in d.items()}
    elif isinstance(d, list):
        return [clean_dict(v) for v in d]
    elif isinstance(d, (float, np.float64, np.float32)):
        if np.isnan(d) or np.isinf(d):
            return None
        return float(d)
    return d

def detect_price_action(df: pd.DataFrame):
    """Detects Head and Shoulders, Double Top/Bottom, and Triangles."""
    close = df['close'].values
    high = df['high'].values
    low = df['low'].values

    # Find pivots (order=5 means 5 candles on each side)
    peak_idx = argrelextrema(high, np.greater, order=5)[0]
    valley_idx = argrelextrema(low, np.less, order=5)[0]

    peaks = high[peak_idx]
    valleys = low[valley_idx]

    patterns = []

    # 1. Head and Shoulders
    if len(peaks) >= 3:
        p1, p2, p3 = peaks[-3], peaks[-2], peaks[-1]
        if p2 > p1 and p2 > p3:
            # Check if shoulders are similar height (within 10%)
            if abs(p1 - p3) / max(p1, p3) < 0.1:
                patterns.append("Head and Shoulders")

    # 2. Double Top
    if len(peaks) >= 2:
        p1, p2 = peaks[-2], peaks[-1]
        if abs(p1 - p2) / max(p1, p2) < 0.02: # Within 2%
            patterns.append("Double Top")

    # 3. Double Bottom
    if len(valleys) >= 2:
        v1, v2 = valleys[-2], valleys[-1]
        if abs(v1 - v2) / max(v1, v2) < 0.02:
            patterns.append("Double Bottom")

    # 4. Triangles (Descending/Ascending/Symmetrical)
    if len(peaks) >= 2 and len(valleys) >= 2:
        # Check slopes of recent 2 pivots
        high_slope = (peaks[-1] - peaks[-2]) / (peak_idx[-1] - peak_idx[-2])
        low_slope = (valleys[-1] - valleys[-2]) / (valley_idx[-1] - valley_idx[-2])

        if high_slope < 0 and low_slope > 0:
            patterns.append("Symmetrical Triangle")
        elif high_slope < 0 and abs(low_slope) < 0.001:
            patterns.append("Descending Triangle")
        elif low_slope > 0 and abs(high_slope) < 0.001:
            patterns.append("Ascending Triangle")

    return patterns

def get_indicator_status(df: pd.DataFrame):
    """Calculates indicator values and their human-readable status."""
    cl = df['close'].values
    hi = df['high'].values
    lo = df['low'].values

    # RSI
    rsi = talib.RSI(cl, timeperiod=14)
    current_rsi = rsi[-1]
    rsi_status = "Neutral"
    if current_rsi > 70: rsi_status = "Overbought"
    elif current_rsi < 30: rsi_status = "Oversold"

    # Bollinger Bands
    upper, mid, lower = talib.BBANDS(cl, timeperiod=20)
    current_close = cl[-1]
    bb_status = "Inside Bands"
    if current_close >= upper[-1]: bb_status = "Touching Upper Band"
    elif current_close <= lower[-1]: bb_status = "Touching Lower Band"

    # MACD
    macd, signal, hist = talib.MACD(cl)
    macd_status = "Neutral"
    if macd[-1] > signal[-1] and macd[-2] <= signal[-2]:
        macd_status = "Bullish Crossover"
    elif macd[-1] < signal[-1] and macd[-2] >= signal[-2]:
        macd_status = "Bearish Crossover"

    # EMA 200 Trend
    ema200 = talib.EMA(cl, timeperiod=min(len(cl), 200))
    trend = "Bullish" if current_close > ema200[-1] else "Bearish"

    # Candlestick Patterns (Using full series for context, then checking latest 3)
    op = df['open'].values
    candlestick_patterns = []

    # Run on full series
    hammers = talib.CDLHAMMER(op, hi, lo, cl)
    dojis = talib.CDLDOJI(op, hi, lo, cl)
    engulfing = talib.CDLENGULFING(op, hi, lo, cl)

    # Check latest 3
    if any(hammers[-3:] != 0): candlestick_patterns.append("Hammer")
    if any(dojis[-3:] != 0): candlestick_patterns.append("Doji")
    if any(engulfing[-3:] != 0): candlestick_patterns.append("Engulfing")

    # Price Action Patterns
    price_action_patterns = detect_price_action(df)

    # Ichimoku Cloud
    ichimoku = IchimokuIndicator(high=df['high'], low=df['low'])
    span_a = ichimoku.ichimoku_a()
    span_b = ichimoku.ichimoku_b()

    # Summary Signal Logic
    score = 0
    if rsi_status == "Oversold": score += 1
    if rsi_status == "Overbought": score -= 1
    if bb_status == "Touching Lower Band": score += 1
    if bb_status == "Touching Upper Band": score -= 1
    if macd_status == "Bullish Crossover": score += 1
    if macd_status == "Bearish Crossover": score -= 1
    if trend == "Bullish": score += 0.5
    else: score -= 0.5

    signal_str = "Hold"
    if score >= 1.5: signal_str = "Buy"
    elif score <= -1.5: signal_str = "Sell"

    return {
        "current_price": float(current_close),
        "indicators": {
            "rsi": {"value": float(current_rsi), "status": rsi_status},
            "bbands": {"upper": float(upper[-1]), "lower": float(lower[-1]), "status": bb_status},
            "macd": {"value": float(macd[-1]), "status": macd_status},
            "ema200": {"value": float(ema200[-1]), "status": trend},
            "ichimoku": {"span_a": float(span_a.iloc[-1]), "span_b": float(span_b.iloc[-1])}
        },
        "candlestick_patterns": candlestick_patterns,
        "price_action_patterns": price_action_patterns,
        "summary_signal": signal_str
    }

async def fetch_market_data(provider: str, symbol: str, timeframe: str):
    """Fetches data from CCXT or yfinance."""
    if provider == "crypto":
        # Using Kraken as Binance is restricted in some environments
        exchange = ccxt.kraken()
        ohlcv = exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=200)
        df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    else:
        # yfinance mapping
        yf_map = {"15m": "15m", "4h": "1h", "1d": "1d"}
        # Adjust period to ensure 200 candles
        period_map = {"15m": "1mo", "4h": "1mo", "1d": "2y"}

        ticker_symbol = symbol if provider == "stock" else f"{symbol}=X"

        data = yf.download(ticker_symbol, period=period_map[timeframe], interval=yf_map[timeframe], progress=False)

        # Handle yfinance MultiIndex columns if present
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)

        if timeframe == "4h":
            # Resample 1h to 4h
            data = data.resample('4h').agg({
                'Open': 'first',
                'High': 'max',
                'Low': 'min',
                'Close': 'last',
                'Volume': 'sum'
            }).dropna()

        df = data.tail(200).reset_index()
        df.columns = [str(c).lower() for c in df.columns]

        if 'date' in df.columns: df = df.rename(columns={'date': 'timestamp'})
        if 'datetime' in df.columns: df = df.rename(columns={'datetime': 'timestamp'})

    return df

# --- Endpoints ---

@app.post("/analyze/upload")
async def analyze_upload(request: UploadRequest):
    if len(request.data) < 30:
        raise HTTPException(status_code=400, detail="Need at least 30 candles.")

    df = pd.DataFrame([c.model_dump() for c in request.data])

    analysis = get_indicator_status(df)
    return clean_dict(analysis)

@app.post("/analyze/market")
async def analyze_market(request: MarketRequest):
    cache_key = f"{request.provider}_{request.symbol}_{request.timeframe}"
    cached_res = cache.get(cache_key)
    if cached_res:
        return cached_res

    try:
        df = await fetch_market_data(request.provider, request.symbol, request.timeframe)
        if df.empty:
            raise HTTPException(status_code=404, detail="No data found for symbol.")

        analysis = get_indicator_status(df)

        response = {
            "meta_data": {
                "symbol": request.symbol,
                "timeframe": request.timeframe,
                "provider": request.provider,
                "timestamp": time.time()
            },
            **analysis
        }

        cleaned_response = clean_dict(response)

        # Cache for 60 seconds
        cache.set(cache_key, cleaned_response, expire=60)
        return cleaned_response

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
