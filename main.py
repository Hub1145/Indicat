import pandas as pd
import numpy as np
import talib
import ccxt
import yfinance as yf
import ta
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
    indicators: Optional[List[str]] = None

class MarketRequest(BaseModel):
    provider: Literal["crypto", "stock", "forex"]
    symbol: str
    timeframe: Literal["15m", "4h", "1d"]
    exchange: Optional[str] = "binance"
    indicators: Optional[List[str]] = None

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

def get_indicator_status(df: pd.DataFrame, selected_indicators: Optional[List[str]] = None):
    """Calculates indicators and filters them based on user selection."""
    # List of 'ta' library columns to exclude if they have 'talib' equivalents
    DUPLICATE_TA_FEATURES = [
        'momentum_rsi', 'trend_macd', 'trend_macd_signal', 'trend_macd_diff',
        'volatility_bbh', 'volatility_bbl', 'volatility_bbm', 'volatility_bbhi', 'volatility_bbli',
        'volatility_atr', 'trend_adx', 'trend_adx_pos', 'trend_adx_neg',
        'trend_sma_fast', 'trend_sma_slow', 'trend_ema_fast', 'trend_ema_slow',
        'momentum_stoch', 'momentum_stoch_signal', 'momentum_wr', 'trend_cci',
        'volume_obv', 'volume_adi', 'volume_mfi', 'trend_trix', 'momentum_roc'
    ]

    # 1. Prepare data for libraries
    # The 'ta' library expects specific column names (Open, High, Low, Close, Volume)
    ta_df = df.copy()
    ta_df.columns = [c.capitalize() for c in ta_df.columns]

    # 2. Add 'ta' library features (with deduplication)
    try:
        ta_df = ta.add_all_ta_features(
            ta_df, open="Open", high="High", low="Low", close="Close", volume="Volume", fillna=False
        )
        # Remove duplicates
        ta_df = ta_df.drop(columns=[c for c in DUPLICATE_TA_FEATURES if c in ta_df.columns])
    except Exception as e:
        print(f"Error adding 'ta' features: {e}")

    # 3. Add ALL 'talib' candlestick patterns
    op = df['open'].values
    hi = df['high'].values
    lo = df['low'].values
    cl = df['close'].values
    vo = df['volume'].values

    talib_patterns = [f for f in talib.get_functions() if f.startswith('CDL')]
    found_patterns = []

    for pattern_func_name in talib_patterns:
        func = getattr(talib, pattern_func_name)
        # All CDL functions take (open, high, low, close)
        result = func(op, hi, lo, cl)
        ta_df[pattern_func_name] = result
        if result[-1] != 0:
            found_patterns.append({
                "pattern": pattern_func_name,
                "sentiment": "Bullish" if result[-1] > 0 else "Bearish"
            })

    # 4. Add key 'talib' indicators (comprehensive list)
    # Initialize variables to avoid UnboundLocalError
    upper, mid, lower = [np.array([])]*3
    macd, signal, hist = [np.array([])]*3
    talib_indicators = {}

    # - Indicators taking 'close'
    for func_name in ['SMA', 'EMA', 'WMA', 'DEMA', 'TEMA', 'TRIMA', 'KAMA', 'MAMA', 'T3', 'MOM', 'ROC', 'ROCP', 'ROCR', 'ROCR100', 'TRIX', 'STDDEV', 'TSF', 'VAR', 'RSI']:
        try:
            res = getattr(talib, func_name)(cl)
            if isinstance(res, tuple):
                for i, r in enumerate(res):
                    ta_df[f"talib_{func_name}_{i}"] = r
            else:
                ta_df[f"talib_{func_name}"] = res
        except: pass

    # - Indicators taking 'high, low, close'
    for func_name in ['ADX', 'ADXR', 'ATR', 'NATR', 'WILLR', 'CCI', 'DX', 'MINUS_DI', 'MINUS_DM', 'PLUS_DI', 'PLUS_DM', 'ULTOSC', 'MEDPRICE', 'TYPPRICE', 'WCLPRICE', 'SAR']:
        try:
            res = getattr(talib, func_name)(hi, lo, cl)
            if isinstance(res, tuple):
                for i, r in enumerate(res):
                    ta_df[f"talib_{func_name}_{i}"] = r
            else:
                ta_df[f"talib_{func_name}"] = res
        except: pass

    # - Indicators taking 'high, low, close, volume'
    for func_name in ['MFI', 'AD', 'ADOSC', 'OBV']:
        try:
            res = getattr(talib, func_name)(hi, lo, cl, vo)
            if isinstance(res, tuple):
                for i, r in enumerate(res):
                    ta_df[f"talib_{func_name}_{i}"] = r
            else:
                ta_df[f"talib_{func_name}"] = res
        except: pass

    # - Special cases (Multi-output with names)
    try:
        macd, signal, hist = talib.MACD(cl)
        ta_df['talib_MACD'], ta_df['talib_MACD_signal'], ta_df['talib_MACD_hist'] = macd, signal, hist
    except: pass

    try:
        upper, mid, lower = talib.BBANDS(cl)
        ta_df['talib_BB_upper'], ta_df['talib_BB_mid'], ta_df['talib_BB_lower'] = upper, mid, lower
    except: pass

    try:
        slowk, slowd = talib.STOCH(hi, lo, cl)
        ta_df['talib_STOCH_k'], ta_df['talib_STOCH_d'] = slowk, slowd
    except: pass

    try:
        aroondown, aroonup = talib.AROON(hi, lo)
        ta_df['talib_AROON_down'], ta_df['talib_AROON_up'] = aroondown, aroonup
    except: pass

    # 5. Summary and Status Logic
    current_rsi = ta_df['talib_RSI'].iloc[-1] if 'talib_RSI' in ta_df else 50
    ema200 = talib.EMA(cl, timeperiod=min(len(cl), 200))
    current_close = cl[-1]

    # Status Logic
    rsi_status = "Neutral"
    if current_rsi > 70: rsi_status = "Overbought"
    elif current_rsi < 30: rsi_status = "Oversold"

    bb_status = "Inside Bands"
    if current_close >= upper[-1]: bb_status = "Touching Upper Band"
    elif current_close <= lower[-1]: bb_status = "Touching Lower Band"

    macd_status = "Neutral"
    if len(macd) > 1:
        if macd[-1] > signal[-1] and macd[-2] <= signal[-2]:
            macd_status = "Bullish Crossover"
        elif macd[-1] < signal[-1] and macd[-2] >= signal[-2]:
            macd_status = "Bearish Crossover"

    trend = "Bullish" if current_close > ema200[-1] else "Bearish"

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

    # Price Action
    price_action = detect_price_action(df)

    # Categorize indicators for a cleaner response
    latest = ta_df.iloc[-1].to_dict()

    # Filtering logic
    if selected_indicators:
        # Normalize selected indicators to lowercase for easier matching
        selected_lower = [i.lower() for i in selected_indicators]
        filtered_latest = {k: v for k, v in latest.items() if k.lower() in selected_lower}

        # If the user selected specific indicators, we just return those in a flat list
        # plus the summary and price action (which are always included)
        return {
            "current_price": float(current_close),
            "summary": {
                "signal": signal_str,
                "trend": trend,
                "rsi_status": rsi_status,
                "bb_status": bb_status,
                "macd_status": macd_status
            },
            "selected_indicators": filtered_latest,
            "patterns_detected": [p for p in found_patterns if p['pattern'].lower() in selected_lower] if found_patterns else [],
            "price_action": price_action
        }

    categorized = {
        "trend": {k: v for k, v in latest.items() if "trend" in k.lower() or "ema" in k.lower() or "sma" in k.lower()},
        "momentum": {k: v for k, v in latest.items() if "momentum" in k.lower() or "rsi" in k.lower() or "macd" in k.lower() or "stoch" in k.lower()},
        "volatility": {k: v for k, v in latest.items() if "volatility" in k.lower() or "bb" in k.lower() or "atr" in k.lower()},
        "volume": {k: v for k, v in latest.items() if "volume" in k.lower() or "obv" in k.lower() or "ad" in k.lower()},
        "candlestick_patterns": {k: v for k, v in latest.items() if k.startswith("CDL")},
        "others": {k: v for k, v in latest.items() if not any(x in k.lower() for x in ["trend", "momentum", "volatility", "volume", "rsi", "macd", "stoch", "bb", "atr", "obv", "ad", "ema", "sma"]) and not k.startswith("CDL")}
    }

    return {
        "current_price": float(current_close),
        "summary": {
            "signal": signal_str,
            "trend": trend,
            "rsi_status": rsi_status,
            "bb_status": bb_status,
            "macd_status": macd_status
        },
        "indicators": categorized,
        "patterns_detected": found_patterns,
        "price_action": price_action
    }

async def fetch_market_data(provider: str, symbol: str, timeframe: str, exchange_id: str = "binance"):
    """Fetches data from CCXT or yfinance."""
    if provider == "crypto":
        try:
            exchange_class = getattr(ccxt, exchange_id)
            exchange = exchange_class()
        except:
            # Fallback to binance then kraken if specified exchange fails
            try:
                exchange = ccxt.binance()
            except:
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

    analysis = get_indicator_status(df, request.indicators)
    return clean_dict(analysis)

@app.post("/analyze/market")
async def analyze_market(request: MarketRequest):
    # Include indicators in cache key to avoid returning wrong filtered results
    indicators_key = ",".join(sorted(request.indicators)) if request.indicators else "all"
    cache_key = f"{request.provider}_{request.symbol}_{request.timeframe}_{request.exchange}_{indicators_key}"

    cached_res = cache.get(cache_key)
    if cached_res:
        return cached_res

    try:
        df = await fetch_market_data(request.provider, request.symbol, request.timeframe, request.exchange)
        if df.empty:
            raise HTTPException(status_code=404, detail="No data found for symbol.")

        analysis = get_indicator_status(df, request.indicators)

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

@app.post("/scan-patterns")
async def scan_patterns(request: MarketRequest):
    """Scans for patterns across all candles."""
    df = await fetch_market_data(request.provider, request.symbol, request.timeframe, request.exchange)
    if df.empty:
        raise HTTPException(status_code=404, detail="No data found.")

    op = df['open'].values
    hi = df['high'].values
    lo = df['low'].values
    cl = df['close'].values

    talib_patterns = [f for f in talib.get_functions() if f.startswith('CDL')]
    detected = []

    for i in range(len(df)):
        for pattern_func_name in talib_patterns:
            func = getattr(talib, pattern_func_name)
            res = func(op, hi, lo, cl)
            if res[i] != 0:
                detected.append({
                    "index": i,
                    "timestamp": str(df.iloc[i].get('timestamp')),
                    "pattern": pattern_func_name,
                    "sentiment": "Bullish" if res[i] > 0 else "Bearish"
                })

    return {"patterns_found": detected}

@app.post("/is-trend-bullish")
async def is_trend_bullish(request: MarketRequest):
    """Simplified trend check."""
    df = await fetch_market_data(request.provider, request.symbol, request.timeframe, request.exchange)
    if df.empty:
        raise HTTPException(status_code=404, detail="No data found.")

    cl = df['close'].values
    hi = df['high'].values
    lo = df['low'].values

    ema200 = talib.EMA(cl, timeperiod=min(len(cl), 200))
    adx = talib.ADX(hi, lo, cl, timeperiod=14)

    is_bullish = cl[-1] > ema200[-1] and adx[-1] > 20

    return {
        "is_bullish": bool(is_bullish),
        "close": float(cl[-1]),
        "ema200": float(ema200[-1]),
        "adx": float(adx[-1])
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
