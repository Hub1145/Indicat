import pandas as pd
import numpy as np
import talib
import ccxt
import yfinance as yf
import ta
from scipy.signal import argrelextrema
from fastapi import FastAPI, HTTPException, Depends
from pydantic import BaseModel, Field
from typing import List, Optional, Literal
import diskcache
import time
import asyncio
import os
from fastapi import Header, Security

app = FastAPI(title="Master Trader TA API")

# --- Monetization & Security ---
# Set this in your environment or .env file
RAPIDAPI_SECRET = os.getenv("RAPIDAPI_PROXY_SECRET", "dev_secret")

async def verify_rapidapi_key(x_rapidapi_proxy_secret: str = Header(None)):
    """Middleware to verify requests from RapidAPI or other proxies."""
    # In development mode (secret == "dev_secret"), we bypass the check if header is missing
    if RAPIDAPI_SECRET == "dev_secret":
        return True
    if x_rapidapi_proxy_secret != RAPIDAPI_SECRET:
        raise HTTPException(status_code=403, detail="Unauthorized access. Invalid API Key.")
    return True

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
    data: List[Candle] = Field(..., max_length=2000, description="List of OHLCV candles (Max 2000)")
    indicators: Optional[List[str]] = None
    include_history: bool = False

class MarketRequest(BaseModel):
    provider: Literal["crypto", "stock", "forex"]
    symbol: str
    timeframe: Literal["15m", "4h", "1d"]
    exchange: Optional[str] = "binance"
    indicators: Optional[List[str]] = None
    include_history: bool = False

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

def detect_smc_concepts(df: pd.DataFrame):
    """Detects SMC/ICT concepts: Fair Value Gaps, Order Blocks, and Market Structure Shifts."""
    high = df['high'].values
    low = df['low'].values
    close = df['close'].values

    fvgs = []
    order_blocks = []
    mss = "None"

    # 1. Fair Value Gaps (FVG) - Look at last 10 candles for relevant gaps
    # Bullish FVG: low[i] > high[i-2]
    # Bearish FVG: high[i] < low[i-2]
    for i in range(len(df)-1, len(df)-11, -1):
        if i < 2: break
        # Bullish FVG
        if low[i] > high[i-2]:
            fvgs.append({"type": "Bullish FVG", "top": low[i], "bottom": high[i-2], "index": i-1})
        # Bearish FVG
        elif high[i] < low[i-2]:
            fvgs.append({"type": "Bearish FVG", "top": low[i-2], "bottom": high[i], "index": i-1})

    # 2. Order Blocks (OB) - Simplified: Last opposite candle before a significant move
    # Looking for a "displacement" move (ATR based)
    atr = talib.ATR(high, low, close, timeperiod=14)
    for i in range(len(df)-2, len(df)-12, -1):
        if i < 1: break
        move = close[i+1] - close[i]
        if abs(move) > 2 * atr[i]: # Strong displacement
            if move > 0: # Bullish displacement
                order_blocks.append({"type": "Bullish OB", "price": close[i], "index": i})
            else: # Bearish displacement
                order_blocks.append({"type": "Bearish OB", "price": close[i], "index": i})

    # 3. Market Structure Shift (MSS)
    # Check if last candle closed above recent peak or below recent valley
    peak_idx = argrelextrema(high, np.greater, order=5)[0]
    valley_idx = argrelextrema(low, np.less, order=5)[0]

    if len(peak_idx) > 0 and close[-1] > high[peak_idx[-1]]:
        mss = "Bullish MSS"
    elif len(valley_idx) > 0 and close[-1] < low[valley_idx[-1]]:
        mss = "Bearish MSS"

    return {
        "fair_value_gaps": fvgs[:3], # Return top 3 recent
        "order_blocks": order_blocks[:3],
        "market_structure_shift": mss
    }

def clean_column_name(name: str) -> str:
    """Converts internal library names to clean, user-friendly names."""
    orig_name = name
    # Remove common prefixes
    for prefix in ["talib_", "trend_", "momentum_", "volatility_", "volume_", "others_"]:
        if name.startswith(prefix):
            name = name[len(prefix):]
            break

    # Specific mappings for common indicators
    mapping = {
        "bbh": "Bollinger_High",
        "bbl": "Bollinger_Low",
        "bbm": "Bollinger_Mid",
        "macd_signal": "MACD_Signal",
        "macd_diff": "MACD_Hist",
        "stoch_rsi": "Stoch_RSI",
        "ema_fast": "EMA_Fast",
        "ema_slow": "EMA_Slow",
        "sma_fast": "SMA_Fast",
        "sma_slow": "SMA_Slow",
        "ema_20": "EMA20",
        "ema_50": "EMA50",
        "ema_200": "EMA200",
        "sma_20": "SMA20",
    }

    # Check if name is already like EMA_20 (clean prefix was 'talib_')
    if name.lower() in mapping:
        return mapping[name.lower()]

    # If it's something like CDLDOJI, just keep it or clean slightly
    if name.startswith("CDL"):
        return name

    return name.upper()

def get_indicator_status(df: pd.DataFrame, selected_indicators: Optional[List[str]] = None, include_history: bool = False):
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

    talib_results = {}
    for pattern_func_name in talib_patterns:
        func = getattr(talib, pattern_func_name)
        # All CDL functions take (open, high, low, close)
        result = func(op, hi, lo, cl)
        talib_results[pattern_func_name] = result
        if result[-1] != 0:
            found_patterns.append({
                "pattern": pattern_func_name,
                "sentiment": "Bullish" if result[-1] > 0 else "Bearish"
            })

    # 4. Add key 'talib' indicators (comprehensive list)
    # Initialize variables to avoid UnboundLocalError
    upper, mid, lower = [np.array([])]*3
    macd, signal, hist = [np.array([])]*3

    # - Indicators taking 'close'
    for func_name in ['SMA', 'EMA', 'WMA', 'DEMA', 'TEMA', 'TRIMA', 'KAMA', 'MAMA', 'T3', 'MOM', 'ROC', 'ROCP', 'ROCR', 'ROCR100', 'TRIX', 'STDDEV', 'TSF', 'VAR', 'RSI']:
        try:
            res = getattr(talib, func_name)(cl)
            if isinstance(res, tuple):
                for i, r in enumerate(res):
                    talib_results[f"talib_{func_name}_{i}"] = r
            else:
                talib_results[f"talib_{func_name}"] = res
        except: pass

    # - Indicators taking 'high, low, close'
    for func_name in ['ADX', 'ADXR', 'ATR', 'NATR', 'WILLR', 'CCI', 'DX', 'MINUS_DI', 'MINUS_DM', 'PLUS_DI', 'PLUS_DM', 'ULTOSC', 'MEDPRICE', 'TYPPRICE', 'WCLPRICE', 'SAR']:
        try:
            res = getattr(talib, func_name)(hi, lo, cl)
            if isinstance(res, tuple):
                for i, r in enumerate(res):
                    talib_results[f"talib_{func_name}_{i}"] = r
            else:
                talib_results[f"talib_{func_name}"] = res
        except: pass

    # - Indicators taking 'high, low, close, volume'
    for func_name in ['MFI', 'AD', 'ADOSC', 'OBV']:
        try:
            res = getattr(talib, func_name)(hi, lo, cl, vo)
            if isinstance(res, tuple):
                for i, r in enumerate(res):
                    talib_results[f"talib_{func_name}_{i}"] = r
            else:
                talib_results[f"talib_{func_name}"] = res
        except: pass

    # - Special cases (Multi-output with names)
    try:
        macd, signal, hist = talib.MACD(cl)
        talib_results.update({'talib_MACD': macd, 'talib_MACD_signal': signal, 'talib_MACD_hist': hist})
    except: pass

    try:
        upper, mid, lower = talib.BBANDS(cl)
        talib_results.update({'talib_BB_upper': upper, 'talib_BB_mid': mid, 'talib_BB_lower': lower})
    except: pass

    try:
        slowk, slowd = talib.STOCH(hi, lo, cl)
        talib_results.update({'talib_STOCH_k': slowk, 'talib_STOCH_d': slowd})
    except: pass

    try:
        aroondown, aroonup = talib.AROON(hi, lo)
        talib_results.update({'talib_AROON_down': aroondown, 'talib_AROON_up': aroonup})
    except: pass

    # Join all talib results to ta_df at once
    # Ensure no exact duplicate column names before joining
    talib_df = pd.DataFrame(talib_results, index=ta_df.index)
    ta_df = pd.concat([ta_df, talib_df[[c for c in talib_df.columns if c not in ta_df.columns]]], axis=1)

    # 5. Summary and Status Logic (Vectorized for history)
    rsi_history = ta_df['talib_RSI'] if 'talib_RSI' in ta_df else pd.Series([50]*len(df))
    ema200_history = talib.EMA(cl, timeperiod=min(len(cl), 200))

    scores = pd.Series(0.0, index=df.index)

    # RSI signals
    scores[rsi_history < 30] += 1
    scores[rsi_history > 70] -= 1

    # BBands signals
    if not upper.size == 0:
        scores[cl >= upper] -= 1
        scores[cl <= lower] += 1

    # MACD signals
    if macd is not None and signal is not None and len(macd) > 1:
        macd_s = pd.Series(macd, index=df.index)
        signal_s = pd.Series(signal, index=df.index)
        # Bullish crossover
        bullish_cross = (macd_s > signal_s) & (macd_s.shift(1) <= signal_s.shift(1))
        scores[bullish_cross] += 1
        # Bearish crossover
        bearish_cross = (macd_s < signal_s) & (macd_s.shift(1) >= signal_s.shift(1))
        scores[bearish_cross] -= 1

    # Trend signals
    scores[cl > ema200_history] += 0.5
    scores[cl <= ema200_history] -= 0.5

    # Map scores to signals
    def score_to_signal(s):
        if s >= 1.5: return "Buy"
        elif s <= -1.5: return "Sell"
        return "Hold"

    df['signal'] = scores.apply(score_to_signal)

    # 6. Final Clean Data Prep
    # Map all internal names to clean names, ensuring uniqueness
    new_cols = []
    seen = set()
    for c in ta_df.columns:
        clean = clean_column_name(c)
        if clean in seen:
            # Append library prefix if collision
            if c.startswith("talib_"): clean = f"TALIB_{clean}"
            elif "_" in c: clean = f"{c.split('_', 1)[0].upper()}_{clean}"

            # Final fallback if still seen
            temp_clean = clean
            i = 1
            while temp_clean in seen:
                temp_clean = f"{clean}_{i}"
                i += 1
            clean = temp_clean

        seen.add(clean)
        new_cols.append(clean)

    ta_df.columns = new_cols

    # Price Action & SMC (already calculated for latest, but we need summary status for latest)
    price_action = detect_price_action(df)
    smc = detect_smc_concepts(df)

    latest_idx = -1
    current_rsi = rsi_history.iloc[latest_idx]
    rsi_status = "Neutral"
    if current_rsi > 70: rsi_status = "Overbought"
    elif current_rsi < 30: rsi_status = "Oversold"

    bb_status = "Inside Bands"
    if not upper.size == 0:
        if cl[latest_idx] >= upper[latest_idx]: bb_status = "Touching Upper Band"
        elif cl[latest_idx] <= lower[latest_idx]: bb_status = "Touching Lower Band"

    macd_status = "Neutral"
    if not macd.size == 0:
        if macd[latest_idx] > signal[latest_idx] and macd[latest_idx-1] <= signal[latest_idx-1]:
            macd_status = "Bullish Crossover"
        elif macd[latest_idx] < signal[latest_idx] and macd[latest_idx-1] >= signal[latest_idx-1]:
            macd_status = "Bearish Crossover"

    trend = "Bullish" if cl[latest_idx] > ema200_history[latest_idx] else "Bearish"

    # 7. Construct Response
    if include_history:
        # Merge signals and clean indicators back to original df
        output_df = df.copy()
        for col in ta_df.columns:
            if col not in output_df.columns:
                output_df[col] = ta_df[col]

        # Filtering for history
        if selected_indicators:
            selected_clean = [clean_column_name(i) for i in selected_indicators]
            # Keep core OHLCV + signal + selected
            keep = ['timestamp', 'open', 'high', 'low', 'close', 'volume', 'signal'] + [c for c in output_df.columns if c in selected_clean]
            output_df = output_df[keep]

        return {
            "summary": {
                "signal": df['signal'].iloc[-1],
                "trend": trend,
                "rsi_status": rsi_status
            },
            "history": output_df.to_dict(orient="records")
        }

    # Latest only response (Categorized)
    latest_indicators = ta_df.iloc[-1].to_dict()
    if selected_indicators:
        selected_clean = [clean_column_name(i) for i in selected_indicators]
        latest_indicators = {k: v for k, v in latest_indicators.items() if k in selected_clean}

    categorized = {
        "trend": {k: v for k, v in latest_indicators.items() if any(x in k.lower() for x in ["trend", "ema", "sma", "ichimoku", "psar", "adx", "aroon"])},
        "momentum": {k: v for k, v in latest_indicators.items() if any(x in k.lower() for x in ["momentum", "rsi", "macd", "stoch", "tsi", "uo", "roc", "ppo", "pvo", "kama"])},
        "volatility": {k: v for k, v in latest_indicators.items() if any(x in k.lower() for x in ["volatility", "bollinger", "atr", "ui", "kc", "dc"])},
        "volume": {k: v for k, v in latest_indicators.items() if any(x in k.lower() for x in ["volume", "obv", "adi", "mfi", "cmf", "fi", "em", "vpt", "vwap", "nvi", "ad"])},
        "candlestick_patterns": {k: v for k, v in latest_indicators.items() if k.startswith("CDL")},
        "others": {k: v for k, v in latest_indicators.items() if not any(x in k.lower() for x in ["trend", "momentum", "volatility", "volume", "rsi", "macd", "stoch", "bollinger", "atr", "obv", "ad", "ema", "sma", "psar", "ichimoku", "adx", "aroon", "tsi", "uo", "roc", "ppo", "pvo", "kama", "ui", "kc", "dc", "adi", "mfi", "cmf", "fi", "em", "vpt", "vwap", "nvi"]) and not k.startswith("CDL")}
    }

    return {
        "current_price": float(cl[-1]),
        "summary": {
            "signal": df['signal'].iloc[-1],
            "trend": trend,
            "rsi_status": rsi_status,
            "bb_status": bb_status,
            "macd_status": macd_status
        },
        "indicators": categorized if not selected_indicators else latest_indicators,
        "patterns_detected": [p for p in found_patterns if p['pattern'].lower() in [i.lower() for i in (selected_indicators or [])]] if selected_indicators else found_patterns,
        "price_action": price_action,
        "smc": smc
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

@app.get("/indicators")
async def get_available_indicators():
    """Returns a categorized list of all available indicators and patterns."""
    talib_patterns = sorted([f for f in talib.get_functions() if f.startswith('CDL')])

    # Core categories with clean names
    core_indicators = sorted(['SMA', 'EMA', 'WMA', 'DEMA', 'TEMA', 'TRIMA', 'KAMA', 'MAMA', 'T3', 'MOM', 'ROC', 'ROCP', 'ROCR', 'ROCR100', 'TRIX', 'STDDEV', 'TSF', 'VAR', 'RSI', 'ADX', 'ADXR', 'ATR', 'NATR', 'WILLR', 'CCI', 'DX', 'MINUS_DI', 'MINUS_DM', 'PLUS_DI', 'PLUS_DM', 'ULTOSC', 'MEDPRICE', 'TYPPRICE', 'WCLPRICE', 'SAR', 'MFI', 'AD', 'ADOSC', 'OBV', 'MACD', 'BBANDS', 'STOCH', 'AROON'])

    # 'ta' library categories (using clean names)
    ta_indicators = [
        "ADI", "OBV", "CMF", "FI", "EM", "VPT", "VWAP", "MFI", "NVI",
        "BOLLINGER_MID", "BOLLINGER_HIGH", "BOLLINGER_LOW", "ATR", "UI",
        "MACD", "MACD_SIGNAL", "MACD_HIST", "SMA_FAST", "SMA_SLOW", "EMA_FAST", "EMA_SLOW",
        "VORTEX_POS", "VORTEX_NEG", "TRIX", "MASS_INDEX", "DPO", "KST", "ICHIMOKU_A", "ICHIMOKU_B", "STC", "ADX", "CCI", "AROON_UP", "AROON_DOWN", "PSAR",
        "EMA20", "EMA50", "EMA200", "SMA20"
    ]

    return {
        "technical_indicators": sorted(list(set(core_indicators + ta_indicators))),
        "candlestick_patterns": talib_patterns,
        "price_action": [
            "Head and Shoulders", "Double Top", "Double Bottom",
            "Symmetrical Triangle", "Descending Triangle", "Ascending Triangle"
        ],
        "smc_ict": ["Fair Value Gap (FVG)", "Order Block (OB)", "Market Structure Shift (MSS)"]
    }

@app.post("/analyze/upload", dependencies=[Depends(verify_rapidapi_key)])
async def analyze_upload(request: UploadRequest):
    if len(request.data) < 30:
        raise HTTPException(status_code=400, detail="Need at least 30 candles.")

    # Trim to last 2000 candles if more provided (SaaS safeguard)
    data = request.data[-2000:]
    df = pd.DataFrame([c.model_dump() for c in data])

    analysis = get_indicator_status(df, request.indicators, request.include_history)
    return clean_dict(analysis)

@app.post("/analyze/market", dependencies=[Depends(verify_rapidapi_key)])
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

        analysis = get_indicator_status(df, request.indicators, request.include_history)

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

@app.post("/scan-patterns", dependencies=[Depends(verify_rapidapi_key)])
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

@app.post("/is-trend-bullish", dependencies=[Depends(verify_rapidapi_key)])
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

@app.post("/confluence-score", dependencies=[Depends(verify_rapidapi_key)])
async def get_confluence_score(request: MarketRequest):
    """
    Advanced Endpoint: Returns a unified Confluence Score from -100 to +100.
    Combines RSI, MACD, Bollinger Bands, and Price Action.
    """
    df = await fetch_market_data(request.provider, request.symbol, request.timeframe, request.exchange)
    if df.empty:
        raise HTTPException(status_code=404, detail="No data found.")

    analysis = get_indicator_status(df)
    summary = analysis['summary']
    pa = analysis['price_action']
    smc = analysis['smc']

    score = 0

    # 1. Base Signal Score (up to 40 points)
    if summary['signal'] == "Buy": score += 40
    elif summary['signal'] == "Sell": score -= 40

    # 2. Trend Confluence (20 points)
    if summary['trend'] == "Bullish": score += 20
    else: score -= 20

    # 3. RSI Extremes (15 points)
    if summary['rsi_status'] == "Oversold": score += 15
    elif summary['rsi_status'] == "Overbought": score -= 15

    # 4. Price Action Bonus (15 points)
    if any(p in pa for p in ["Double Bottom", "Ascending Triangle"]): score += 15
    if any(p in pa for p in ["Double Top", "Descending Triangle", "Head and Shoulders"]): score -= 15

    # 5. SMC/ICT Bias (10 points)
    if smc['market_structure_shift'] == "Bullish MSS": score += 10
    elif smc['market_structure_shift'] == "Bearish MSS": score -= 10

    # Ensure range -100 to +100
    score = max(-100, min(100, score))

    sentiment = "Neutral"
    if score > 50: sentiment = "Strong Buy"
    elif score > 10: sentiment = "Buy"
    elif score < -50: sentiment = "Strong Sell"
    elif score < -10: sentiment = "Sell"

    return {
        "symbol": request.symbol,
        "confluence_score": score,
        "sentiment": sentiment,
        "components": {
            "summary": summary['signal'],
            "trend": summary['trend'],
            "price_action_found": pa
        }
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
