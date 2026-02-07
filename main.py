import pandas as pd
import numpy as np
import talib
import ccxt
import yfinance as yf
import ta
from scipy.signal import argrelextrema
from scipy.stats import norm
from fastapi import FastAPI, HTTPException, Depends, Header
from pydantic import BaseModel, Field
from typing import List, Optional, Literal, Dict, Any
import diskcache
import time
import asyncio
import os
from datetime import datetime

app = FastAPI(title="Pro-Trader Ultimate TA-as-a-Service API")

# --- Monetization & Security ---
RAPIDAPI_SECRET = os.getenv("RAPIDAPI_PROXY_SECRET", "dev_secret")

async def verify_rapidapi_key(x_rapidapi_proxy_secret: str = Header(None)):
    if RAPIDAPI_SECRET == "dev_secret": return True
    if x_rapidapi_proxy_secret != RAPIDAPI_SECRET:
        raise HTTPException(status_code=403, detail="Unauthorized access. Invalid API Key.")
    return True

# --- Caching ---
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
    data: List[Candle] = Field(..., max_length=2000)
    indicators: Optional[List[str]] = None
    include_history: bool = False

class MarketRequest(BaseModel):
    provider: Literal["crypto", "stock", "forex"]
    symbol: str
    timeframe: Literal["15m", "1h", "4h", "1d"]
    exchange: Optional[str] = "binance"
    indicators: Optional[List[str]] = None
    include_history: bool = False

class MTFRequest(BaseModel):
    provider: Literal["crypto", "stock", "forex"]
    symbol: str
    timeframes: List[Literal["15m", "1h", "4h", "1d"]]
    exchange: Optional[str] = "binance"
    indicators: Optional[List[str]] = None

class CorrelationRequest(BaseModel):
    assets: List[str]
    provider: Literal["crypto", "stock", "forex"] = "crypto"
    timeframe: str = "1d"

class OptionsRequest(BaseModel):
    underlying_price: float
    strike: float
    expiry: str
    volatility: float
    risk_free_rate: float = 0.05
    option_type: Literal["call", "put"] = "call"

class HeatmapRequest(BaseModel):
    assets: List[str]
    provider: Literal["crypto", "stock", "forex"] = "crypto"
    metric: str = "RSI"

# --- Metadata ---

INDICATOR_METADATA = {
    "technical_indicators": {
        "SMA": "Simple Moving Average", "EMA": "Exponential Moving Average", "WMA": "Weighted Moving Average",
        "RSI": "Relative Strength Index", "ADX": "Average Directional Index", "ATR": "Average True Range",
        "MACD": "Moving Average Convergence Divergence", "BBANDS": "Bollinger Bands",
        "VWAP": "Volume Weighted Average Price", "OBV": "On Balance Volume", "CMF": "Chaikin Money Flow",
        "EMA20": "20-Period EMA", "EMA50": "50-Period EMA", "EMA200": "200-Period EMA"
    },
    "patterns": {
        "CDLDOJI": "Doji", "CDLHAMMER": "Hammer", "CDLENGULFING": "Engulfing",
        "Head and Shoulders": "Head and Shoulders Reversal",
        "Double Top": "Double Top", "Double Bottom": "Double Bottom",
        "Ascending Triangle": "Ascending Triangle", "Descending Triangle": "Descending Triangle"
    },
    "institutional_strategies": {
        "FVG": "Fair Value Gap (Liquidity Imbalance)",
        "OB": "Institutional Order Block",
        "MSS": "Market Structure Shift"
    }
}

# --- Core Logic Helpers ---

def clean_dict(d):
    if isinstance(d, dict): return {k: clean_dict(v) for k, v in d.items()}
    if isinstance(d, list): return [clean_dict(v) for v in d]
    if isinstance(d, (float, np.float64, np.float32)):
        return None if np.isnan(d) or np.isinf(d) else float(d)
    return d

def clean_column_name(name: str) -> str:
    for prefix in ["talib_", "trend_", "momentum_", "volatility_", "volume_", "others_"]:
        if name.startswith(prefix):
            name = name[len(prefix):]
            break
    mapping = {
        "bbh": "Bollinger_High", "bbl": "Bollinger_Low", "bbm": "Bollinger_Mid",
        "macd_signal": "MACD_Signal", "macd_diff": "MACD_Hist",
        "ema_20": "EMA20", "ema_50": "EMA50", "ema_200": "EMA200", "sma_20": "SMA20"
    }
    return mapping.get(name.lower(), name.upper())

def get_sessions(timestamp_str: Optional[str]) -> List[str]:
    if not timestamp_str: return []
    try:
        if str(timestamp_str).isdigit(): dt = datetime.fromtimestamp(int(timestamp_str)/1000.0)
        else: dt = datetime.fromisoformat(str(timestamp_str).replace('Z', '+00:00'))
        h = dt.hour
        s = []
        if 0 <= h < 9: s.append("Tokyo")
        if 8 <= h < 17: s.append("London")
        if 13 <= h < 22: s.append("New York")
        return s
    except: return []

def detect_sr_levels(df: pd.DataFrame):
    cl = df['close'].values
    p = argrelextrema(df['high'].values, np.greater, order=10)[0]
    v = argrelextrema(df['low'].values, np.less, order=10)[0]
    pivots = np.sort(np.concatenate([df['high'].values[p], df['low'].values[v]]))
    if not len(pivots): return []
    groups = []
    curr = [pivots[0]]
    for i in range(1, len(pivots)):
        if (pivots[i] - np.mean(curr)) / np.mean(curr) < 0.01: curr.append(pivots[i])
        else:
            groups.append(np.mean(curr))
            curr = [pivots[i]]
    groups.append(np.mean(curr))
    cp = cl[-1]
    return [{"type": "Resistance" if l > cp else "Support", "price": float(l)} for l in groups]

def detect_smc_concepts(df: pd.DataFrame):
    h, l, c = df['high'].values, df['low'].values, df['close'].values
    fvgs, obs = [], []
    for i in range(len(df)-1, len(df)-11, -1):
        if i < 2: break
        if l[i] > h[i-2]: fvgs.append({"type": "Bullish FVG", "top": float(l[i]), "bottom": float(h[i-2]), "index": i-1})
        elif h[i] < l[i-2]: fvgs.append({"type": "Bearish FVG", "top": float(l[i-2]), "bottom": float(h[i]), "index": i-1})

    atr = talib.ATR(h, l, c)
    for i in range(len(df)-2, len(df)-12, -1):
        move = c[i+1] - c[i]
        if i < len(atr) and abs(move) > 2 * (atr[i] or 1):
            obs.append({"type": "Bullish OB" if move > 0 else "Bearish OB", "price": float(c[i]), "index": i})

    p = argrelextrema(h, np.greater, order=5)[0]
    v = argrelextrema(l, np.less, order=5)[0]
    mss = "None"
    if len(p) and c[-1] > h[p[-1]]: mss = "Bullish MSS"
    elif len(v) and c[-1] < l[v[-1]]: mss = "Bearish MSS"
    return {"fvgs": fvgs[:3], "order_blocks": obs[:3], "mss": mss}

def detect_divergences(df: pd.DataFrame, indicator: str = "RSI"):
    c = df['close'].values
    if indicator == "RSI": ind = talib.RSI(c)
    elif indicator == "MACD": ind, _, _ = talib.MACD(c)
    else: return None
    v, p = argrelextrema(c, np.less, order=5)[0], argrelextrema(c, np.greater, order=5)[0]
    divs = {"bullish": False, "bearish": False}
    if len(v) >= 2 and len(ind) > max(v):
        if c[v[-1]] < c[v[-2]] and ind[v[-1]] > ind[v[-2]]: divs["bullish"] = True
    if len(p) >= 2 and len(ind) > max(p):
        if c[p[-1]] > c[p[-2]] and ind[p[-1]] < ind[p[-2]]: divs["bearish"] = True
    return divs

def calculate_volume_profile(df: pd.DataFrame):
    if df['volume'].sum() == 0: return None
    bins = 20
    counts, edges = np.histogram(df['close'], bins=bins, weights=df['volume'])
    idx = np.argmax(counts)
    return {"poc": float(edges[idx]), "vah": float(edges[min(idx+2, bins)]), "val": float(edges[max(idx-2, 0)])}

def get_indicator_status(df: pd.DataFrame, selected: Optional[List[str]] = None, history: bool = False):
    ta_df = df.copy()
    ta_df.columns = [c.capitalize() for c in ta_df.columns]
    try: ta_df = ta.add_all_ta_features(ta_df, open="Open", high="High", low="Low", close="Close", volume="Volume", fillna=False)
    except: pass

    cl = df['close'].values
    rsi = talib.RSI(cl)
    ema200 = talib.EMA(cl, timeperiod=min(len(cl), 200))
    scores = pd.Series(0.0, index=df.index)
    scores[rsi < 30] += 1
    scores[rsi > 70] -= 1
    scores[cl > ema200] += 0.5
    scores[cl <= ema200] -= 0.5
    df['signal'] = scores.apply(lambda s: "Buy" if s >= 1.5 else ("Sell" if s <= -1.5 else "Hold"))

    latest = ta_df.iloc[-1].to_dict()
    clean_latest = {clean_column_name(k): v for k, v in latest.items() if not selected or clean_column_name(k) in selected}

    res = {
        "current_price": float(cl[-1]),
        "summary": {"signal": df['signal'].iloc[-1], "trend": "Bullish" if cl[-1] > ema200[-1] else "Bearish", "session": get_sessions(df['timestamp'].iloc[-1])},
        "levels": detect_sr_levels(df),
        "smc": detect_smc_concepts(df),
        "divergences": {"rsi": detect_divergences(df, "RSI"), "macd": detect_divergences(df, "MACD")},
        "volume_profile": calculate_volume_profile(df)
    }
    if history: res["history"] = df.tail(500).to_dict(orient="records")
    else: res["indicators"] = clean_latest
    return res

async def fetch_market_data(provider: str, symbol: str, timeframe: str, exchange_id: str = "binance"):
    if provider == "crypto":
        ex_id = exchange_id if exchange_id else "binance"
        ex = getattr(ccxt, ex_id)()
        ohlcv = ex.fetch_ohlcv(symbol, timeframe=timeframe, limit=200)
        return pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    else:
        yf_map = {"15m": "15m", "1h": "1h", "4h": "1h", "1d": "1d"}
        data = yf.download(symbol if provider == "stock" else f"{symbol}=X", period="1y", interval=yf_map[timeframe], progress=False)
        if isinstance(data.columns, pd.MultiIndex): data.columns = data.columns.get_level_values(0)
        if timeframe == "4h": data = data.resample('4h').agg({'Open':'first', 'High':'max', 'Low':'min', 'Close':'last', 'Volume':'sum'}).dropna()
        df = data.tail(200).reset_index()
        df.columns = [str(c).lower() for c in df.columns]
        return df.rename(columns={'date': 'timestamp', 'datetime': 'timestamp'})

# --- Endpoints ---

@app.get("/indicators")
async def get_available_indicators():
    def fmt(cat):
        return [{"code": k, "full_name": v} for k, v in INDICATOR_METADATA.get(cat, {}).items()]
    return {"technical_indicators": fmt("technical_indicators"), "patterns": fmt("patterns"), "institutional": fmt("institutional_strategies")}

@app.post("/analyze/market", dependencies=[Depends(verify_rapidapi_key)])
async def analyze_market(request: MarketRequest):
    df = await fetch_market_data(request.provider, request.symbol, request.timeframe, request.exchange)
    return clean_dict(get_indicator_status(df, request.indicators, request.include_history))

@app.post("/analyze/upload", dependencies=[Depends(verify_rapidapi_key)])
async def analyze_upload(request: UploadRequest):
    df = pd.DataFrame([c.model_dump() for c in request.data[-2000:]])
    return clean_dict(get_indicator_status(df, request.indicators, request.include_history))

@app.post("/analyze/mtf", dependencies=[Depends(verify_rapidapi_key)])
async def analyze_mtf(request: MTFRequest):
    tfs = {}
    for tf in request.timeframes:
        df = await fetch_market_data(request.provider, request.symbol, tf, request.exchange)
        tfs[tf] = get_indicator_status(df, request.indicators)
    return clean_dict({"symbol": request.symbol, "timeframes": tfs})

@app.post("/analyze/correlation", dependencies=[Depends(verify_rapidapi_key)])
async def analyze_correlation(request: CorrelationRequest):
    series = {}
    for asset in request.assets:
        df = await fetch_market_data(request.provider, asset, request.timeframe)
        series[asset] = df['close']
    return clean_dict(pd.DataFrame(series).corr().to_dict())

@app.post("/analyze/heatmap", dependencies=[Depends(verify_rapidapi_key)])
async def analyze_heatmap(request: HeatmapRequest):
    async def get_m(a):
        try:
            df = await fetch_market_data(request.provider, a, "1d")
            if request.metric == "RSI": v = talib.RSI(df['close'].values)[-1]
            else: v = ((df['close'].iloc[-1] - df['close'].iloc[-2]) / df['close'].iloc[-2]) * 100
            return a, float(v)
        except: return a, None
    res = await asyncio.gather(*[get_m(a) for a in request.assets])
    return clean_dict(dict(res))

@app.post("/options/greeks")
async def options_greeks(req: OptionsRequest):
    S, K, T = req.underlying_price, req.strike, (datetime.strptime(req.expiry, "%Y-%m-%d") - datetime.now()).days / 365.0
    if T <= 0: T = 1/365.0
    r, sigma = req.risk_free_rate, req.volatility
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    if req.option_type == "call":
        delta, theta = norm.cdf(d1), (-S * norm.pdf(d1) * sigma / (2 * np.sqrt(T)) - r * K * np.exp(-r * T) * norm.cdf(d2)) / 365.0
    else:
        delta, theta = norm.cdf(d1) - 1, (-S * norm.pdf(d1) * sigma / (2 * np.sqrt(T)) + r * K * np.exp(-r * T) * norm.cdf(-d2)) / 365.0
    return clean_dict({"delta": delta, "gamma": norm.pdf(d1) / (S * sigma * np.sqrt(T)), "theta": theta, "vega": S * norm.pdf(d1) * np.sqrt(T) / 100.0})

@app.post("/confluence-score", dependencies=[Depends(verify_rapidapi_key)])
async def get_confluence_score(request: MarketRequest):
    df = await fetch_market_data(request.provider, request.symbol, request.timeframe, request.exchange)
    a = get_indicator_status(df)
    s, smc = a['summary'], a['smc']
    score = 0
    if s['signal'] == "Buy": score += 40
    elif s['signal'] == "Sell": score -= 40
    score += 20 if s['trend'] == "Bullish" else -20
    if smc['mss'] == "Bullish MSS": score += 20
    elif smc['mss'] == "Bearish MSS": score -= 20
    score = max(-100, min(100, score))
    return {"symbol": request.symbol, "confluence_score": score, "sentiment": "Strong Buy" if score > 50 else ("Buy" if score > 10 else ("Strong Sell" if score < -50 else ("Sell" if score < -10 else "Neutral")))}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
