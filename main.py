import pandas as pd
import numpy as np
import talib
import ccxt
import yfinance as yf
import ta
import asyncio
import os
import time
from datetime import datetime
from typing import List, Optional, Literal, Dict, Any
from scipy.signal import argrelextrema
from scipy.stats import norm
from fastapi import FastAPI, HTTPException, Depends, Header
from pydantic import BaseModel, Field
import diskcache

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
    timestamp: Optional[Any] = None
    open: float
    high: float
    low: float
    close: float
    volume: float

class UploadRequest(BaseModel):
    data: List[Candle] = Field(..., description="List of OHLCV candles")
    indicators: Optional[List[str]] = None
    include_history: bool = False

class MarketRequest(BaseModel):
    provider: Literal["crypto", "stock", "forex"]
    symbol: str
    timeframe: Literal["15m", "1h", "4h", "1d"]
    exchange: Optional[str] = "kraken"
    indicators: Optional[List[str]] = None
    include_history: bool = False

class MTFRequest(BaseModel):
    provider: Literal["crypto", "stock", "forex"]
    symbol: str
    timeframes: List[Literal["15m", "1h", "4h", "1d"]]
    exchange: Optional[str] = "kraken"
    indicators: Optional[List[str]] = None

class CorrelationRequest(BaseModel):
    assets: List[str]
    provider: Literal["crypto", "stock", "forex"] = "crypto"
    timeframe: str = "1d"

class HeatmapRequest(BaseModel):
    assets: List[str]
    provider: Literal["crypto", "stock", "forex"] = "crypto"
    metric: str = "RSI"

class OptionsRequest(BaseModel):
    underlying_price: float
    strike: float
    expiry: str
    volatility: float
    risk_free_rate: float = 0.05
    option_type: Literal["call", "put"] = "call"

# --- Metadata ---

INDICATOR_METADATA = {
    "technical_indicators": {
        "SMA": "Simple Moving Average", "EMA": "Exponential Moving Average", "WMA": "Weighted Moving Average",
        "RSI": "Relative Strength Index", "ADX": "Average Directional Index", "ATR": "Average True Range",
        "MACD": "Moving Average Convergence Divergence", "BBANDS": "Bollinger Bands",
        "VWAP": "Volume Weighted Average Price", "OBV": "On Balance Volume", "CMF": "Chaikin Money Flow",
        "EMA20": "20-Period EMA", "EMA50": "50-Period EMA", "EMA200": "200-Period EMA", "SMA20": "20-Period SMA"
    },
    "candlestick_patterns": {f: f.replace("CDL", "").replace("_", " ").title() for f in talib.get_functions() if f.startswith('CDL')},
    "institutional_strategies": {
        "Fair_Value_Gap": "FVG (Liquidity Imbalance)",
        "Order_Block": "Institutional Support/Demand Zone",
        "Market_Structure_Shift": "Trend Character Change (MSS/BOS)"
    },
    "price_action_patterns": {
        "Head_and_Shoulders": "Classic Reversal Structure",
        "Double_Top": "Bearish Reversal",
        "Double_Bottom": "Bullish Reversal",
        "Symmetrical_Triangle": "Consolidation Pattern",
        "Descending_Triangle": "Bearish Structure",
        "Ascending_Triangle": "Bullish Structure",
        "RSI_Divergence": "Momentum vs Price Divergence",
        "MACD_Divergence": "Trend vs Price Divergence"
    },
    "market_dynamics": {
        "Sessions": "Global Market Hours (Tokyo/London/NY)",
        "SR_Levels": "Horizontal Support & Resistance Levels",
        "Liquidity": "Buy Side & Sell Side Liquidity Pools",
        "Volume_Profile": "Price Distribution Analysis"
    }
}

# --- Core Logic Helpers ---

def clean_dict(d):
    if isinstance(d, dict): return {k: clean_dict(v) for k, v in d.items()}
    if isinstance(d, list): return [clean_dict(v) for v in d]
    if isinstance(d, (float, np.float64, np.float32)):
        return None if np.isnan(d) or np.isinf(d) else float(d)
    return d

def clean_name(name: str) -> str:
    for p in ["talib_", "trend_", "momentum_", "volatility_", "volume_", "others_"]:
        if name.startswith(p):
            name = name[len(p):]
            break
    mapping = {
        "bbh": "Bollinger_High", "bbl": "Bollinger_Low", "bbm": "Bollinger_Mid",
        "macd_signal": "MACD_Signal", "macd_diff": "MACD_Hist",
        "ema_20": "EMA20", "ema_50": "EMA50", "ema_200": "EMA200", "sma_20": "SMA20"
    }
    return mapping.get(name.lower(), name.upper())

def get_sessions(ts: Optional[Any]) -> List[str]:
    if ts is None: return []
    try:
        if isinstance(ts, (int, float)) or str(ts).isdigit(): dt = datetime.fromtimestamp(int(ts)/1000.0)
        else: dt = datetime.fromisoformat(str(ts).replace('Z', '+00:00'))
        h = dt.hour
        s = []
        if 0 <= h < 9: s.append("Tokyo")
        if 8 <= h < 17: s.append("London")
        if 13 <= h < 22: s.append("New York")
        return s
    except: return []

def detect_smc(df):
    h, l, c = df['high'].values, df['low'].values, df['close'].values
    fvgs, obs = [], []
    for i in range(len(df)-1, len(df)-20, -1):
        if i < 2: break
        if l[i] > h[i-2]: fvgs.append({"type": "Bullish FVG", "top": float(l[i]), "bottom": float(h[i-2]), "index": i-1})
        elif h[i] < l[i-2]: fvgs.append({"type": "Bearish FVG", "top": float(l[i-2]), "bottom": float(h[i]), "index": i-1})
    atr = talib.ATR(h, l, c)
    for i in range(len(df)-2, len(df)-20, -1):
        move = c[i+1] - c[i]
        if i < len(atr) and abs(move) > 2 * (atr[i] if not np.isnan(atr[i]) else 1):
            obs.append({"type": "Bullish OB" if move > 0 else "Bearish OB", "price": float(c[i]), "index": i})
    p = argrelextrema(h, np.greater, order=5)[0]
    v = argrelextrema(l, np.less, order=5)[0]
    mss = "None"
    if len(p) and c[-1] > h[p[-1]]: mss = "Bullish MSS"
    elif len(v) and c[-1] < l[v[-1]]: mss = "Bearish MSS"
    return {"fair_value_gaps": fvgs[:3], "order_blocks": obs[:3], "market_structure_shift": mss}

def detect_sr_levels(df: pd.DataFrame):
    h, l, c = df['high'].values, df['low'].values, df['close'].values
    p = argrelextrema(h, np.greater, order=10)[0]
    v = argrelextrema(l, np.less, order=10)[0]
    pivots = np.sort(np.concatenate([h[p], l[v]]))
    if not len(pivots): return []
    groups, curr = [], [pivots[0]]
    for i in range(1, len(pivots)):
        if (pivots[i] - np.mean(curr)) / np.mean(curr) < 0.01: curr.append(pivots[i])
        else:
            groups.append(np.mean(curr))
            curr = [pivots[i]]
    groups.append(np.mean(curr))
    return [{"type": "Resistance" if lv > c[-1] else "Support", "price": float(lv)} for lv in groups]

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

def get_indicator_results_sync(df, selected=None, history=False):
    op, hi, lo, cl, vo = df['open'].values, df['high'].values, df['low'].values, df['close'].values, df['volume'].values
    ta_df = df.copy()
    ta_df.columns = [c.capitalize() for c in ta_df.columns]
    try: ta_df = ta.add_all_ta_features(ta_df, open="Open", high="High", low="Low", close="Close", volume="Volume", fillna=False)
    except: pass

    talib_res = {}
    for f in talib.get_functions():
        try:
            func = getattr(talib, f)
            if f.startswith('CDL'): res = func(op, hi, lo, cl)
            elif f in ['AD', 'ADOSC', 'OBV', 'MFI']: res = func(hi, lo, cl, vo)
            elif f in ['ADX', 'ADXR', 'ATR', 'NATR', 'WILLR', 'CCI', 'DX', 'MINUS_DI', 'MINUS_DM', 'PLUS_DI', 'PLUS_DM', 'ULTOSC', 'MEDPRICE', 'TYPPRICE', 'WCLPRICE', 'SAR', 'TRANGE']: res = func(hi, lo, cl)
            elif f in ['BETA', 'CORREL']: res = func(hi, lo)
            elif f == 'AROON':
                d, u = func(hi, lo)
                talib_res['AROON_DOWN'], talib_res['AROON_UP'] = d, u
                continue
            elif f in ['MACD', 'MACDEXT', 'MACDFIX']:
                m, s, h_ = func(cl)
                talib_res[f], talib_res[f+'_SIGNAL'], talib_res[f+'_HIST'] = m, s, h_
                continue
            elif f == 'BBANDS':
                u, m, l_ = func(cl)
                talib_res['BOLLINGER_HIGH'], talib_res['BOLLINGER_MID'], talib_res['BOLLINGER_LOW'] = u, m, l_
                continue
            elif f in ['STOCH', 'STOCHF']:
                k, d = func(hi, lo, cl)
                talib_res[f+'_K'], talib_res[f+'_D'] = k, d
                continue
            else: res = func(cl)
            if isinstance(res, tuple):
                for i, r in enumerate(res): talib_res[f"{f}_{i}"] = r
            else: talib_res[f] = res
        except: pass

    all_raw = pd.DataFrame(index=df.index)
    for k, v in talib_res.items():
        clean_k = clean_name(k)
        if clean_k not in all_raw.columns: all_raw[clean_k] = v
    for c in ta_df.columns:
        clean_c = clean_name(c)
        if clean_c not in all_raw.columns: all_raw[clean_c] = ta_df[c]

    rsi = all_raw.get('RSI', pd.Series([50]*len(df), index=df.index))
    ema200 = talib.EMA(cl, timeperiod=min(len(cl), 200))
    macd_s, sig_s = all_raw.get('MACD', pd.Series([0]*len(df), index=df.index)), all_raw.get('MACD_SIGNAL', pd.Series([0]*len(df), index=df.index))

    scores = pd.Series(0.0, index=df.index)
    scores[rsi < 30] += 1
    scores[rsi > 70] -= 1
    scores[cl > ema200] += 0.5
    scores[cl <= ema200] -= 0.5
    if len(df) > 1:
        scores[(macd_s > sig_s) & (macd_s.shift(1) <= sig_s.shift(1))] += 1
        scores[(macd_s < sig_s) & (macd_s.shift(1) >= sig_s.shift(1))] -= 1
    df['signal'] = scores.apply(lambda s: "Buy" if s >= 1.5 else ("Sell" if s <= -1.5 else "Hold"))

    smc = detect_smc(df)
    res = {
        "current_price": float(cl[-1]),
        "summary": {"signal": df['signal'].iloc[-1], "trend": "Bullish" if cl[-1] > ema200[-1] else "Bearish", "session": get_sessions(df['timestamp'].iloc[-1])},
        "institutional_strategies": smc,
        "market_dynamics": {
            "levels": detect_sr_levels(df),
            "volume_profile": calculate_volume_profile(df)
        },
        "divergences": {"rsi": detect_divergences(df, "RSI"), "macd": detect_divergences(df, "MACD")}
    }

    latest_data = all_raw.iloc[-1].to_dict()
    if selected:
        sel_up = [s.upper() for s in selected]
        res["indicators"] = {k: v for k, v in latest_data.items() if k.upper() in sel_up or k in sel_up}
    else: res["indicators"] = latest_data

    if history:
        h_df = df.copy()
        for col in all_raw.columns:
            if col not in h_df.columns: h_df[col] = all_raw[col]
        res["history"] = h_df.tail(1000).to_dict(orient="records")
    return res

async def get_indicator_results(df, selected=None, history=False):
    return await asyncio.to_thread(get_indicator_results_sync, df, selected, history)

async def fetch_data(provider, symbol, tf, ex_id="kraken"):
    if provider == "crypto":
        ex = getattr(ccxt, ex_id if ex_id else "kraken")()
        ohlcv = await asyncio.to_thread(ex.fetch_ohlcv, symbol, timeframe=tf, limit=200)
        return pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    else:
        mapping = {"15m": "15m", "1h": "1h", "4h": "1h", "1d": "1d"}
        data = await asyncio.to_thread(yf.download, symbol if provider == "stock" else f"{symbol}=X", period="1y", interval=mapping[tf], progress=False)
        if isinstance(data.columns, pd.MultiIndex): data.columns = data.columns.get_level_values(0)
        if tf == "4h": data = data.resample('4h').agg({'Open':'first', 'High':'max', 'Low':'min', 'Close':'last', 'Volume':'sum'}).dropna()
        df = data.tail(200).reset_index()
        df.columns = [str(c).lower() for c in df.columns]
        return df.rename(columns={'date': 'timestamp', 'datetime': 'timestamp'})

# --- Endpoints ---

@app.get("/indicators")
async def list_indicators():
    ti, cp = INDICATOR_METADATA["technical_indicators"].copy(), INDICATOR_METADATA["candlestick_patterns"].copy()
    for f in talib.get_functions():
        if f.startswith("CDL"):
            if f not in cp: cp[f] = f.replace("CDL", "").replace("_", " ").title()
        elif f not in ti: ti[f] = f.replace("_", " ").title()
    def fmt(d): return [{"code": k, "full_name": v} for k, v in sorted(d.items())]
    return {"technical_indicators": fmt(ti), "candlestick_patterns": fmt(cp), "institutional_strategies": fmt(INDICATOR_METADATA["institutional_strategies"]), "price_action_patterns": fmt(INDICATOR_METADATA["price_action_patterns"]), "market_dynamics": fmt(INDICATOR_METADATA["market_dynamics"])}

@app.post("/analyze/market", dependencies=[Depends(verify_rapidapi_key)])
async def analyze_market(req: MarketRequest):
    df = await fetch_data(req.provider, req.symbol, req.timeframe, req.exchange)
    return clean_dict(await get_indicator_results(df, req.indicators, req.include_history))

@app.post("/analyze/upload", dependencies=[Depends(verify_rapidapi_key)])
async def analyze_upload(req: UploadRequest):
    df = pd.DataFrame([c.model_dump() for c in req.data[-2000:]])
    return clean_dict(await get_indicator_results(df, req.indicators, req.include_history))

@app.post("/analyze/mtf", dependencies=[Depends(verify_rapidapi_key)])
async def analyze_mtf(req: MTFRequest):
    tasks = [fetch_data(req.provider, req.symbol, tf, req.exchange) for tf in req.timeframes]
    dfs = await asyncio.gather(*tasks)
    results = {}
    for tf, df in zip(req.timeframes, dfs):
        results[tf] = await get_indicator_results(df, req.indicators)
    return clean_dict({"symbol": req.symbol, "timeframes": results})

@app.post("/analyze/correlation", dependencies=[Depends(verify_rapidapi_key)])
async def analyze_correlation(req: CorrelationRequest):
    series = {}
    for asset in req.assets:
        try:
            df = await fetch_data(req.provider, asset, req.timeframe)
            series[asset] = df['close']
        except: pass
    if not series: raise HTTPException(status_code=400, detail="Could not fetch data.")
    return clean_dict(pd.DataFrame(series).corr().to_dict())

@app.post("/analyze/heatmap", dependencies=[Depends(verify_rapidapi_key)])
async def analyze_heatmap(req: HeatmapRequest):
    async def get_m(a):
        try:
            df = await fetch_data(req.provider, a, "1d")
            v = talib.RSI(df['close'].values)[-1] if req.metric == "RSI" else ((df['close'].iloc[-1]-df['close'].iloc[-2])/df['close'].iloc[-2])*100
            return a, float(v)
        except: return a, None
    res = await asyncio.gather(*[get_m(a) for a in req.assets])
    return clean_dict(dict(res))

@app.post("/confluence-score", dependencies=[Depends(verify_rapidapi_key)])
async def confluence(req: MarketRequest):
    df = await fetch_data(req.provider, req.symbol, req.timeframe, req.exchange)
    a = await get_indicator_results(df)
    score = 0
    if a['summary']['signal'] == "Buy": score += 40
    elif a['summary']['signal'] == "Sell": score -= 40
    score += 20 if a['summary']['trend'] == "Bullish" else -20
    if a['institutional_strategies']['market_structure_shift'] == "Bullish MSS": score += 20
    elif a['institutional_strategies']['market_structure_shift'] == "Bearish MSS": score -= 20
    score = max(-100, min(100, score))
    return clean_dict({"symbol": req.symbol, "score": score, "sentiment": "Strong Buy" if score > 50 else "Buy" if score > 10 else "Strong Sell" if score < -50 else "Sell" if score < -10 else "Neutral"})

@app.post("/is-trend-bullish", dependencies=[Depends(verify_rapidapi_key)])
async def is_bullish(req: MarketRequest):
    df = await fetch_data(req.provider, req.symbol, req.timeframe, req.exchange)
    ema200 = talib.EMA(df['close'].values, timeperiod=min(len(df), 200))[-1]
    return {"bullish": bool(df['close'].iloc[-1] > ema200)}

@app.post("/scan-patterns", dependencies=[Depends(verify_rapidapi_key)])
async def scan(req: MarketRequest):
    df = await fetch_data(req.provider, req.symbol, req.timeframe, req.exchange)
    op, hi, lo, cl = df['open'].values, df['high'].values, df['low'].values, df['close'].values
    found = []
    for f in [f for f in talib.get_functions() if f.startswith('CDL')]:
        res = getattr(talib, f)(op, hi, lo, cl)
        for i, v in enumerate(res):
            if v != 0: found.append({"index": i, "pattern": f, "sentiment": "Bullish" if v > 0 else "Bearish"})
    return {"patterns": found}

@app.post("/options/greeks")
async def greeks(req: OptionsRequest):
    T = (datetime.strptime(req.expiry, "%Y-%m-%d") - datetime.now()).total_seconds() / (365.0 * 24 * 3600)
    if T <= 0: T = 1e-5
    S, K, r, sigma = req.underlying_price, req.strike, req.risk_free_rate, req.volatility
    d1 = (np.log(S/K) + (r + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))
    d2 = d1 - sigma*np.sqrt(T)
    delta = norm.cdf(d1) if req.option_type == "call" else norm.cdf(d1)-1
    return clean_dict({"delta": delta, "gamma": norm.pdf(d1) / (S*sigma*np.sqrt(T))})

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
