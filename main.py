import pandas as pd
import numpy as np
import talib
from binance.um_futures import UMFutures
from binance.error import ClientError
import yfinance as yf
import asyncio
import os
import io
import orjson
import html
from datetime import datetime
from typing import List, Optional, Literal, Dict, Any
from fastapi import FastAPI, HTTPException, Depends, Header, Query, Request, Response
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.gzip import GZipMiddleware
from pydantic import BaseModel, Field
import diskcache
import redis.asyncio as redis
from fastapi_limiter import FastAPILimiter
from fastapi_limiter.depends import RateLimiter
from sqlalchemy.future import select

# Import refactored logic
from lib.indicators import (
    INDICATOR_METADATA,
    clean_dict,
    get_indicator_results,
    calculate_portfolio_metrics
)
from lib.database import init_db, get_db, APIRequest, User, Alert
from lib.auth import get_current_user, get_tier_limit

# --- Models ---

class Candle(BaseModel):
    timestamp: Optional[Any] = None
    open: float
    high: float
    low: float
    close: float
    volume: float

class UploadRequest(BaseModel):
    data: List[Candle]
    indicators: Optional[List[str]] = None
    include_history: bool = False

class MarketRequest(BaseModel):
    provider: Literal["crypto", "stock", "forex"]
    symbol: str
    timeframe: Literal["15m", "1h", "4h", "1d"]
    indicators: Optional[List[str]] = None
    include_history: bool = False
    exchange: Optional[str] = "binance"

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

class PortfolioRequest(BaseModel):
    assets: List[str]
    provider: Literal["crypto", "stock", "forex"] = "crypto"

class AlertRequest(BaseModel):
    symbol: str
    condition: Dict[str, Any]
    webhook_url: str

class BacktestRequest(BaseModel):
    symbol: str
    strategy: Dict[str, Any]
    period: str = "1y"

# --- App Setup ---

app = FastAPI(title="Pro-Trader Ultimate TA-as-a-Service API")
app.add_middleware(GZipMiddleware, minimum_size=1000)

class CustomORJSONResponse(JSONResponse):
    def render(self, content: Any) -> bytes:
        return orjson.dumps(content, option=orjson.OPT_NON_STR_KEYS | orjson.OPT_SERIALIZE_NUMPY)

@app.middleware("http")
async def error_handling_middleware(request: Request, call_next):
    try:
        return await call_next(request)
    except Exception as e:
        return CustomORJSONResponse(status_code=500, content={"error": {"message": str(e)}})

@app.on_event("startup")
async def startup():
    await init_db()
    try:
        r = redis.from_url(os.getenv("REDIS_URL", "redis://redis:6379/0"), encoding="utf-8", decode_responses=True)
        await FastAPILimiter.init(r)
    except:
        print("Redis connection failed. Rate limiting might be disabled.")

cache = diskcache.Cache("./cache")
binance_client = UMFutures()

# --- Helpers ---

async def tiered_rate_limit(request: Request, response: Response, user: User = Depends(get_current_user)):
    limit = get_tier_limit(user.tier)
    try:
        limiter = RateLimiter(times=limit["rpm"], seconds=60, identifier=lambda r: user.id)
        await limiter(request, response)
    except: pass # Bypass if redis fails

async def fetch_data_binance(symbol, timeframe, limit=500):
    try:
        s = symbol.upper().replace("/", "")
        if s.endswith("USD"): s = s.replace("USD", "USDT")
        if not (s.endswith("USDT") or s.endswith("BUSD")): s += "USDT"
        resp = await asyncio.to_thread(binance_client.klines, s, timeframe, limit=limit)
        df = pd.DataFrame(resp).iloc[:, :6]
        df.columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
        return df.astype(float)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Binance Error: {str(e)}")

async def fetch_data(provider, symbol, tf):
    fetch_limit = 500
    cache_key = f"data_{provider}_{symbol}_{tf}"
    cached = cache.get(cache_key)
    if cached is not None: return pd.read_json(io.StringIO(cached))

    if provider == "crypto":
        df = await fetch_data_binance(symbol, tf, limit=fetch_limit)
    else:
        mapping = {"15m": "15m", "1h": "1h", "4h": "1h", "1d": "1d"}
        s = symbol if provider == "stock" else f"{symbol}=X"
        data = await asyncio.to_thread(yf.download, s, period="2y", interval=mapping[tf], progress=False)
        if isinstance(data.columns, pd.MultiIndex): data.columns = data.columns.get_level_values(0)
        if tf == "4h": data = data.resample('4h').agg({'Open':'first', 'High':'max', 'Low':'min', 'Close':'last', 'Volume':'sum'}).dropna()
        df = data.tail(fetch_limit).reset_index()
        df.columns = [str(c).lower() for c in df.columns]
        df = df.rename(columns={'date': 'timestamp', 'datetime': 'timestamp'})

    cache.set(cache_key, df.to_json(), expire=60)
    return df

# --- Routes ---

@app.get("/health")
async def health():
    return {"status": "healthy", "timestamp": datetime.utcnow()}

@app.get("/indicators")
async def list_indicators():
    return clean_dict({k: [{"code": c, "name": n} for c, n in v.items()] for k, v in INDICATOR_METADATA.items()})

@app.post("/analyze/market", response_class=CustomORJSONResponse, dependencies=[Depends(tiered_rate_limit)])
async def analyze_market(req: MarketRequest, user: User = Depends(get_current_user)):
    df = await fetch_data(req.provider, req.symbol, req.timeframe)
    if len(df) < 30: raise HTTPException(status_code=400, detail="Insufficient data.")
    res = clean_dict(await get_indicator_results(df, req.indicators, req.include_history))
    return res

@app.post("/analyze/upload", dependencies=[Depends(get_current_user)])
async def analyze_upload(req: UploadRequest):
    if len(req.data) < 30: raise HTTPException(status_code=400, detail="Minimum 30 candles.")
    df = pd.DataFrame([c.model_dump() for c in req.data[-2000:]])
    return clean_dict(await get_indicator_results(df, req.indicators, req.include_history))

@app.post("/analyze/mtf", dependencies=[Depends(get_current_user)])
async def analyze_mtf(req: MTFRequest):
    tasks = [fetch_data(req.provider, req.symbol, tf) for tf in req.timeframes]
    dfs = await asyncio.gather(*tasks)
    results = {tf: await get_indicator_results(df, req.indicators) for tf, df in zip(req.timeframes, dfs)}
    return clean_dict({"symbol": req.symbol, "timeframes": results})

@app.post("/analyze/correlation", dependencies=[Depends(get_current_user)])
async def analyze_correlation(req: CorrelationRequest):
    series = {}
    for asset in req.assets:
        try:
            df = await fetch_data(req.provider, asset, req.timeframe)
            series[asset] = df['close']
        except: pass
    if not series: raise HTTPException(status_code=400, detail="Could not fetch data.")
    return clean_dict(pd.DataFrame(series).corr().to_dict())

@app.post("/analyze/heatmap", dependencies=[Depends(get_current_user)])
async def analyze_heatmap(req: HeatmapRequest):
    async def get_m(a):
        try:
            df = await fetch_data(req.provider, a, "1d")
            v = talib.RSI(df['close'].values)[-1] if req.metric == "RSI" else ((df['close'].iloc[-1]-df['close'].iloc[-2])/df['close'].iloc[-2])*100
            return a, float(v)
        except: return a, None
    res = await asyncio.gather(*[get_m(a) for a in req.assets])
    return clean_dict(dict(res))

@app.post("/confluence-score", dependencies=[Depends(get_current_user)])
async def confluence(req: MarketRequest):
    df = await fetch_data(req.provider, req.symbol, req.timeframe)
    a = await get_indicator_results(df)
    score = 0
    if a['summary']['trend'] == "Bullish": score += 20
    else: score -= 20
    # Simplified score based on summary signal
    ms = a.get('institutional_strategies', {}).get('market_structure', {})
    if ms.get('mss', '').startswith('Bullish') or ms.get('bos', '').startswith('Bullish'): score += 30
    elif ms.get('mss', '').startswith('Bearish') or ms.get('bos', '').startswith('Bearish'): score -= 30
    score = max(-100, min(100, score))
    return {"symbol": req.symbol, "score": score, "sentiment": "Strong Buy" if score > 50 else "Buy" if score > 10 else "Strong Sell" if score < -50 else "Sell" if score < -10 else "Neutral"}

@app.post("/is-trend-bullish", dependencies=[Depends(get_current_user)])
async def is_bullish(req: MarketRequest):
    df = await fetch_data(req.provider, req.symbol, req.timeframe)
    ema200 = talib.EMA(df['close'].values, timeperiod=min(len(df), 200))[-1]
    return {"bullish": bool(df['close'].iloc[-1] > ema200)}

@app.post("/scan-patterns", dependencies=[Depends(get_current_user)])
async def scan(req: MarketRequest):
    df = await fetch_data(req.provider, req.symbol, req.timeframe)
    op, hi, lo, cl = df['open'].values, df['high'].values, df['low'].values, df['close'].values
    found = []
    for f in [f for f in talib.get_functions() if f.startswith('CDL')]:
        res = getattr(talib, f)(op, hi, lo, cl)
        for i, v in enumerate(res):
            if v != 0: found.append({"index": i, "pattern": f, "sentiment": "Bullish" if v > 0 else "Bearish"})
    return {"patterns": found}

@app.post("/options/greeks")
async def greeks(req: OptionsRequest):
    from scipy.stats import norm
    T = (datetime.strptime(req.expiry, "%Y-%m-%d") - datetime.now()).total_seconds() / (365.0 * 24 * 3600)
    if T <= 0: T = 1e-5
    S, K, r, sigma = req.underlying_price, req.strike, req.risk_free_rate, req.volatility
    d1 = (np.log(S/K) + (r + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))
    delta = norm.cdf(d1) if req.option_type == "call" else norm.cdf(d1)-1
    return {"delta": float(delta)}

@app.post("/portfolio/analyze")
async def analyze_portfolio(req: PortfolioRequest):
    series = {a: (await fetch_data(req.provider, a, "1d"))['close'] for a in req.assets}
    return clean_dict(calculate_portfolio_metrics(series))

@app.post("/backtest", dependencies=[Depends(get_current_user)])
async def backtest(req: BacktestRequest):
    df = await fetch_data("crypto", req.symbol, "1d")
    from lib.indicators import backtest_strategy
    res = backtest_strategy(df)
    return {"symbol": req.symbol, "results": res}

@app.post("/alerts/create")
async def create_alert(req: AlertRequest, user: User = Depends(get_current_user), db=Depends(get_db)):
    alert = Alert(user_id=user.id, symbol=req.symbol, condition=req.condition, webhook_url=req.webhook_url)
    db.add(alert); await db.commit()
    return {"id": alert.id}

@app.get("/analyze/chart", response_class=HTMLResponse)
async def get_chart(provider: str = "crypto", symbol: str = "BTC/USDT", timeframe: str = "1d", indicators: Optional[str] = Query(None)):
    ind_list = indicators.split(",") if indicators else None
    df = await fetch_data(provider, symbol, timeframe)
    analysis = await get_indicator_results(df, ind_list, history=True)
    h = analysis["history"][-200:]
    def to_t(ts): return int(ts/1000) if isinstance(ts, (int, float)) else int(pd.to_datetime(ts).timestamp())
    candles = [{"time": to_t(r["timestamp"]), "open": r["open"], "high": r["high"], "low": r["low"], "close": r["close"]} for r in h]
    indicator_series = {}
    for r in h:
        t = to_t(r["timestamp"])
        for k, v in r.items():
            if k in ["open", "high", "low", "close", "volume", "timestamp", "signal"]: continue
            if not isinstance(v, (int, float, np.number)): continue
            if k not in indicator_series: indicator_series[k] = []
            indicator_series[k].append({"time": t, "value": float(v)})

    safe_symbol = html.escape(symbol)
    html_content = f"""<html><head><script src="https://unpkg.com/lightweight-charts@4.0.0/dist/lightweight-charts.standalone.production.js"></script></head>
    <body style="background:#131722;color:white"><h2>{safe_symbol}</h2><div id="c" style="width:1000px;height:600px"></div><div id="oscillators"></div><script>
    const chart = LightweightCharts.createChart(document.getElementById('c'), {width:1000, height:600, layout:{background:{color:'#131722'},textColor:'#d1d4dc'}});
    const cs = chart.addCandlestickSeries(); cs.setData("""+orjson.dumps(candles).decode()+""");
    const indData = """+orjson.dumps(indicator_series).decode()+""";
    const requested = """ + orjson.dumps(ind_list).decode() + """;
    Object.keys(indData).forEach(k => {
        if (requested && !requested.includes(k) && !k.toUpperCase().includes("EMA") && !k.toUpperCase().includes("SMA")) return;
        const data = indData[k];
        if (!data || data.length === 0) return;
        const valAvg = data.reduce((a,b) => a + (b.value || 0), 0) / data.length;
        const price = candles[candles.length - 1].close;
        const isOverlay = Math.abs(valAvg - price) / price < 0.5;

        if (isOverlay) {
            const s = chart.addLineSeries({title:k, color: '#' + Math.floor(Math.random()*16777215).toString(16), lineWidth: 2});
            s.setData(data.filter(d => d.value !== null && !isNaN(d.value)));
        } else {
            const container = document.createElement('div');
            container.style.width = '1000px'; container.style.height = '150px';
            container.style.border = '1px solid #2B2B43';
            document.getElementById('oscillators').appendChild(container);
            const oscChart = LightweightCharts.createChart(container, {width:1000, height:150, layout:{background:{color:'#131722'},textColor:'#d1d4dc'}, grid:{vertLines:{visible:false},horzLines:{visible:false}}});
            const s = oscChart.addLineSeries({title:k, color: '#2962FF'});
            s.setData(data.filter(d => d.value !== null && !isNaN(d.value)));
        }
    });
    </script></body></html>"""
    return html_content

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
