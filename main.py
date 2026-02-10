import pandas as pd
import numpy as np
import talib
import ccxt
from binance.um_futures import UMFutures
from binance.error import ClientError
import yfinance as yf
import asyncio
import os
import io
from datetime import datetime
from typing import List, Optional, Literal, Dict, Any
from fastapi import FastAPI, HTTPException, Depends, Header, Query
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field
import diskcache
import json

# Import refactored logic
from lib.indicators import (
    INDICATOR_METADATA,
    clean_dict,
    get_indicator_results,
    get_indicator_results_sync
)

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

# --- Binance Client ---
binance_client = UMFutures()

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

# --- Data Fetching Helpers ---

async def fetch_data_binance(symbol, timeframe, limit=500):
    """Fetches OHLCV from Binance UMFutures."""
    try:
        s = symbol.upper().replace("/", "")
        if s.endswith("USD"): s = s.replace("USD", "USDT")
        if not (s.endswith("USDT") or s.endswith("BUSD")): s += "USDT"

        resp = await asyncio.to_thread(binance_client.klines, s, timeframe, limit=limit)
        if not resp:
            raise HTTPException(status_code=404, detail=f"No data found for {s}")

        df = pd.DataFrame(resp).iloc[:, :6]
        df.columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
        df = df.astype(float)
        return df
    except ClientError as error:
        raise HTTPException(status_code=500, detail=f"Binance Error: {error.error_message}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Data Fetch Error: {str(e)}")

async def fetch_data(provider, symbol, tf, ex_id="binance"):
    fetch_limit = 500
    cache_key = f"data_{provider}_{symbol}_{tf}_{ex_id}_{fetch_limit}"
    cached = cache.get(cache_key)
    if cached is not None: return pd.read_json(io.StringIO(cached))

    if provider == "crypto":
        try:
            df = await fetch_data_binance(symbol, tf, limit=fetch_limit)
        except Exception as e:
            ex = ccxt.kraken()
            s = symbol if "/" in symbol else f"{symbol}/USDT"
            ohlcv = await asyncio.to_thread(ex.fetch_ohlcv, s, timeframe=tf, limit=fetch_limit)
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    else:
        mapping = {"15m": "15m", "1h": "1h", "4h": "1h", "1d": "1d"}
        data = await asyncio.to_thread(yf.download, symbol if provider == "stock" else f"{symbol}=X", period="2y", interval=mapping[tf], progress=False)
        if isinstance(data.columns, pd.MultiIndex): data.columns = data.columns.get_level_values(0)
        if tf == "4h": data = data.resample('4h').agg({'Open':'first', 'High':'max', 'Low':'min', 'Close':'last', 'Volume':'sum'}).dropna()
        df = data.tail(fetch_limit).reset_index()
        df.columns = [str(c).lower() for c in df.columns]
        df = df.rename(columns={'date': 'timestamp', 'datetime': 'timestamp'})

    cache.set(cache_key, df.to_json(), expire=60)
    return df

# --- Endpoints ---

@app.get("/indicators")
async def list_indicators():
    ti, cp = INDICATOR_METADATA["technical_indicators"].copy(), INDICATOR_METADATA["candlestick_patterns"].copy()
    for f in talib.get_functions():
        if f.startswith("CDL"):
            if f not in cp: cp[f] = f.replace("CDL", "").replace("_", " ").title()
        elif f not in ti: ti[f] = f.replace("_", " ").title()
    def fmt(d): return [{"code": k, "full_name": v} for k, v in sorted(d.items())]
    return {
        "technical_indicators": fmt(ti),
        "candlestick_patterns": fmt(cp),
        "institutional_strategies": fmt(INDICATOR_METADATA["institutional_strategies"]),
        "price_action_patterns": fmt(INDICATOR_METADATA["price_action_patterns"]),
        "market_dynamics": fmt(INDICATOR_METADATA["market_dynamics"]),
        "custom_lux_algo": fmt(INDICATOR_METADATA["custom_lux_algo"]),
        "squeeze_momentum": fmt(INDICATOR_METADATA["squeeze_momentum"]),
        "trend_following": fmt(INDICATOR_METADATA["trend_following"])
    }

@app.post("/analyze/market", dependencies=[Depends(verify_rapidapi_key)])
async def analyze_market(req: MarketRequest):
    cache_key = f"analysis_{req.provider}_{req.symbol}_{req.timeframe}_{req.exchange}_{req.indicators}_{req.include_history}"
    cached = cache.get(cache_key)
    if cached: return cached

    df = await fetch_data(req.provider, req.symbol, req.timeframe, req.exchange)
    if len(df) < 30:
        raise HTTPException(status_code=400, detail="Not enough data points fetched. Minimum 30 required.")

    res = clean_dict(await get_indicator_results(df, req.indicators, req.include_history))
    cache.set(cache_key, res, expire=60)
    return res

@app.post("/analyze/upload", dependencies=[Depends(verify_rapidapi_key)])
async def analyze_upload(req: UploadRequest):
    if len(req.data) < 30:
        raise HTTPException(status_code=400, detail="Not enough candles. Minimum 30 required.")
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

    struct = a['institutional_strategies']['structure']
    if "Bullish" in struct['swing']: score += 20
    elif "Bearish" in struct['swing']: score -= 20

    if "Bullish" in struct['internal']: score += 10
    elif "Bearish" in struct['internal']: score -= 10

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
    from scipy.stats import norm
    T = (datetime.strptime(req.expiry, "%Y-%m-%d") - datetime.now()).total_seconds() / (365.0 * 24 * 3600)
    if T <= 0: T = 1e-5
    S, K, r, sigma = req.underlying_price, req.strike, req.risk_free_rate, req.volatility
    d1 = (np.log(S/K) + (r + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))
    d2 = d1 - sigma*np.sqrt(T)
    delta = norm.cdf(d1) if req.option_type == "call" else norm.cdf(d1)-1
    return clean_dict({"delta": delta, "gamma": norm.pdf(d1) / (S*sigma*np.sqrt(T))})

@app.get("/analyze/chart", response_class=HTMLResponse)
async def get_chart(
    provider: Literal["crypto", "stock", "forex"] = "crypto",
    symbol: str = "BTC/USD",
    timeframe: Literal["15m", "1h", "4h", "1d"] = "1d",
    exchange: str = "kraken",
    indicators: Optional[List[str]] = Query(None)
):
    if not indicators:
        indicators = ["EMA20", "EMA50", "EMA200", "RSI", "MACD", "SUPERTREND", "IMBA_TREND", "SQUEEZE_MOMENTUM"]

    category_map = {
        "SMC": ["SUPERTREND", "IMBA_TREND", "SQUEEZE_MOMENTUM", "LUX_ZSCORE"],
        "LUX": ["LUX_ZSCORE"],
        "TREND": ["SUPERTREND", "IMBA_TREND"],
        "SQUEEZE": ["SQUEEZE_MOMENTUM"]
    }
    target_indicators = [i.upper() for i in indicators]
    for ind in indicators:
        if ind.upper() in category_map:
            target_indicators.extend(category_map[ind.upper()])

    df = await fetch_data(provider, symbol, timeframe, exchange)
    analysis = await get_indicator_results(df, indicators, history=True)
    chart_data = analysis["history"][-200:]

    idx_to_time = {}
    for i, row in df.iterrows():
        ts = row["timestamp"]
        if isinstance(ts, (int, float, np.integer)): t = int(ts/1000)
        else: t = int(pd.to_datetime(ts).timestamp())
        idx_to_time[i] = t

    candles, indicator_series, markers = [], {}, []
    smc = analysis.get("institutional_strategies", {})
    price_patterns = analysis.get("price_action_patterns", [])
    dynamics = analysis.get("market_dynamics", {})

    for p in price_patterns:
        if p.get("index") in idx_to_time:
            markers.append({"time": idx_to_time[p["index"]], "position": "aboveBar", "color": "#f23645", "shape": "arrowDown", "text": p["pattern"]})

    for ob in smc.get("order_blocks", []):
        if ob.get("index") in idx_to_time:
            markers.append({"time": idx_to_time[ob["index"]], "position": "belowBar", "color": "#2158f3", "shape": "square", "text": "OB"})

    for fvg in smc.get("fair_value_gaps", []):
        if fvg.get("index") in idx_to_time:
            markers.append({"time": idx_to_time[fvg["index"]], "position": "belowBar", "color": "#00ff68", "shape": "circle", "text": "FVG"})

    price_avg = np.mean([r["close"] for r in chart_data])

    for row in chart_data:
        ts = row["timestamp"]
        if isinstance(ts, (int, float)): t = int(ts/1000)
        else: t = int(datetime.fromisoformat(str(ts).replace('Z', '+00:00')).timestamp())

        candles.append({"time": t, "open": row["open"], "high": row["high"], "low": row["low"], "close": row["close"]})

        for k, v in row.items():
            if k in ["open", "high", "low", "close", "volume", "timestamp", "signal"]: continue
            if not isinstance(v, (int, float, np.number)): continue

            if k.startswith("CDL"):
                if v != 0:
                    markers.append({
                        "time": t,
                        "position": "aboveBar" if v < 0 else "belowBar",
                        "color": "#e91e63" if v < 0 else "#9c27b0",
                        "shape": "arrowDown" if v < 0 else "arrowUp",
                        "text": k[3:]
                    })
                continue

            if k.upper() not in target_indicators: continue
            if k not in indicator_series: indicator_series[k] = []
            indicator_series[k].append({"time": t, "value": float(v)})

    sr_levels = dynamics.get("levels", [])
    js_logic = """
            const chartWidth = 1000;
            const mainChart = LightweightCharts.createChart(document.getElementById('main-chart'), {
                width: chartWidth, height: 500,
                layout: { backgroundColor: '#131722', textColor: '#d1d4dc' },
                grid: { vertLines: { color: '#1e222d' }, horzLines: { color: '#1e222d' } },
                timeScale: { borderColor: '#485c7b', timeVisible: true }
            });

            const candleSeries = mainChart.addCandlestickSeries({
                upColor: '#089981', downColor: '#f23645', borderVisible: false,
                wickUpColor: '#089981', wickDownColor: '#f23645'
            });

            candleSeries.setData(candles);
            candleSeries.setMarkers(markers);

            srLevels.forEach(lv => {
                candleSeries.createPriceLine({
                    price: lv.price, color: lv.type === 'Resistance' ? '#f23645' : '#089981',
                    lineWidth: 1, lineStyle: 2, axisLabelVisible: true, title: lv.type
                });
            });

            const oscCharts = [];
            Object.keys(indicatorSeriesData).forEach(name => {
                const data = indicatorSeriesData[name];
                if (data.length === 0) return;
                const valAvg = data.reduce((a,b) => a + b.value, 0) / data.length;
                const color = '#' + (Math.random().toString(16) + '000000').substring(2,8);
                const isOverlay = Math.abs(valAvg - priceAvg) / priceAvg < 0.5 ||
                              ["UPPER", "LOWER", "MID", "STOP", "TREND", "IMBA", "BANDS", "EMA", "SMA", "VWAP"].some(k => name.toUpperCase().includes(k));

                if (isOverlay) {
                    const line = mainChart.addLineSeries({ title: name, lineWidth: 1, color: color });
                    line.setData(data);
                } else {
                    const container = document.createElement('div');
                    container.className = 'osc-container';
                    document.getElementById('oscillators').appendChild(container);
                    const oscChart = LightweightCharts.createChart(container, {
                        width: chartWidth, height: 150,
                        layout: { backgroundColor: '#131722', textColor: '#d1d4dc' },
                        grid: { vertLines: { color: '#1e222d' }, horzLines: { color: '#1e222d' } },
                        timeScale: { visible: false }
                    });
                    if (name.includes('HIST') || name.includes('MOMENTUM')) {
                        const hist = oscChart.addHistogramSeries({ title: name, color: color, priceFormat: { type: 'volume' } });
                        hist.setData(data.map(d => ({ ...d, color: d.value >= 0 ? '#26a69a' : '#ef5350' })));
                    } else {
                        const line = oscChart.addLineSeries({ title: name, lineWidth: 1, color: color });
                        line.setData(data);
                    }
                    oscCharts.push(oscChart);
                }
            });
            mainChart.timeScale().subscribeVisibleTimeRangeChange(range => {
                oscCharts.forEach(c => c.timeScale().setVisibleRange(range));
            });
    """

    html_template = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>SYMBOL_PLACEHOLDER - TA Visualizer</title>
        <script src="https://unpkg.com/lightweight-charts@4.0.0/dist/lightweight-charts.standalone.production.js"></script>
        <style>
            body { margin: 0; padding: 20px; background: #131722; color: white; font-family: sans-serif; }
            .chart-container { width: 100%; height: 500px; margin-bottom: 10px; }
            .osc-container { width: 100%; height: 150px; margin-bottom: 10px; }
            .controls { margin-bottom: 10px; border-bottom: 1px solid #2B2B43; padding-bottom: 10px; }
        </style>
    </head>
    <body>
        <div class="controls">
            <h2 style="margin:0;">SYMBOL_PLACEHOLDER (TIMEFRAME_PLACEHOLDER)</h2>
            <p style="color: #878b94; margin: 5px 0;">TA-as-a-Service Visualizer • Multi-Pane Layout</p>
        </div>
        <div id="main-chart" style="width: 100%; height: 500px;"></div>
        <div id="oscillators"></div>
        <script>
            window.addEventListener('DOMContentLoaded', () => {
                const candles = """ + json.dumps(candles) + """;
                const indicatorSeriesData = """ + json.dumps(indicator_series) + """;
                const markers = """ + json.dumps(markers) + """;
                const srLevels = """ + json.dumps(sr_levels) + """;
                const priceAvg = """ + str(price_avg) + """;
                """ + js_logic + """
            });
        </script>
    </body>
    </html>
    """

    html_content = html_template.replace("SYMBOL_PLACEHOLDER", symbol).replace("TIMEFRAME_PLACEHOLDER", timeframe)
    return html_content

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
