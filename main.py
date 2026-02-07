import pandas as pd
import numpy as np
import talib
from ta.trend import IchimokuIndicator
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any

app = FastAPI(title="Pro-Trader Ultimate TA-API")

# --- Models ---

class Candle(BaseModel):
    timestamp: Optional[str] = None
    open: float
    high: float
    low: float
    close: float
    volume: float

class AnalysisRequest(BaseModel):
    data: List[Candle]
    include_patterns: bool = True
    include_indicators: bool = True

# --- Helpers ---

def map_pattern_result(value: int) -> str:
    """Converts TA-Lib pattern output to human-readable string."""
    if value > 0:
        return "Bullish Signal"
    elif value < 0:
        return "Bearish Signal"
    return "No Pattern"

def calculate_supertrend(high, low, close, period=10, multiplier=3):
    """Calculates the Supertrend indicator."""
    hl2 = (high + low) / 2
    atr = talib.ATR(high, low, close, timeperiod=period)

    upperband = hl2 + (multiplier * atr)
    lowerband = hl2 - (multiplier * atr)

    supertrend = np.zeros(len(close))
    in_uptrend = np.ones(len(close), dtype=bool)

    for i in range(period, len(close)):
        if close[i] > upperband[i-1]:
            in_uptrend[i] = True
        elif close[i] < lowerband[i-1]:
            in_uptrend[i] = False
        else:
            in_uptrend[i] = in_uptrend[i-1]

            if in_uptrend[i] and lowerband[i] < lowerband[i-1]:
                lowerband[i] = lowerband[i-1]
            if not in_uptrend[i] and upperband[i] > upperband[i-1]:
                upperband[i] = upperband[i-1]

        supertrend[i] = lowerband[i] if in_uptrend[i] else upperband[i]

    return supertrend

# --- Endpoints ---

@app.post("/analyze")
async def analyze(request: AnalysisRequest):
    if len(request.data) < 52:
        raise HTTPException(
            status_code=400,
            detail="Minimum 52 candles required for full analysis (Ichimoku requirement)."
        )

    # Convert to DataFrame
    df = pd.DataFrame([c.model_dump() for c in request.data])

    # Prepare Numpy arrays for TA-Lib
    op = df['open'].values
    hi = df['high'].values
    lo = df['low'].values
    cl = df['close'].values
    vo = df['volume'].values

    # 1. Core Indicators (TA-Lib)
    if request.include_indicators:
        # RSI
        df['RSI'] = talib.RSI(cl, timeperiod=14)

        # MACD
        macd, macdsignal, macdhist = talib.MACD(cl, fastperiod=12, slowperiod=26, signalperiod=9)
        df['MACD'], df['MACD_signal'], df['MACD_hist'] = macd, macdsignal, macdhist

        # Bollinger Bands
        upper, mid, lower = talib.BBANDS(cl, timeperiod=20, nbdevup=2, nbdevdn=2)
        df['BB_upper'], df['BB_mid'], df['BB_lower'] = upper, mid, lower

        # EMAs
        df['EMA_20'] = talib.EMA(cl, timeperiod=20)
        df['EMA_50'] = talib.EMA(cl, timeperiod=50)
        df['EMA_200'] = talib.EMA(cl, timeperiod=200)

        # SMA
        df['SMA_20'] = talib.SMA(cl, timeperiod=20)

        # ATR
        df['ATR'] = talib.ATR(hi, lo, cl, timeperiod=14)

        # ADX
        df['ADX'] = talib.ADX(hi, lo, cl, timeperiod=14)

        # Stochastic
        slowk, slowd = talib.STOCH(hi, lo, cl, fastk_period=5, slowk_period=3, slowk_matype=0, slowd_period=3, slowd_matype=0)
        df['STOCH_k'], df['STOCH_d'] = slowk, slowd

        # VWAP
        df['VWAP'] = (df['volume'] * (df['high'] + df['low'] + df['close']) / 3).cumsum() / df['volume'].cumsum()

        # Supertrend
        df['Supertrend'] = calculate_supertrend(hi, lo, cl)

    # 2. Specialized Indicators (ta library)
    ichimoku = IchimokuIndicator(high=df['high'], low=df['low'])
    df['ichimoku_a'] = ichimoku.ichimoku_a()
    df['ichimoku_b'] = ichimoku.ichimoku_b()
    df['ichimoku_base'] = ichimoku.ichimoku_base_line()
    df['ichimoku_conv'] = ichimoku.ichimoku_conversion_line()

    # 3. Candlestick Patterns (TA-Lib)
    if request.include_patterns:
        patterns = {
            "Doji": talib.CDLDOJI(op, hi, lo, cl),
            "Hammer": talib.CDLHAMMER(op, hi, lo, cl),
            "Engulfing": talib.CDLENGULFING(op, hi, lo, cl),
            "MorningStar": talib.CDLMORNINGSTAR(op, hi, lo, cl),
            "ShootingStar": talib.CDLSHOOTINGSTAR(op, hi, lo, cl)
        }

        for name, results in patterns.items():
            df[name] = [map_pattern_result(v) for v in results]

    # 4. Signal Score Logic
    df['signal_score'] = 0
    if request.include_indicators:
        # RSI components
        df.loc[df['RSI'] < 30, 'signal_score'] += 20
        df.loc[df['RSI'] > 70, 'signal_score'] -= 20

        # BBands components
        df.loc[df['close'] < df['BB_lower'], 'signal_score'] += 15
        df.loc[df['close'] > df['BB_upper'], 'signal_score'] -= 15

        # EMA components
        df.loc[df['EMA_20'] > df['EMA_50'], 'signal_score'] += 10
        df.loc[df['EMA_20'] < df['EMA_50'], 'signal_score'] -= 10

    if request.include_patterns:
        # Pattern components
        for pattern in ["Doji", "Hammer", "Engulfing", "MorningStar", "ShootingStar"]:
            df.loc[df[pattern] == "Bullish Signal", 'signal_score'] += 25
            df.loc[df[pattern] == "Bearish Signal", 'signal_score'] -= 25

    # Clean up NaNs
    df = df.replace({np.nan: None})

    # Return only the last 5 candles
    return df.tail(5).to_dict(orient="records")

@app.post("/is-trend-bullish")
async def is_trend_bullish(request: AnalysisRequest):
    if len(request.data) < 200:
        raise HTTPException(status_code=400, detail="Need 200 candles for EMA 200 check.")

    df = pd.DataFrame([c.model_dump() for c in request.data])
    cl = df['close'].values
    hi = df['high'].values
    lo = df['low'].values

    ema200 = talib.EMA(cl, timeperiod=200)
    adx = talib.ADX(hi, lo, cl, timeperiod=14)

    current_close = cl[-1]
    current_ema = ema200[-1]
    current_adx = adx[-1]

    # Trend is bullish if price is above EMA 200 and ADX > 25 (strong trend)
    bullish = bool(current_close > current_ema and current_adx > 25)

    return {
        "bullish": bullish,
        "close": current_close,
        "ema_200": current_ema,
        "adx": current_adx
    }

@app.post("/scan-patterns")
async def scan_patterns(request: AnalysisRequest):
    df = pd.DataFrame([c.model_dump() for c in request.data])
    op = df['open'].values
    hi = df['high'].values
    lo = df['low'].values
    cl = df['close'].values

    patterns = {
        "Doji": talib.CDLDOJI(op, hi, lo, cl),
        "Hammer": talib.CDLHAMMER(op, hi, lo, cl),
        "Engulfing": talib.CDLENGULFING(op, hi, lo, cl),
        "MorningStar": talib.CDLMORNINGSTAR(op, hi, lo, cl),
        "ShootingStar": talib.CDLSHOOTINGSTAR(op, hi, lo, cl)
    }

    detected = []
    for i in range(len(df)):
        for name, results in patterns.items():
            if results[i] != 0:
                detected.append({
                    "index": i,
                    "timestamp": df.iloc[i].get('timestamp'),
                    "pattern": name,
                    "sentiment": map_pattern_result(results[i])
                })

    return {"patterns_found": detected}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
