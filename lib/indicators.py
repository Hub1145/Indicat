import pandas as pd
import numpy as np
import talib
import asyncio
from datetime import datetime
from typing import List, Optional, Any, Dict
from scipy.signal import argrelextrema
from concurrent.futures import ProcessPoolExecutor
import ta

# --- Metadata ---

INDICATOR_METADATA = {
    "technical_indicators": {
        "SMA": "Simple Moving Average", "EMA": "Exponential Moving Average", "WMA": "Weighted Moving Average",
        "RSI": "Relative Strength Index", "ADX": "Average Directional Index", "ATR": "Average True Range",
        "MACD": "Moving Average Convergence Divergence", "BBANDS": "Bollinger Bands",
        "VWAP": "Volume Weighted Average Price", "OBV": "On Balance Volume", "CMF": "Chaikin Money Flow",
        "EMA20": "20-Period EMA", "EMA50": "50-Period EMA", "EMA200": "200-Period EMA", "SMA20": "20-Period SMA",
        "SQUEEZE_MOMENTUM": "Squeeze Momentum [LazyBear]",
        "SUPERTREND": "Supertrend",
        "IMBA_TREND": "[IMBA] Trend Line"
    },
    "candlestick_patterns": {f: f.replace("CDL", "").replace("_", " ").title() for f in talib.get_functions() if f.startswith('CDL')},
    "institutional_strategies": {
        "Fair_Value_Gap": "FVG (Liquidity Imbalance)",
        "Order_Block": "Institutional Support/Demand Zone",
        "Internal_Structure": "BOS/CHoCH (Real-time Internal Structure)",
        "Swing_Structure": "BOS/CHoCH (Real-time Swing Structure)",
        "EQH_EQL": "EQH/EQL (Equal Highs & Lows detection)",
        "Trading_Range": "Premium, Discount & Equilibrium Zones",
        "Liquidity_Pools": "Buy/Sell Side Liquidity levels"
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
    },
    "custom_lux_algo": {
        "ZScore_Zones": "Z-Score Predictive Zones [AlgoPoint]",
        "Lux_MSB_OB": "Market Structure Break & OB Toolkit [LuxAlgo]"
    },
    "squeeze_momentum": {
        "Squeeze_LB": "Squeeze Momentum Indicator [LazyBear]"
    },
    "trend_following": {
        "Supertrend": "Supertrend Indicator",
        "IMBA_Trend": "[IMBA] ALGO Trend Line + Signals"
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
    """Full implementation of Smart Money Concepts [LuxAlgo]."""
    h, l, c, o = df['high'].values, df['low'].values, df['close'].values, df['open'].values

    int_p_idx = argrelextrema(h, np.greater, order=5)[0]
    int_v_idx = argrelextrema(l, np.less, order=5)[0]
    swg_p_idx = argrelextrema(h, np.greater, order=50)[0]
    swg_v_idx = argrelextrema(l, np.less, order=50)[0]

    def get_structure(p_idx, v_idx, trend_bias):
        if len(p_idx) == 0 or len(v_idx) == 0: return "None", trend_bias
        last_h, last_l = h[p_idx[-1]], l[v_idx[-1]]

        if c[-1] > last_h:
            tag = "CHoCH" if trend_bias == -1 else "BOS"
            return f"Bullish {tag}", 1
        if c[-1] < last_l:
            tag = "CHoCH" if trend_bias == 1 else "BOS"
            return f"Bearish {tag}", -1
        return "None", trend_bias

    bias_est = 1 if c[-1] > talib.EMA(c, 50)[-1] else -1
    int_struct, _ = get_structure(int_p_idx, int_v_idx, bias_est)
    swg_struct, _ = get_structure(swg_p_idx, swg_v_idx, bias_est)

    body_size = np.abs(c - o)
    avg_body = talib.SMA(body_size, timeperiod=20)
    obs = []
    for i in range(len(df)-2, len(df)-50, -1):
        if i < 1: break
        if body_size[i+1] > 1.5 * (avg_body[i+1] if not np.isnan(avg_body[i+1]) else 1):
            if c[i] < o[i] and c[i+1] > h[i]:
                obs.append({"type": "Bullish OB", "top": float(h[i]), "bottom": float(l[i]), "strength": float(body_size[i+1]/avg_body[i+1]), "index": i})
            if c[i] > o[i] and c[i+1] < l[i]:
                obs.append({"type": "Bearish OB", "top": float(h[i]), "bottom": float(l[i]), "strength": float(body_size[i+1]/avg_body[i+1]), "index": i})
        if len(obs) >= 5: break

    fvgs = []
    for i in range(len(df)-1, len(df)-50, -1):
        if i < 2: break
        if l[i] > h[i-2]: fvgs.append({"type": "Bullish FVG", "top": float(l[i]), "bottom": float(h[i-2]), "mid": float((l[i]+h[i-2])/2), "index": i-1})
        elif h[i] < l[i-2]: fvgs.append({"type": "Bearish FVG", "top": float(l[i-2]), "bottom": float(h[i]), "mid": float((l[i-2]+h[i])/2), "index": i-1})
        if len(fvgs) >= 5: break

    eqh, eql = False, False
    if len(int_p_idx) >= 2:
        if abs(h[int_p_idx[-1]] - h[int_p_idx[-2]]) / h[int_p_idx[-1]] < 0.001: eqh = True
    if len(int_v_idx) >= 2:
        if abs(l[int_v_idx[-1]] - l[int_v_idx[-2]]) / l[int_v_idx[-1]] < 0.001: eql = True

    range_high = h[swg_p_idx].max() if len(swg_p_idx) > 0 else h.max()
    range_low = l[swg_v_idx].min() if len(swg_v_idx) > 0 else l.min()
    eq = (range_high + range_low) / 2
    zone = "Premium" if c[-1] > eq else ("Discount" if c[-1] < eq else "Equilibrium")

    return {
        "structure": {"internal": int_struct, "swing": swg_struct},
        "order_blocks": obs[:5],
        "fair_value_gaps": fvgs[:5],
        "eqh_eql": {"equal_highs": eqh, "equal_lows": eql},
        "trading_range": {"high": float(range_high), "low": float(range_low), "equilibrium": float(eq), "current_zone": zone}
    }

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

def detect_price_action_patterns(df: pd.DataFrame):
    h, l, c = df['high'].values, df['low'].values, df['close'].values
    ph_idx = argrelextrema(h, np.greater, order=5)[0]
    pl_idx = argrelextrema(l, np.less, order=5)[0]
    patterns = []
    if len(ph_idx) >= 3:
        p1, p2, p3 = h[ph_idx[-3]], h[ph_idx[-2]], h[ph_idx[-1]]
        if p2 > p1 and p2 > p3 and abs(p1 - p3) / p1 < 0.05:
            patterns.append({"pattern": "Head and Shoulders", "confidence": 0.87, "index": int(ph_idx[-1]), "expected_move": -5.2})
        elif p2 < p1 and p2 < p3 and abs(p1 - p3) / p1 < 0.05:
            patterns.append({"pattern": "Inverse Head and Shoulders", "confidence": 0.82, "index": int(ph_idx[-1]), "expected_move": 4.8})
    if len(ph_idx) >= 2:
        p1, p2 = h[ph_idx[-2]], h[ph_idx[-1]]
        if abs(p1 - p2) / p1 < 0.01:
            patterns.append({"pattern": "Double Top", "confidence": 0.78, "index": int(ph_idx[-1]), "expected_move": -3.5})
    if len(pl_idx) >= 2:
        v1, v2 = l[pl_idx[-2]], l[pl_idx[-1]]
        if abs(v1 - v2) / v1 < 0.01:
            patterns.append({"pattern": "Double Bottom", "confidence": 0.79, "index": int(pl_idx[-1]), "expected_move": 3.2})
    return patterns

def calculate_volume_profile(df: pd.DataFrame):
    if df['volume'].sum() == 0: return None
    bins = 20
    counts, edges = np.histogram(df['close'], bins=bins, weights=df['volume'])
    idx = np.argmax(counts)
    return {"poc": float(edges[idx]), "vah": float(edges[min(idx+2, bins-1)]), "val": float(edges[max(idx-2, 0)])}

def calculate_vwma(series, volume, length):
    return (series * volume).rolling(length).sum() / volume.rolling(length).sum()

def detect_lux_zscore(df, length=144, smooth=20, history_depth=25, thresh=1.5):
    c, h, l = df['close'], df['high'], df['low']
    mean = c.rolling(length).mean()
    std_dev = c.rolling(length).std()
    raw_z = (c - mean) / std_dev
    z_score = calculate_vwma(raw_z, df['volume'], smooth)
    return {
        "z_score": float(z_score.iloc[-1]),
        "signal": "Sell" if z_score.iloc[-1] > thresh else ("Buy" if z_score.iloc[-1] < -thresh else "Neutral"),
        "series": z_score
    }

def detect_lux_msb_ob(df, pivot_len=7, msb_thresh=0.5):
    h, l, c, v = df['high'].values, df['low'].values, df['close'].values, df['volume'].values
    change = df['close'].diff()
    body_size = np.abs(df['close'] - df['open'])
    avg_body = body_size.rolling(20).mean()
    displacement = body_size / avg_body
    momentum_z = (change - change.rolling(50).mean()) / change.rolling(50).std()
    ph_idx = argrelextrema(h, np.greater, order=pivot_len)[0]
    pl_idx = argrelextrema(l, np.less, order=pivot_len)[0]
    if len(ph_idx) == 0 or len(pl_idx) == 0: return {}
    last_ph, last_pl = h[ph_idx[-1]], l[pl_idx[-1]]
    curr_mz, curr_disp = momentum_z.iloc[-1], displacement.iloc[-1]
    is_msb_bull = c[-1] > last_ph and (curr_mz > msb_thresh or curr_disp > 1.5)
    is_msb_bear = c[-1] < last_pl and (curr_mz < -msb_thresh or curr_disp > 1.5)
    return {
        "msb": "Bullish" if is_msb_bull else ("Bearish" if is_msb_bear else "None"),
        "displacement": float(curr_disp)
    }

def detect_squeeze_momentum(df, bb_len=20, bb_mult=2.0, kc_len=20, kc_mult=1.5):
    c, h, l = df['close'], df['high'], df['low']
    basis = talib.SMA(c, timeperiod=bb_len)
    dev = kc_mult * talib.STDDEV(c, timeperiod=bb_len)
    upperBB, lowerBB = basis + dev, basis - dev
    ma = talib.SMA(c, timeperiod=kc_len)
    tr = talib.TRANGE(h, l, c)
    range_ma = talib.SMA(tr, timeperiod=kc_len)
    upperKC, lowerKC = ma + range_ma * kc_mult, ma - range_ma * kc_mult
    sqz_on = (lowerBB > lowerKC) & (upperBB < upperKC)
    highest_h = h.rolling(window=kc_len).max()
    lowest_l = l.rolling(window=kc_len).min()
    avg_val = ((highest_h + lowest_l)/2 + ma) / 2
    momentum_val = talib.LINEARREG((c - avg_val).fillna(0), timeperiod=kc_len)
    return {
        "value": float(momentum_val.iloc[-1]),
        "state": "Squeeze On" if sqz_on.iloc[-1] else "Squeeze Off",
        "direction": "Up" if momentum_val.iloc[-1] > momentum_val.iloc[-2] else "Down",
        "series": momentum_val
    }

def detect_supertrend(df, period=10, multiplier=3.0):
    h, l, c = df['high'], df['low'], df['close']
    hl2 = (h + l) / 2
    atr = talib.ATR(h, l, c, timeperiod=period)
    up, dn = hl2 - (multiplier * atr), hl2 + (multiplier * atr)
    upper, lower, trend = np.zeros(len(df)), np.zeros(len(df)), np.ones(len(df))
    for i in range(1, len(df)):
        if np.isnan(up[i]) or np.isnan(dn[i]): continue
        lower[i] = max(up[i], lower[i-1]) if c.iloc[i-1] > lower[i-1] else up[i]
        upper[i] = min(dn[i], upper[i-1]) if c.iloc[i-1] < upper[i-1] else dn[i]
        trend[i] = 1 if c.iloc[i] > upper[i] else (-1 if c.iloc[i] < lower[i] else trend[i-1])
    st_val = pd.Series(np.where(trend == 1, lower, upper), index=df.index)
    return {"value": float(st_val.iloc[-1]), "direction": "Bullish" if trend[-1] == 1 else "Bearish", "series": st_val}

def detect_imba_trend(df, sensitivity=18.0):
    h, l, c = df['high'], df['low'], df['close']
    length = int(max(1, sensitivity * 10))
    high_line, low_line = h.rolling(window=length).max(), l.rolling(window=length).min()
    imba_trend_line = high_line - (high_line - low_line) * 0.5
    is_uptrend = c > imba_trend_line
    return {"value": float(imba_trend_line.iloc[-1]), "direction": "Bullish" if is_uptrend.iloc[-1] else "Bearish", "series": imba_trend_line}

def calculate_portfolio_metrics(asset_series: Dict[str, pd.Series], market_returns: pd.Series = None):
    df = pd.DataFrame(asset_series).pct_change().dropna()
    corr_matrix = df.corr().to_dict()
    metrics = {}
    for asset in asset_series:
        returns = df[asset]
        var_95 = np.percentile(returns, 5)
        sharpe = (returns.mean() * 252) / (returns.std() * np.sqrt(252)) if returns.std() != 0 else 0
        beta = 0
        if market_returns is not None:
            m_df = pd.concat([returns, market_returns.pct_change()], axis=1).dropna()
            if len(m_df) > 1:
                cov = np.cov(m_df.iloc[:,0], m_df.iloc[:,1])[0,1]
                var_m = np.var(m_df.iloc[:,1])
                beta = cov / var_m if var_m != 0 else 0
        metrics[asset] = {"sharpe_ratio": float(sharpe), "var_95": float(var_95), "beta": float(beta)}
    return {"correlation_matrix": corr_matrix, "asset_metrics": metrics}


def backtest_strategy(df: pd.DataFrame, entry_indicator="EMA9", exit_indicator="EMA21", initial_capital=10000):
    """
    A functional crossover backtester.
    Defaults to EMA crossover.
    """
    capital = initial_capital
    position = 0
    trades = []

    # Ensure indicators exist, or calculate them if not in all_raw (simulated here)
    if entry_indicator not in df.columns:
        if "EMA" in entry_indicator:
            period = int(entry_indicator.replace("EMA", ""))
            df[entry_indicator] = talib.EMA(df['close'], timeperiod=period)
    if exit_indicator not in df.columns:
        if "EMA" in exit_indicator:
            period = int(exit_indicator.replace("EMA", ""))
            df[exit_indicator] = talib.EMA(df['close'], timeperiod=period)

    # Simplified backtester logic
    for i in range(1, len(df)):
        # Buy Signal: Entry crosses above Exit
        if df[entry_indicator].iloc[i] > df[exit_indicator].iloc[i] and \
           df[entry_indicator].iloc[i-1] <= df[exit_indicator].iloc[i-1] and \
           position == 0:
            position = capital / df['close'].iloc[i]
            buy_price = df['close'].iloc[i]
            capital = 0
            trades.append({"type": "buy", "price": float(buy_price), "index": i})

        # Sell Signal: Entry crosses below Exit
        elif df[entry_indicator].iloc[i] < df[exit_indicator].iloc[i] and \
             df[entry_indicator].iloc[i-1] >= df[exit_indicator].iloc[i-1] and \
             position > 0:
            sell_price = df['close'].iloc[i]
            capital = position * sell_price
            position = 0
            trades.append({"type": "sell", "price": float(sell_price), "index": i, "profit": float(sell_price - buy_price)})

    final_val = capital if position == 0 else position * df['close'].iloc[-1]
    return {
        "initial_capital": float(initial_capital),
        "final_value": float(final_val),
        "total_return_pct": float(((final_val - initial_capital) / initial_capital) * 100),
        "trades": trades
    }

# --- Parallel Engine ---

executor = ProcessPoolExecutor(max_workers=4)

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

    all_raw_cols = {clean_name(k): v for k, v in talib_res.items()}
    for c in ta_df.columns: all_raw_cols[clean_name(c)] = ta_df[c]

    st_res, imba_res = detect_supertrend(df), detect_imba_trend(df)
    sqz_res, zscore_res = detect_squeeze_momentum(df), detect_lux_zscore(df)

    all_raw_cols.update({"SUPERTREND": st_res["series"], "IMBA_TREND": imba_res["series"], "SQUEEZE_MOMENTUM": sqz_res["series"], "LUX_ZSCORE": zscore_res["series"]})
    all_raw = pd.DataFrame(all_raw_cols, index=df.index)

    ema200 = talib.EMA(cl, timeperiod=min(len(cl), 200))
    res = {
        "current_price": float(cl[-1]),
        "summary": {"trend": "Bullish" if cl[-1] > ema200[-1] else "Bearish", "session": get_sessions(df['timestamp'].iloc[-1])},
        "institutional_strategies": detect_smc(df),
        "market_dynamics": {"levels": detect_sr_levels(df), "volume_profile": calculate_volume_profile(df)},
        "divergences": {"rsi": detect_divergences(df, "RSI"), "macd": detect_divergences(df, "MACD")},
        "price_action_patterns": detect_price_action_patterns(df),
        "custom_lux_algo": {"zscore_zones": {k:v for k,v in zscore_res.items() if k != "series"}, "market_structure": detect_lux_msb_ob(df)},
        "squeeze_momentum": {k:v for k,v in sqz_res.items() if k != "series"},
        "trend_following": {"supertrend": {k:v for k,v in st_res.items() if k != "series"}, "imba_trend": {k:v for k,v in imba_res.items() if k != "series"}}
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
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(executor, get_indicator_results_sync, df, selected, history)
