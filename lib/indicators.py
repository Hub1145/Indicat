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

def _get_indicators_metadata():
    """Dynamically builds metadata from TA-Lib groups and custom indicators."""
    talib_groups = talib.get_function_groups()

    # Custom/Advanced categories
    meta = {
        "institutional_strategies": {
            "Fair_Value_Gap": "FVG (Liquidity Imbalance)",
            "Order_Block": "Institutional Support/Demand Zone",
            "Market_Structure": "MSS/BOS (Trend Shifts & Continuation)",
            "BPR": "Balanced Price Range (Overlapping FVGs)",
            "Volume_Imbalance": "Price gaps between candle bodies",
            "Liquidity_Pools": "Buy/Sell Side Liquidity levels",
            "Opening_Gaps": "NWOG & NDOG (Weekly/Daily Gaps)"
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
            "Volume_Profile": "Price Distribution Analysis",
            "VWAP_Volume_Profile": "VWAP Volume Profile [BigBeluga]",
            "UT_BOT_ALERTS": "UT Bot Alerts [QuantNomad]"
        },
        "custom_lux_algo": {
            "ZScore_Zones": "Z-Score Predictive Zones [AlgoPoint]",
            "Lux_MSB_OB": "Market Structure Break & OB Toolkit [LuxAlgo]"
        },
        "squeeze_momentum": {
            "Squeeze_LB": "Squeeze Momentum Indicator [LazyBear]"
        },
        "cycle_indicators": {
            "Normalized_Resonator": "Normalized Resonator [LuxAlgo]"
        },
        "trend_following": {
            "Supertrend": "Supertrend Indicator",
            "IMBA_Trend": "[IMBA] ALGO Trend Line + Signals",
            "Trend_Pro_Z": "Trend-Pro + Z [andrwxwy]",
            "Ichimoku": "Ichimoku Cloud (Tenkan, Kijun, Senkou A/B)"
        },
        "forecasting_models": {
            "Harmonic_Forecast": "Adaptive Harmonic Forecast [LuxAlgo]",
            "MTF_MACD_Forecast": "MTF MACD Strategy with Forecasting"
        },
        "market_intelligence": {
            "Market_Regime": "Market Regime Detection Engine",
            "Liquidity_Heatmap": "Smart Liquidity & Sweep Probability",
            "Strategy_Discovery": "AI Strategy Generator & Optimizer"
        }
    }

    # Map TA-Lib groups to metadata categories
    group_mapping = {
        'Cycle Indicators': 'cycle_indicators',
        'Math Operators': 'math_operators',
        'Math Transform': 'math_transform',
        'Momentum Indicators': 'momentum_indicators',
        'Overlap Studies': 'overlap_studies',
        'Pattern Recognition': 'candlestick_patterns',
        'Price Transform': 'price_transform',
        'Statistic Functions': 'statistic_functions',
        'Volatility Indicators': 'volatility_indicators',
        'Volume Indicators': 'volume_indicators'
    }

    for talib_group, category in group_mapping.items():
        if category not in meta:
            meta[category] = {}
        for func in talib_groups[talib_group]:
            # Friendly name generation
            name = func
            if func.startswith('CDL'):
                name = func.replace("CDL", "").replace("_", " ").title()

            # Special case for well-known indicators
            known = {
                "RSI": "Relative Strength Index", "ADX": "Average Directional Index",
                "ATR": "Average True Range", "MACD": "Moving Average Convergence Divergence",
                "BBANDS": "Bollinger Bands", "SMA": "Simple Moving Average",
                "EMA": "Exponential Moving Average", "WMA": "Weighted Moving Average",
                "OBV": "On Balance Volume"
            }
            if func in known:
                name = known[func]

            meta[category][func] = name

    return meta

INDICATOR_METADATA = _get_indicators_metadata()

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

def detect_displacement(df):
    """Detects displacement candles based on body-to-wick ratio and relative size."""
    o, h, l, c = df['open'].values, df['high'].values, df['low'].values, df['close'].values
    body = np.abs(c - o)
    mean_body = talib.SMA(body, timeperiod=20)

    # Body should be > 36% of total range and larger than average
    range_tot = h - l
    range_tot = np.where(range_tot == 0, 1e-9, range_tot)
    body_perc = body / range_tot

    is_displacement = (body_perc > 0.36) & (body > (mean_body if not np.isnan(mean_body[-1]) else 0))
    return is_displacement

def detect_volume_imbalance(df):
    """Detects gaps between the bodies of adjacent candles (Volume Imbalance)."""
    o, h, l, c = df['open'].values, df['high'].values, df['low'].values, df['close'].values
    mx = np.maximum(c, o)
    mn = np.minimum(c, o)

    vimb_bl = (o > c[np.maximum(0, np.arange(len(c))-1)]) & (h[np.maximum(0, np.arange(len(c))-1)] > l) & (c > c[np.maximum(0, np.arange(len(c))-1)]) & (o > o[np.maximum(0, np.arange(len(c))-1)]) & (h[np.maximum(0, np.arange(len(c))-1)] < mn)
    vimb_br = (o < c[np.maximum(0, np.arange(len(c))-1)]) & (l[np.maximum(0, np.arange(len(c))-1)] < h) & (c < c[np.maximum(0, np.arange(len(c))-1)]) & (o < o[np.maximum(0, np.arange(len(c))-1)]) & (l[np.maximum(0, np.arange(len(c))-1)] > mx)

    return vimb_bl, vimb_br

def detect_opening_gaps(df):
    """Detects New Week Opening Gaps (NWOG) and New Day Opening Gaps (NDOG)."""
    if 'timestamp' not in df.columns: return [], []

    df = df.copy()
    df['dt'] = pd.to_datetime(df['timestamp'])
    df['day'] = df['dt'].dt.dayofweek

    nwogs = []
    ndogs = []

    for i in range(1, len(df)):
        if df['day'].iloc[i] != df['day'].iloc[i-1]:
            c_prev = df['close'].iloc[i-1]
            o_curr = df['open'].iloc[i]
            if abs(o_curr - c_prev) > 0:
                ndogs.append({"type": "NDOG", "top": float(max(c_prev, o_curr)), "bottom": float(min(c_prev, o_curr)), "index": i})
            if df['day'].iloc[i] == 0 and df['day'].iloc[i-1] == 4:
                nwogs.append({"type": "NWOG", "top": float(max(c_prev, o_curr)), "bottom": float(min(c_prev, o_curr)), "index": i})

    return nwogs[-5:], ndogs[-5:]

def detect_smc(df):
    """Full implementation of ICT/SMC Concepts [LuxAlgo]."""
    h, l, c, o = df['high'].values, df['low'].values, df['close'].values, df['open'].values
    int_p_idx = argrelextrema(h, np.greater, order=5)[0]
    int_v_idx = argrelextrema(l, np.less, order=5)[0]

    def get_mss_bos(p_idx, v_idx):
        if len(p_idx) < 2 or len(v_idx) < 2: return "None", "None"
        last_h, last_l = h[p_idx[-1]], l[v_idx[-1]]
        prev_h, prev_l = h[p_idx[-2]], l[v_idx[-2]]
        ema50 = talib.EMA(c, 50)[-1]
        bias = 1 if c[-1] > ema50 else -1
        mss = "None"
        if bias == 1 and c[-1] > last_h: mss = "Bullish MSS"
        if bias == -1 and c[-1] < last_l: mss = "Bearish MSS"
        bos = "None"
        if bias == 1 and c[-1] > prev_h: bos = "Bullish BOS"
        if bias == -1 and c[-1] < prev_l: bos = "Bearish BOS"
        return mss, bos

    mss, bos = get_mss_bos(int_p_idx, int_v_idx)
    is_displ = detect_displacement(df)
    fvgs = []
    for i in range(len(df)-1, 2, -1):
        if l[i] > h[i-2]: fvgs.append({"type": "Bullish FVG", "top": float(l[i]), "bottom": float(h[i-2]), "is_displaced": bool(is_displ[i-1]), "index": i-1})
        elif h[i] < l[i-2]: fvgs.append({"type": "Bearish FVG", "top": float(l[i-2]), "bottom": float(h[i]), "is_displaced": bool(is_displ[i-1]), "index": i-1})
        if len(fvgs) >= 10: break

    bprs = []
    bull_fvgs = [f for f in fvgs if f["type"] == "Bullish FVG"]
    bear_fvgs = [f for f in fvgs if f["type"] == "Bearish FVG"]
    for bf in bull_fvgs:
        for rf in bear_fvgs:
            top = min(bf["top"], rf["top"])
            bottom = max(bf["bottom"], rf["bottom"])
            if top > bottom: bprs.append({"top": float(top), "bottom": float(bottom), "mid": float((top+bottom)/2)})
            if len(bprs) >= 5: break
        if len(bprs) >= 5: break

    obs = []
    body_size = np.abs(c - o)
    avg_body = talib.SMA(body_size, timeperiod=20)
    for i in range(len(df)-2, 1, -1):
        if body_size[i+1] > 1.5 * (avg_body[i+1] if not np.isnan(avg_body[i+1]) else 1):
            ob_type = None
            if c[i] < o[i] and c[i+1] > h[i]: ob_type = "Bullish OB"
            if c[i] > o[i] and c[i+1] < l[i]: ob_type = "Bearish OB"
            if ob_type:
                is_breaker = (ob_type == "Bullish OB" and c[-1] < l[i]) or (ob_type == "Bearish OB" and c[-1] > h[i])
                obs.append({"type": "Breaker" if is_breaker else ob_type, "top": float(h[i]), "bottom": float(l[i]), "index": i})
        if len(obs) >= 5: break

    liq_b, liq_s = [], []
    if len(int_p_idx) >= 3:
        recent_hs = h[int_p_idx[-10:]]
        for p in recent_hs:
            if np.sum(np.abs(recent_hs - p) / p < 0.002) >= 2:
                liq_b.append({"price": float(p), "type": "Buyside Liquidity"})
                break
    if len(int_v_idx) >= 3:
        recent_ls = l[int_v_idx[-10:]]
        for p in recent_ls:
            if np.sum(np.abs(recent_ls - p) / p < 0.002) >= 2:
                liq_s.append({"price": float(p), "type": "Sellside Liquidity"})
                break

    vi_bl, vi_br = detect_volume_imbalance(df)
    vis = []
    if len(vi_bl) > 0 and vi_bl[-1]: vis.append("Bullish Volume Imbalance")
    if len(vi_br) > 0 and vi_br[-1]: vis.append("Bearish Volume Imbalance")
    nwogs, ndogs = detect_opening_gaps(df)

    return {
        "market_structure": {"mss": mss, "bos": bos},
        "fvgs": fvgs[:5],
        "bpr": bprs,
        "order_blocks": obs,
        "liquidity_pools": {"buyside": liq_b, "sellside": liq_s},
        "volume_imbalance": vis,
        "opening_gaps": {"nwog": nwogs, "ndog": ndogs},
        "trading_range": {"zone": "Premium" if c[-1] > (h.max()+l.min())/2 else "Discount"}
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
        if p2 > p1 and p2 > p3 and abs(p1 - p3) / p1 < 0.05: patterns.append({"pattern": "Head and Shoulders", "confidence": 0.87, "index": int(ph_idx[-1]), "expected_move": -5.2})
        elif p2 < p1 and p2 < p3 and abs(p1 - p3) / p1 < 0.05: patterns.append({"pattern": "Inverse Head and Shoulders", "confidence": 0.82, "index": int(ph_idx[-1]), "expected_move": 4.8})
    if len(ph_idx) >= 2:
        p1, p2 = h[ph_idx[-2]], h[ph_idx[-1]]
        if abs(p1 - p2) / p1 < 0.01: patterns.append({"pattern": "Double Top", "confidence": 0.78, "index": int(ph_idx[-1]), "expected_move": -3.5})
    if len(pl_idx) >= 2:
        v1, v2 = l[pl_idx[-2]], l[pl_idx[-1]]
        if abs(v1 - v2) / v1 < 0.01: patterns.append({"pattern": "Double Bottom", "confidence": 0.79, "index": int(pl_idx[-1]), "expected_move": 3.2})
    return patterns

def calculate_volume_profile(df: pd.DataFrame):
    if df['volume'].sum() == 0: return None
    bins = 20
    counts, edges = np.histogram(df['close'], bins=bins, weights=df['volume'])
    idx = np.argmax(counts)
    return {"poc": float(edges[idx]), "vah": float(edges[min(idx+2, bins-1)]), "val": float(edges[max(idx-2, 0)])}

def calculate_hma(series, length):
    half_length = int(length / 2)
    sqrt_length = int(np.sqrt(length))
    wma_half = talib.WMA(series, timeperiod=half_length)
    wma_full = talib.WMA(series, timeperiod=length)
    diff = 2 * wma_half - wma_full
    return talib.WMA(diff, timeperiod=sqrt_length)

def calculate_linreg_with_offset(series, length, offset):
    slope = talib.LINEARREG_SLOPE(series, length)
    intercept = talib.LINEARREG_INTERCEPT(series, length)
    return intercept + slope * (length - 1 + offset)

def calculate_ichimoku(df: pd.DataFrame):
    """Ichimoku Cloud Calculation."""
    if len(df) < 52: return None
    h, l = df['high'], df['low']

    tenkan_sen = (h.rolling(window=9).max() + l.rolling(window=9).min()) / 2
    kijun_sen = (h.rolling(window=26).max() + l.rolling(window=26).min()) / 2
    senkou_span_a = ((tenkan_sen + kijun_sen) / 2).shift(26)
    senkou_span_b = ((h.rolling(window=52).max() + l.rolling(window=52).min()) / 2).shift(26)
    chikou_span = df['close'].shift(-26)

    return {
        "tenkan_sen": float(tenkan_sen.iloc[-1]),
        "kijun_sen": float(kijun_sen.iloc[-1]),
        "senkou_span_a": float(senkou_span_a.iloc[-1]) if not np.isnan(senkou_span_a.iloc[-1]) else None,
        "senkou_span_b": float(senkou_span_b.iloc[-1]) if not np.isnan(senkou_span_b.iloc[-1]) else None,
        "is_bullish": bool(df['close'].iloc[-1] > senkou_span_a.iloc[-1] and df['close'].iloc[-1] > senkou_span_b.iloc[-1]) if not np.isnan(senkou_span_a.iloc[-1]) else False
    }

def calculate_trend_pro_z(df: pd.DataFrame):
    if len(df) < 55: return None
    c, h, l, o = df['close'].values, df['high'].values, df['low'].values, df['open'].values
    hlc3 = (h + l + c) / 3
    lsma = calculate_linreg_with_offset(hlc3, 39, -9)
    hma_azul = calculate_hma(hlc3, 45)
    slope_azul = np.diff(hma_azul, prepend=np.nan)
    pseudo_azul = hma_azul + 9 * slope_azul
    final_azul = talib.EMA(pseudo_azul[~np.isnan(pseudo_azul)], timeperiod=7)
    final_azul_full = np.full(len(df), np.nan)
    final_azul_full[len(df)-len(final_azul):] = final_azul
    hma_ = calculate_hma(hlc3, 55)
    slope_ = np.diff(hma_, prepend=np.nan)
    pseudo_ = hma_ + 11 * slope_
    final_ = talib.EMA(pseudo_[~np.isnan(pseudo_)], timeperiod=7)
    final_full = np.full(len(df), np.nan)
    final_full[len(df)-len(final_):] = final_
    is_up = final_azul_full > np.roll(final_azul_full, 1)
    is_down = final_azul_full < np.roll(final_azul_full, 1)
    above_gma = c > lsma
    below_gma = c < lsma
    b_cond = (o > final_full) | (c > final_full)
    s_cond = (o < final_full) | (c < final_full)
    is_green = is_up & above_gma
    is_red = is_down & below_gma
    buy_signal = is_green & ~np.roll(is_green, 1) & (c >= o) & b_cond
    sell_signal = is_red & ~np.roll(is_red, 1) & (c <= o) & s_cond
    atr9 = talib.ATR(h, l, c, timeperiod=9)
    trail_s = np.zeros(len(df))
    trail_s[0] = c[0]
    for i in range(1, len(df)):
        band = atr9[i] * 1.0
        up, dn = c[i] + band, c[i] - band
        curr_trail = trail_s[i-1]
        if dn > curr_trail: curr_trail = dn
        if up < curr_trail: curr_trail = up
        trail_s[i] = curr_trail
    score_s = np.zeros(len(df))
    for i in range(12, len(df)):
        s = sum([1 if trail_s[i] > trail_s[i-j] else -1 for j in range(1, 13)])
        score_s[i] = s
    sig_s = np.ones(len(df))
    for i in range(1, len(df)):
        if score_s[i] > 4: sig_s[i] = 1
        elif score_s[i] < -4 and score_s[i-1] >= -4: sig_s[i] = -1
        else: sig_s[i] = sig_s[i-1]
    return {"buy_signal": bool(buy_signal[-1]), "sell_signal": bool(sell_signal[-1]), "long_signal": bool((sig_s[-1] == 1) and (sig_s[-2] == -1) and above_gma[-1]), "short_signal": bool((sig_s[-1] == -1) and (sig_s[-2] == 1) and below_gma[-1]), "score": float(score_s[-1]), "lsma": float(lsma[-1]), "trail_s": float(trail_s[-1])}

def calculate_adaptive_harmonic_forecast(df: pd.DataFrame, lookback=100, extrap=50, num_sines=5, min_p=10):
    """Adaptive Harmonic Forecast [LuxAlgo]"""
    if len(df) < lookback: return None
    c = df['close'].tail(lookback).values
    x = np.arange(lookback)
    A = np.vstack([x, np.ones(lookback)]).T
    try: m, b = np.linalg.lstsq(A, c, rcond=None)[0]
    except: return None
    detrended = c - (m * x + b)
    periods = np.arange(min_p, lookback + 1)
    powers = np.array([np.sum(detrended * np.sin((2*np.pi/p) * x))**2 + np.sum(detrended * np.cos((2*np.pi/p) * x))**2 for p in periods])
    peak_idx = argrelextrema(powers, np.greater)[0]
    if len(peak_idx) == 0: return None
    best_periods = periods[peak_idx[np.argsort(powers[peak_idx])[::-1][:num_sines]]]
    X_mat = np.zeros((lookback, len(best_periods)*2 + 2))
    for i in range(lookback):
        for j, p in enumerate(best_periods):
            X_mat[i, j*2], X_mat[i, j*2+1] = np.sin((2*np.pi/p)*i), np.cos((2*np.pi/p)*i)
        X_mat[i, -2], X_mat[i, -1] = float(i), 1.0
    try: Beta = np.linalg.lstsq(X_mat, c, rcond=None)[0]
    except: return None
    forecast = []
    for i in range(extrap + 1):
        t = float(lookback - 1 + i)
        y = sum([Beta[j*2]*np.sin((2*np.pi/p)*t) + Beta[j*2+1]*np.cos((2*np.pi/p)*t) for j, p in enumerate(best_periods)]) + Beta[-2]*t + Beta[-1]
        forecast.append({"step": i, "value": float(y), "trend": float(Beta[-2]*t + Beta[-1])})
    return {"best_periods": [float(p) for p in best_periods], "forecast": forecast, "slope": float(Beta[-2])}

def calculate_liquidity_heatmap(df: pd.DataFrame):
    """
    Smart Liquidity Map
    Calculates liquidity density and sweep probabilities.
    """
    if len(df) < 50: return None
    h, l, c = df['high'].values, df['low'].values, df['close'].values

    # Identify key zones (using argrelextrema for pivot clusters)
    ph_idx = argrelextrema(h, np.greater, order=10)[0]
    pl_idx = argrelextrema(l, np.less, order=10)[0]

    recent_price = c[-1]
    atr = talib.ATR(h, l, c, timeperiod=14)[-1]

    def get_density_and_prob(levels, is_highs):
        if len(levels) == 0: return []

        # Grid price range
        min_p, max_p = l.min(), h.max()
        grid = np.linspace(min_p, max_p, 50)
        density = np.zeros(len(grid))

        for p in levels:
            # Structures contribute to density around their price
            dist = np.abs(grid - p)
            # Gaussian contribution
            density += np.exp(-(dist**2) / (2 * (atr * 0.5)**2))

        # Find clusters
        top_idx = np.argsort(density)[::-1][:5]
        clusters = []
        for idx in top_idx:
            price_lv = grid[idx]
            # Probability of sweep: higher if price is close but hasn't hit yet
            dist_to_price = abs(price_lv - recent_price)
            prob = 0
            if dist_to_price < atr * 2:
                prob = 100 * np.exp(-dist_to_price / (atr))

            clusters.append({
                "price": float(price_lv),
                "density_score": float(density[idx]),
                "sweep_probability": float(prob),
                "type": "Buy-side" if is_highs else "Sell-side"
            })
        return clusters

    buyside = get_density_and_prob(h[ph_idx], True)
    sellside = get_density_and_prob(l[pl_idx], False)

    # Combined heatmap score
    all_clusters = buyside + sellside
    max_prob = max([c['sweep_probability'] for c in all_clusters]) if all_clusters else 0

    return {
        "clusters": all_clusters,
        "max_sweep_risk": float(max_prob),
        "liquidity_state": "High" if max_prob > 70 else ("Medium" if max_prob > 30 else "Low")
    }

def calculate_normalized_resonator(df: pd.DataFrame, period=100, delta=0.5, lookback_mult=1.0, signal_len=9):
    """
    Normalized Resonator [LuxAlgo]
    Ported from PineScript v6
    """
    if len(df) < period + 2: return None

    src = (df['high'] + df['low']) / 2
    src_vals = src.values

    omega = 2 * np.pi / period
    alpha = np.tan(np.pi * delta / period)
    beta = np.cos(omega)
    r = 1.0 / (1.0 + alpha)

    c1 = 2 * r * beta
    c2 = -(2 * r - 1)
    gain = alpha * r

    bp = np.zeros(len(df))
    # Using a simple loop for the recursive part
    for i in range(2, len(df)):
        bp[i] = gain * (src_vals[i] - src_vals[i-2]) + c1 * bp[i-1] + c2 * bp[i-2]

    peak_lookback = max(1, int(period * lookback_mult))
    bp_abs = np.abs(bp)

    # Efficient highest over rolling window
    peak = pd.Series(bp_abs).rolling(window=peak_lookback).max().values

    oscillator = np.where(peak != 0, bp / peak, 0)
    signal_line = talib.EMA(oscillator, timeperiod=signal_len)

    return {
        "oscillator": float(oscillator[-1]),
        "signal_line": float(signal_line[-1]) if not np.isnan(signal_line[-1]) else None,
        "cross_up": bool(oscillator[-1] > signal_line[-1] and oscillator[-2] <= signal_line[-2] and oscillator[-1] < -0.8),
        "cross_down": bool(oscillator[-1] < signal_line[-1] and oscillator[-2] >= signal_line[-2] and oscillator[-1] > 0.8),
        "series": pd.Series(oscillator, index=df.index)
    }

def calculate_vwap_volume_profile(df: pd.DataFrame, period=250, bins=50):
    if len(df) < 5: return None
    p = min(period, len(df))
    df = df.copy()
    df['vwap'] = (df['close'] * df['volume']).cumsum() / df['volume'].cumsum()
    df['vol_signed'] = np.where(df['vwap'] > df['vwap'].shift(2), df['volume'], -df['volume'])
    recent = df.tail(p)
    H, L = recent['vwap'].max(), recent['vwap'].min()
    if H == L: return None
    step = (H - L) / bins
    vwap_vals, v1_vals, v2_vals = recent['vwap'].values, recent['vol_signed'].values, recent['volume'].values
    profile = []
    for i in range(bins):
        lb, hb = L + step * i, L + step * (i + 1)
        mask = (vwap_vals >= lb - step) & (vwap_vals <= hb + step)
        profile.append({"bin_low": float(lb), "bin_high": float(hb), "signed_vol": float(v1_vals[mask].sum()), "raw_vol": float(v2_vals[mask].sum())})
    vol1 = [p['signed_vol'] for p in profile]
    return {"profile": profile, "pos_poc": profile[np.argmax(vol1)], "neg_poc": profile[np.argmin(vol1)]}

def detect_ut_bot_alerts(df: pd.DataFrame, sensitivity=1.0, atr_period=10):
    if len(df) < atr_period + 1: return None
    c, h, l = df['close'].values, df['high'].values, df['low'].values
    atr = talib.ATR(h, l, c, timeperiod=atr_period)
    n_loss = sensitivity * atr
    ts = np.zeros(len(df))
    fv = next(i for i, v in enumerate(n_loss) if not np.isnan(v))
    ts[fv] = c[fv]
    for i in range(fv + 1, len(df)):
        if c[i] > ts[i-1] and c[i-1] > ts[i-1]: ts[i] = max(ts[i-1], c[i] - n_loss[i])
        elif c[i] < ts[i-1] and c[i-1] < ts[i-1]: ts[i] = min(ts[i-1], c[i] + n_loss[i])
        else: ts[i] = c[i] - n_loss[i] if c[i] > ts[i-1] else c[i] + n_loss[i]
    ts_s, ema_s = pd.Series(ts), pd.Series(c)
    cup = (ema_s > ts_s) & (ema_s.shift(1) <= ts_s.shift(1))
    cdn = (ts_s > ema_s) & (ts_s.shift(1) <= ema_s.shift(1))
    return {"buy": bool((ema_s > ts_s).iloc[-1] and cup.iloc[-1]), "sell": bool((ema_s < ts_s).iloc[-1] and cdn.iloc[-1]), "trailing_stop": float(ts[-1]), "series": pd.Series(ts, index=df.index)}

def calculate_mtf_macd_forecast(df: pd.DataFrame, fast=12, slow=26, sig=9, htf="4h", forecast_len=30):
    if len(df) < 50: return None
    df_copy = df.copy()
    if 'timestamp' in df_copy.columns: df_copy.index = pd.to_datetime(df_copy['timestamp'])
    try:
        tf = htf.replace("240", "4h").replace("60", "1h")
        htf_df = df_copy.resample(tf).last().dropna()
        h_macd, h_sig, _ = talib.MACD(htf_df['close'].values, fast, slow, sig)
        htf_trend = pd.Series(h_macd > h_sig, index=htf_df.index).reindex(df_copy.index, method='ffill').fillna(False)
    except: htf_trend = pd.Series([True] * len(df_copy), index=df_copy.index)
    c = df_copy['close'].values
    macd, signal, _ = talib.MACD(c, fast, slow, sig)
    ut = macd > signal
    mem = {True: {}, False: {}}
    ct, sp, off = None, 0, 0
    for i in range(len(df_copy)):
        if np.isnan(ut[i]): continue
        if ut[i] != ct: ct, sp, off = ut[i], c[i], 0
        else: off += 1
        if off not in mem[ct]: mem[ct][off] = []
        mem[ct][off].append(c[i] - sp)
        if len(mem[ct][off]) > 50: mem[ct][off].pop(0)
    lt = ut[-1]
    ao = 0
    for i in range(len(df_copy)-1, -1, -1):
        if ut[i] == lt: ao += 1
        else: break
    ao -= 1
    asp = c[-1] # This is simplified
    f = []
    for x in range(forecast_len):
        m_list = mem[lt].get(ao + x, [])
        up, mid, lo = (asp + np.percentile(m_list, 80), asp + np.percentile(m_list, 50), asp + np.percentile(m_list, 20)) if len(m_list) >= 3 else (None, None, None)
        f.append({"step": x, "upper": up, "mid": mid, "lower": lo})
    return {"htf_trend": "Bullish" if htf_trend.iloc[-1] else "Bearish", "current_trend": "Bullish" if lt else "Bearish", "forecast": f}

def detect_market_regime(df: pd.DataFrame):
    """
    Market Regime Detection Engine
    Classifies market state based on Trend, Volatility, and Momentum.
    """
    if len(df) < 30: return None
    c, h, l = df['close'].values, df['high'].values, df['low'].values

    # 1. Trend Analysis (ADX + EMA)
    adx = talib.ADX(h, l, c, timeperiod=14)
    ema20, ema50 = talib.EMA(c, 20), talib.EMA(c, 50)

    curr_adx = adx[-1]
    is_trending = curr_adx > 25
    is_strong_trend = curr_adx > 40

    trend_dir = "Neutral"
    if ema20[-1] > ema50[-1]: trend_dir = "Bullish"
    elif ema20[-1] < ema50[-1]: trend_dir = "Bearish"

    # 2. Volatility Analysis (ATR + BBWidth)
    atr = talib.ATR(h, l, c, timeperiod=14)
    rel_atr = (atr[-1] / c[-1]) * 100

    upper, mid, lower = talib.BBANDS(c, timeperiod=20)
    bb_width = (upper - lower) / mid
    curr_bbw = bb_width[-1]

    is_low_vol = curr_bbw < np.percentile(bb_width[~np.isnan(bb_width)], 25)
    is_high_vol = curr_bbw > np.percentile(bb_width[~np.isnan(bb_width)], 75)

    # 3. Momentum (RSI)
    rsi = talib.RSI(c, timeperiod=14)
    curr_rsi = rsi[-1]

    # Classification Logic
    regime = "Ranging"
    if is_trending:
        regime = f"Trending {trend_dir}"
        if is_strong_trend: regime = f"Strong {regime}"
    elif is_low_vol:
        regime = "Volatility Squeeze"
    elif is_high_vol:
        regime = "High Volatility Range"

    if curr_rsi > 70: regime += " (Overbought)"
    elif curr_rsi < 30: regime += " (Oversold)"

    return {
        "regime": regime,
        "trend_strength": float(curr_adx),
        "volatility_score": float(curr_bbw),
        "relative_atr_pct": float(rel_atr),
        "is_trending": bool(is_trending),
        "bias": trend_dir
    }

def calculate_vwma(series, volume, length):
    return (series * volume).rolling(length).sum() / volume.rolling(length).sum()

def detect_lux_zscore(df, length=144, smooth=20, history_depth=25, thresh=1.5):
    c = df['close']
    mean, std = c.rolling(length).mean(), c.rolling(length).std()
    z_score = calculate_vwma((c - mean) / std, df['volume'], smooth)
    return {"z_score": float(z_score.iloc[-1]), "signal": "Sell" if z_score.iloc[-1] > thresh else ("Buy" if z_score.iloc[-1] < -thresh else "Neutral"), "series": z_score}

def detect_lux_msb_ob(df, pivot_len=7, msb_thresh=0.5):
    h, l, c = df['high'].values, df['low'].values, df['close'].values
    change = df['close'].diff()
    bs = np.abs(df['close'] - df['open'])
    disp = bs / bs.rolling(20).mean()
    mz = (change - change.rolling(50).mean()) / change.rolling(50).std()
    ph, pl = argrelextrema(h, np.greater, order=pivot_len)[0], argrelextrema(l, np.less, order=pivot_len)[0]
    if len(ph) == 0 or len(pl) == 0: return {}
    is_msb_bull = c[-1] > h[ph[-1]] and (mz.iloc[-1] > msb_thresh or disp.iloc[-1] > 1.5)
    is_msb_bear = c[-1] < l[pl[-1]] and (mz.iloc[-1] < -msb_thresh or disp.iloc[-1] > 1.5)
    return {"msb": "Bullish" if is_msb_bull else ("Bearish" if is_msb_bear else "None"), "displacement": float(disp.iloc[-1])}

def detect_squeeze_momentum(df, bb_len=20, bb_mult=2.0, kc_len=20, kc_mult=1.5):
    c, h, l = df['close'], df['high'], df['low']
    basis, dev = talib.SMA(c, timeperiod=bb_len), kc_mult * talib.STDDEV(c, timeperiod=bb_len)
    upperBB, lowerBB = basis + dev, basis - dev
    ma, tr = talib.SMA(c, timeperiod=kc_len), talib.TRANGE(h, l, c)
    range_ma = talib.SMA(tr, timeperiod=kc_len)
    upperKC, lowerKC = ma + range_ma * kc_mult, ma - range_ma * kc_mult
    sqz_on = (lowerBB > lowerKC) & (upperBB < upperKC)
    avg_val = ((h.rolling(window=kc_len).max() + l.rolling(window=kc_len).min())/2 + ma) / 2
    momentum_val = talib.LINEARREG((c - avg_val).fillna(0), timeperiod=kc_len)
    return {"value": float(momentum_val.iloc[-1]), "state": "Squeeze On" if sqz_on.iloc[-1] else "Squeeze Off", "direction": "Up" if momentum_val.iloc[-1] > momentum_val.iloc[-2] else "Down", "series": momentum_val}

def detect_supertrend(df, period=10, multiplier=3.0):
    h, l, c = df['high'], df['low'], df['close']
    atr = talib.ATR(h, l, c, timeperiod=period)
    up, dn = (h + l) / 2 - (multiplier * atr), (h + l) / 2 + (multiplier * atr)
    upper, lower, trend = np.zeros(len(df)), np.zeros(len(df)), np.ones(len(df))
    for i in range(1, len(df)):
        if np.isnan(up[i]): continue
        lower[i] = max(up[i], lower[i-1]) if c.iloc[i-1] > lower[i-1] else up[i]
        upper[i] = min(dn[i], upper[i-1]) if c.iloc[i-1] < upper[i-1] else dn[i]
        trend[i] = 1 if c.iloc[i] > upper[i] else (-1 if c.iloc[i] < lower[i] else trend[i-1])
    st_val = pd.Series(np.where(trend == 1, lower, upper), index=df.index)
    return {"value": float(st_val.iloc[-1]), "direction": "Bullish" if trend[-1] == 1 else "Bearish", "series": st_val}

def detect_imba_trend(df, sensitivity=18.0):
    h, l, c = df['high'], df['low'], df['close']
    length = int(max(1, sensitivity * 10))
    imba = h.rolling(window=length).max() - (h.rolling(window=length).max() - l.rolling(window=length).min()) * 0.5
    return {"value": float(imba.iloc[-1]), "direction": "Bullish" if c.iloc[-1] > imba.iloc[-1] else "Bearish", "series": imba}

def calculate_portfolio_metrics(asset_series: Dict[str, pd.Series], market_returns: pd.Series = None):
    df = pd.DataFrame(asset_series).pct_change().dropna()
    metrics = {asset: {"sharpe_ratio": float((df[asset].mean()*252)/(df[asset].std()*np.sqrt(252)) if df[asset].std()!=0 else 0), "var_95": float(np.percentile(df[asset], 5))} for asset in asset_series}
    if market_returns is not None:
        m_ret = market_returns.pct_change()
        for asset in asset_series:
            m_df = pd.concat([df[asset], m_ret], axis=1).dropna()
            metrics[asset]["beta"] = float(np.cov(m_df.iloc[:,0], m_df.iloc[:,1])[0,1]/np.var(m_df.iloc[:,1]) if np.var(m_df.iloc[:,1])!=0 else 0)
    return {"correlation_matrix": df.corr().to_dict(), "asset_metrics": metrics}

def run_monte_carlo(returns, simulations=1000):
    """Shuffles returns to estimate probability distribution of outcomes."""
    if len(returns) == 0: return {}
    results = []
    for _ in range(simulations):
        path = np.random.choice(returns, size=len(returns), replace=True)
        results.append(np.sum(path))
    return {
        "mean_return": float(np.mean(results)),
        "std_dev": float(np.std(results)),
        "p5_value": float(np.percentile(results, 5)),
        "p95_value": float(np.percentile(results, 95))
    }

def backtest_strategy(df: pd.DataFrame, initial_capital=10000, strategy_type="EMA"):
    """
    Advanced Backtesting Engine with Statistics.
    """
    c = df['close'].values
    if strategy_type == "EMA":
        fast, slow = talib.EMA(c, 9), talib.EMA(c, 21)
        long_cond = (fast > slow) & (np.roll(fast, 1) <= np.roll(slow, 1))
        short_cond = (fast < slow) & (np.roll(fast, 1) >= np.roll(slow, 1))
    else:
        # Default to RSI mean reversion if unknown
        rsi = talib.RSI(c, 14)
        long_cond = (rsi < 30) & (np.roll(rsi, 1) >= 30)
        short_cond = (rsi > 70) & (np.roll(rsi, 1) <= 70)

    cap, pos, trades = initial_capital, 0, []
    returns = []

    for i in range(1, len(df)):
        if long_cond[i] and pos == 0:
            pos = cap / c[i]
            buy_p = c[i]
            cap = 0
            trades.append({"type": "buy", "price": float(buy_p), "index": i})
        elif short_cond[i] and pos > 0:
            profit = (pos * c[i]) - (pos * buy_p)
            returns.append(profit / (pos * buy_p))
            cap = pos * c[i]
            pos = 0
            trades.append({"type": "sell", "price": float(c[i]), "index": i, "profit": float(profit)})

    final_val = cap if pos == 0 else pos * c[-1]
    total_ret = ((final_val - initial_capital) / initial_capital)

    # Calculate Stats
    returns = np.array(returns)
    win_rate = np.sum(returns > 0) / len(returns) if len(returns) > 0 else 0
    profit_factor = np.sum(returns[returns > 0]) / abs(np.sum(returns[returns < 0])) if np.any(returns < 0) else float('inf')

    # Sharpe Ratio (annualized approx)
    sharpe = (np.mean(returns) / np.std(returns)) * np.sqrt(252) if len(returns) > 1 and np.std(returns) != 0 else 0

    # Monte Carlo
    mc = run_monte_carlo(returns) if len(returns) > 0 else {}

    return {
        "summary": {
            "initial_capital": float(initial_capital),
            "final_value": float(final_val),
            "total_return_pct": float(total_ret * 100),
            "win_rate": float(win_rate * 100),
            "profit_factor": float(profit_factor),
            "sharpe_ratio": float(sharpe),
            "trade_count": len(trades) // 2
        },
        "monte_carlo": mc,
        "trades": trades
    }

def discover_best_strategy(df: pd.DataFrame):
    """
    AI-lite Strategy Optimizer
    Tests common indicator combinations and returns the best performing rule-set.
    """
    strategies = ["EMA", "RSI", "MACD", "BBANDS"]
    regime = detect_market_regime(df)

    results = []
    for s in strategies:
        res = backtest_strategy(df, strategy_type=s)
        results.append({
            "strategy": s,
            "expectancy": res["summary"]["total_return_pct"] * res["summary"]["win_rate"],
            "metrics": res["summary"]
        })

    # Rank by expectancy
    ranked = sorted(results, key=lambda x: x["expectancy"], reverse=True)

    # Custom recommendation based on regime
    recommendation = "Maintain current strategy."
    if regime["regime"].startswith("Trending"):
        recommendation = "Trend-following strategies (EMA/MACD) are favored."
    elif regime["regime"] == "Ranging":
        recommendation = "Mean-reversion strategies (RSI/BBands) are favored."

    return {
        "top_strategies": ranked[:3],
        "market_context": regime["regime"],
        "recommendation": recommendation
    }

def walk_forward_backtest(df: pd.DataFrame, strategy_type="EMA", segments=4):
    """
    Splits data into IS/OOS segments and runs backtests.
    """
    if len(df) < 100: return None
    segment_size = len(df) // segments
    results = []

    for i in range(segments):
        start = i * segment_size
        end = (i + 1) * segment_size if i < segments - 1 else len(df)
        chunk = df.iloc[start:end]

        # Run backtest on chunk
        bt_res = backtest_strategy(chunk, strategy_type=strategy_type)
        results.append({
            "segment": i,
            "period_start": str(chunk.index[0]),
            "period_end": str(chunk.index[-1]),
            "return_pct": bt_res["summary"]["total_return_pct"],
            "win_rate": bt_res["summary"]["win_rate"]
        })

    avg_ret = np.mean([r["return_pct"] for r in results])
    consistency = 100 - np.std([r["return_pct"] for r in results])

    return {
        "segments": results,
        "average_segment_return": float(avg_ret),
        "stability_score": float(consistency)
    }

executor = ProcessPoolExecutor(max_workers=4)

def get_indicator_results_sync(df, selected=None, history=False):
    op, hi, lo, cl, vo = df['open'].values, df['high'].values, df['low'].values, df['close'].values, df['volume'].values
    sel = [s.upper() for s in selected] if selected else None

    talib_res = {}
    for f in talib.get_functions():
        try:
            func = getattr(talib, f)
            if sel and f.upper() not in sel: continue
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

    # Calculate advanced only if needed or all requested
    def is_req(k): return sel is None or k.upper() in sel

    st_res = detect_supertrend(df) if is_req("SUPERTREND") else None
    imba_res = detect_imba_trend(df) if is_req("IMBA_TREND") else None
    sqz_res = detect_squeeze_momentum(df) if is_req("SQUEEZE_MOMENTUM") else None
    resonator_res = calculate_normalized_resonator(df) if is_req("NORMALIZED_RESONATOR") else None
    zscore_res = detect_lux_zscore(df) if is_req("LUX_ZSCORE") else None
    ut_res = detect_ut_bot_alerts(df) if is_req("UT_BOT_ALERTS") else None

    if st_res: all_raw_cols["SUPERTREND"] = st_res["series"]
    if imba_res: all_raw_cols["IMBA_TREND"] = imba_res["series"]
    if sqz_res: all_raw_cols["SQUEEZE_MOMENTUM"] = sqz_res["series"]
    if resonator_res: all_raw_cols["NORMALIZED_RESONATOR"] = resonator_res["series"]
    if zscore_res: all_raw_cols["LUX_ZSCORE"] = zscore_res["series"]
    if ut_res: all_raw_cols["UT_BOT_TS"] = ut_res["series"]

    all_raw = pd.DataFrame(all_raw_cols, index=df.index)

    res = {
        "current_price": float(cl[-1]),
        "summary": {"trend": "Bullish" if cl[-1] > talib.EMA(cl, 200)[-1] else "Bearish", "session": get_sessions(df['timestamp'].iloc[-1])},
        "institutional_strategies": detect_smc(df) if is_req("SMC") or is_req("INSTITUTIONAL") or sel is None else {},
        "market_dynamics": {
            "levels": detect_sr_levels(df),
            "volume_profile": calculate_volume_profile(df),
            "vwap_volume_profile": calculate_vwap_volume_profile(df),
            "ut_bot_alerts": ut_res if ut_res else {}
        },
        "divergences": {"rsi": detect_divergences(df, "RSI"), "macd": detect_divergences(df, "MACD")},
        "price_action_patterns": detect_price_action_patterns(df),
        "custom_lux_algo": {"zscore_zones": {k:v for k,v in zscore_res.items() if k != "series"} if zscore_res else {}, "market_structure": detect_lux_msb_ob(df)},
        "squeeze_momentum": {k:v for k,v in sqz_res.items() if k != "series"} if sqz_res else {},
        "cycle_indicators": {k:v for k,v in resonator_res.items() if k != "series"} if resonator_res else {},
        "trend_following": {
            "supertrend": {k:v for k,v in st_res.items() if k != "series"} if st_res else {},
            "imba_trend": {k:v for k,v in imba_res.items() if k != "series"} if imba_res else {},
            "trend_pro_z": calculate_trend_pro_z(df) if is_req("TREND_PRO_Z") or sel is None else {},
            "ichimoku": calculate_ichimoku(df) if is_req("ICHIMOKU") or sel is None else {}
        },
        "forecasting_models": {
            "mtf_macd_forecast": calculate_mtf_macd_forecast(df) if is_req("MTF_MACD_FORECAST") or sel is None else {},
            "harmonic_forecast": calculate_adaptive_harmonic_forecast(df) if is_req("HARMONIC_FORECAST") or sel is None else {}
        },
        "market_intelligence": {
            "regime": detect_market_regime(df),
            "liquidity_heatmap": calculate_liquidity_heatmap(df),
            "strategy_discovery": discover_best_strategy(df)
        }
    }

    res["indicators"] = all_raw.iloc[-1].to_dict()
    if history:
        h_df = df.copy()
        for col in all_raw.columns:
            if col not in h_df.columns: h_df[col] = all_raw[col]
        res["history"] = h_df.tail(1000).to_dict(orient="records")
    return res

async def get_indicator_results(df, selected=None, history=False):
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(executor, get_indicator_results_sync, df, selected, history)
