import requests
import time

def test_upload():
    print("Testing /analyze/upload...")
    # Generate dummy data
    data = []
    price = 100.0
    for i in range(100):
        price += 0.1
        data.append({
            "timestamp": str(i),
            "open": price, "high": price+1, "low": price-1, "close": price, "volume": 1000
        })

    resp = requests.post("http://127.0.0.1:8000/analyze/upload", json={"data": data})
    res = resp.json()
    print(res)
    if "price_action" in res:
        print("Success: found price_action field")

def test_market():
    print("\nTesting /analyze/market (Crypto)...")
    payload = {
        "provider": "crypto",
        "symbol": "BTC/USD",
        "timeframe": "1d"
    }
    start = time.time()
    resp = requests.post("http://127.0.0.1:8000/analyze/market", json=payload)
    print(f"Time taken (Fresh): {time.time() - start:.2f}s")
    print(resp.json())

    print("\nTesting Cache...")
    start = time.time()
    resp = requests.post("http://127.0.0.1:8000/analyze/market", json=payload)
    print(f"Time taken (Cached): {time.time() - start:.2f}s")

def test_filtering():
    print("\nTesting Indicator Filtering...")
    payload = {
        "provider": "crypto",
        "symbol": "BTC/USD",
        "timeframe": "1d",
        "exchange": "kraken",
        "indicators": ["talib_RSI", "volatility_bbh"]
    }
    resp = requests.post("http://127.0.0.1:8000/analyze/market", json=payload)
    res = resp.json()
    print("Selected Indicators:", res.get("selected_indicators"))
    if "selected_indicators" in res and "talib_RSI" in res["selected_indicators"]:
        print("Success: only selected indicators returned")

def test_scan_patterns():
    print("\nTesting /scan-patterns...")
    payload = {
        "provider": "crypto",
        "symbol": "BTC/USD",
        "timeframe": "1d",
        "exchange": "kraken"
    }
    resp = requests.post("http://127.0.0.1:8000/scan-patterns", json=payload)
    print(f"Patterns found: {len(resp.json().get('patterns_found', []))}")

def test_is_trend_bullish():
    print("\nTesting /is-trend-bullish...")
    payload = {
        "provider": "crypto",
        "symbol": "BTC/USD",
        "timeframe": "1d",
        "exchange": "kraken"
    }
    resp = requests.post("http://127.0.0.1:8000/is-trend-bullish", json=payload)
    print(resp.json())

def test_confluence():
    print("\nTesting /confluence-score...")
    payload = {
        "provider": "crypto",
        "symbol": "BTC/USD",
        "timeframe": "1d",
        "exchange": "kraken"
    }
    resp = requests.post("http://127.0.0.1:8000/confluence-score", json=payload)
    print(resp.json())

def test_stock():
    print("\nTesting /analyze/market (Stock)...")
    payload = {
        "provider": "stock",
        "symbol": "AAPL",
        "timeframe": "1d"
    }
    resp = requests.post("http://127.0.0.1:8000/analyze/market", json=payload)
    print(resp.json())

if __name__ == "__main__":
    # Ensure server is running
    try:
        test_upload()
        test_market()
        test_filtering()
        test_scan_patterns()
        test_is_trend_bullish()
        test_confluence()
        test_stock()
    except Exception as e:
        print(f"Error: {e}")
