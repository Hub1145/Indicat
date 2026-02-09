import requests
import time
import random

BASE_URL = "http://127.0.0.1:8000"

def test_endpoints():
    # 1. Indicators Discovery
    print("Testing /indicators...")
    r = requests.get(f"{BASE_URL}/indicators")
    print(f"Status: {r.status_code}, Found {len(r.json().get('technical_indicators', []))} indicators")

    # 2. Analyze Upload
    print("\nTesting /analyze/upload...")
    data = [{'open': 100+i, 'high': 110+i, 'low': 90+i, 'close': 105+i, 'volume': 1000} for i in range(100)]
    r = requests.post(f"{BASE_URL}/analyze/upload", json={"data": data, "include_history": True, "indicators": ["RSI", "MACD"]})
    print(f"Status: {r.status_code}, History size: {len(r.json().get('history', []))}")

    # 3. Analyze Market (Crypto)
    print("\nTesting /analyze/market (Kraken)...")
    payload = {"provider": "crypto", "symbol": "BTC/USD", "timeframe": "1d", "exchange": "kraken"}

    start = time.time()
    r = requests.post(f"{BASE_URL}/analyze/market", json=payload)
    t1 = time.time() - start
    print(f"Status: {r.status_code}, Current Price: {r.json().get('current_price')}, Time: {t1:.2f}s")

    # Test Caching
    start = time.time()
    r = requests.post(f"{BASE_URL}/analyze/market", json=payload)
    t2 = time.time() - start
    print(f"Cache Test Time: {t2:.4f}s (Should be < 0.1s)")

    # Check for Price Action Patterns
    data = r.json()
    print(f"Price Action Patterns Found: {[p['pattern'] for p in data.get('price_action_patterns', [])]}")

    # 4. Confluence Score
    print("\nTesting /confluence-score...")
    r = requests.post(f"{BASE_URL}/confluence-score", json=payload)
    print(f"Status: {r.status_code}, Score: {r.json().get('score')}, Sentiment: {r.json().get('sentiment')}")

    # 5. Multi-Timeframe
    print("\nTesting /analyze/mtf...")
    mtf_payload = {"provider": "crypto", "symbol": "BTC/USD", "timeframes": ["1h", "1d"]}
    r = requests.post(f"{BASE_URL}/analyze/mtf", json=mtf_payload)
    print(f"Status: {r.status_code}, Timeframes: {list(r.json().get('timeframes', {}).keys())}")

    # 6. Correlation
    print("\nTesting /analyze/correlation...")
    corr_payload = {"assets": ["BTC/USD", "ETH/USD"], "provider": "crypto"}
    r = requests.post(f"{BASE_URL}/analyze/correlation", json=corr_payload)
    print(f"Status: {r.status_code}, Correlation: {r.json()}")

    # 7. Heatmap
    print("\nTesting /analyze/heatmap...")
    heatmap_payload = {"assets": ["BTC/USD", "ETH/USD"], "metric": "RSI"}
    r = requests.post(f"{BASE_URL}/analyze/heatmap", json=heatmap_payload)
    print(f"Status: {r.status_code}, Data: {r.json()}")

    # 8. Options Greeks
    print("\nTesting /options/greeks...")
    greeks_payload = {"underlying_price": 60000, "strike": 62000, "expiry": "2025-12-31", "volatility": 0.5}
    r = requests.post(f"{BASE_URL}/options/greeks", json=greeks_payload)
    print(f"Status: {r.status_code}, Delta: {r.json().get('delta')}")

    # 9. Bullish Check
    print("\nTesting /is-trend-bullish...")
    r = requests.post(f"{BASE_URL}/is-trend-bullish", json=payload)
    print(f"Status: {r.status_code}, Bullish: {r.json().get('bullish')}")

    # 10. Pattern Scan
    print("\nTesting /scan-patterns...")
    r = requests.post(f"{BASE_URL}/scan-patterns", json=payload)
    print(f"Status: {r.status_code}, Patterns: {len(r.json().get('patterns', []))}")

if __name__ == "__main__":
    test_endpoints()
