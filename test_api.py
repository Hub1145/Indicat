import requests
import json
import random

def generate_dummy_data(n=100):
    data = []
    price = 100.0
    for i in range(n):
        price += random.uniform(-1, 1)
        data.append({
            "timestamp": f"2023-01-01 {i:02d}:00:00",
            "open": price,
            "high": price + random.uniform(0, 0.5),
            "low": price - random.uniform(0, 0.5),
            "close": price + random.uniform(-0.2, 0.2),
            "volume": random.uniform(100, 1000)
        })
    return data

def test_analyze():
    print("Testing /analyze...")
    data = generate_dummy_data(200)
    payload = {
        "data": data,
        "include_patterns": True,
        "include_indicators": True
    }
    response = requests.post("http://127.0.0.1:8000/analyze", json=payload)
    if response.status_code == 200:
        results = response.json()
        print(f"Success! Received {len(results)} candles.")
        print("Sample result keys:", results[0].keys())
        # Check for some expected indicators
        expected = ['RSI', 'MACD', 'BB_upper', 'EMA_20', 'Supertrend', 'VWAP', 'signal_score']
        for key in expected:
            if key in results[0]:
                print(f"Found {key}: {results[0][key]}")
            else:
                print(f"MISSING {key}!")
    else:
        print(f"Failed! Status code: {response.status_code}, Detail: {response.text}")

def test_scan_patterns():
    print("\nTesting /scan-patterns...")
    data = generate_dummy_data(100)
    payload = {
        "data": data
    }
    response = requests.post("http://127.0.0.1:8000/scan-patterns", json=payload)
    if response.status_code == 200:
        results = response.json()
        print(f"Success! Found {len(results['patterns_found'])} patterns.")
        if results['patterns_found']:
            print("First pattern found:", results['patterns_found'][0])
    else:
        print(f"Failed! Status code: {response.status_code}, Detail: {response.text}")

def test_is_trend_bullish():
    print("\nTesting /is-trend-bullish...")
    data = generate_dummy_data(250)
    payload = {
        "data": data
    }
    response = requests.post("http://127.0.0.1:8000/is-trend-bullish", json=payload)
    if response.status_code == 200:
        results = response.json()
        print(f"Success! Bullish: {results['bullish']}")
    else:
        print(f"Failed! Status code: {response.status_code}, Detail: {response.text}")

if __name__ == "__main__":
    test_analyze()
    test_scan_patterns()
    test_is_trend_bullish()
