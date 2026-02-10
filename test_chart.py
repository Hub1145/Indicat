import requests
import time
import os

# Start server in background
os.system("kill $(lsof -t -i :8000) 2>/dev/null || true")
os.system("python main.py > server.log 2>&1 &")
time.sleep(5)

# Try to get the chart
try:
    url = "http://localhost:8000/analyze/chart?symbol=BTC/USDT&timeframe=1d&indicators=EMA20,EMA50,EMA200,RSI,MACD,SUPERTREND,IMBA_TREND,SQUEEZE_MOMENTUM"
    response = requests.get(url)
    if response.status_code == 200:
        with open("chart.html", "w") as f:
            f.write(response.text)
        print("Chart HTML saved to chart.html")
    else:
        print(f"Failed to get chart: {response.status_code}")
        print(response.text)
except Exception as e:
    print(f"Error: {e}")

# Kill server
os.system("kill $(lsof -t -i :8000) 2>/dev/null || true")
