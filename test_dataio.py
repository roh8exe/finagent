# test_dataio.py
from finagent.dataio.prices import load_prices
from finagent.dataio.charts import render_kline
from finagent.indicators.ta import MACD, RSI, KDJ, zscore_mean_reversion

# 1️⃣  Load and display data
df = load_prices("AAPL", "2023-01-01", "2023-02-01")
print(df.head())

# 2️⃣  Plot K-line chart
render_kline(df, "data/charts/AAPL_test.png", "AAPL test")
print("Chart saved!")

# 3️⃣  Compute indicators on the Close column
macd = MACD(df["Close"])
rsi  = RSI(df["Close"])
kdj  = KDJ(df[["High", "Low", "Close"]])
z    = zscore_mean_reversion(df["Close"])

# 4️⃣  Show last values
print("\nLatest indicators:")
print("MACD:", macd.tail(1))
print("RSI:",  rsi.tail(1))
print("KDJ:",  kdj.tail(1))
print("Z-score:", z.tail(1))
