import numpy as np
import pandas as pd

def MACD(close, fast=12, slow=26, signal=9):
    ema_fast = close.ewm(span=fast, adjust=False).mean()
    ema_slow = close.ewm(span=slow, adjust=False).mean()
    macd = ema_fast - ema_slow
    sig  = macd.ewm(span=signal, adjust=False).mean()
    hist = macd - sig
    return pd.DataFrame({"macd": macd, "signal": sig, "hist": hist})

def RSI(close, period=14):
    delta = close.diff()
    gain  = delta.clip(lower=0).rolling(period).mean()
    loss  = (-delta.clip(upper=0)).rolling(period).mean()
    rs = gain / (loss + 1e-9)
    rsi = 100 - (100 / (1 + rs))
    return rsi

def KDJ(hlc, n=9, k=3, d=3):
    low_min  = hlc["Low"].rolling(n).min()
    high_max = hlc["High"].rolling(n).max()
    rsv = (hlc["Close"] - low_min) / (high_max - low_min + 1e-9) * 100
    k_val = rsv.ewm(com=(k-1)/2, adjust=False).mean()
    d_val = k_val.ewm(com=(d-1)/2, adjust=False).mean()
    j_val = 3 * k_val - 2 * d_val
    return pd.DataFrame({"K": k_val, "D": d_val, "J": j_val})

def zscore_mean_reversion(close, window=20):
    ma = close.rolling(window).mean()
    sd = close.rolling(window).std()
    z = (close - ma) / (sd + 1e-9)
    return pd.DataFrame({"ma": ma, "z": z})
