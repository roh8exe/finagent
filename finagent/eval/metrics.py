# finagent/eval/metrics.py
import numpy as np

TRADING_DAYS = 252

def arr(V: np.ndarray):
    V0, VT = V[0], V[-1]
    T = len(V) - 1
    return ((VT - V0) / V0) * (TRADING_DAYS / max(T,1))

def returns(V: np.ndarray):
    return (V[1:] - V[:-1]) / V[:-1]

def sharpe(V):
    r = returns(V)
    return (np.mean(r) / (np.std(r) + 1e-12)) if len(r)>1 else 0.0

def volatility(V):
    r = returns(V); return np.std(r)

def max_drawdown(V):
    R = np.cumprod(1 + returns(V))
    P = np.maximum.accumulate(R)
    dd = (P - R) / (P + 1e-12)
    return np.max(dd) if len(dd) else 0.0

def calmar(V):
    r = returns(V); mdd = max_drawdown(V)
    return (np.mean(r) / (mdd + 1e-12)) if mdd>0 else np.inf

def sortino(V):
    r  = returns(V)
    dd = np.std(np.clip(r, a_max=0, a_min=None))  # downside stdev
    return (np.mean(r) / (dd + 1e-12)) if dd>0 else np.inf
