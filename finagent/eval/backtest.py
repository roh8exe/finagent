# finagent/eval/backtest.py
import numpy as np, pandas as pd
from dataclasses import dataclass

@dataclass
class Position:
    qty: float = 0.0
    cash: float = 0.0
    last_price: float = 0.0

def run_backtest(df, decisions, initial_cash=100_000, commission_bps=1):
    pos = Position(qty=0, cash=initial_cash, last_price=float(df["Close"].iloc[0]))
    equity = []
    for i, (dt, row) in enumerate(df.iterrows()):
        price = float(row["Close"])
        act = decisions.get(dt, "HOLD")
        fee_mult = 1 + commission_bps/10_000.0

        if act == "BUY" and pos.cash > 0:
            qty = pos.cash / (price*fee_mult)
            pos.qty += qty
            pos.cash = 0.0
        elif act == "SELL" and pos.qty > 0:
            proceeds = pos.qty * price / fee_mult
            pos.cash += proceeds
            pos.qty = 0.0

        pos.last_price = price
        equity.append(pos.cash + pos.qty * price)
    return np.array(equity), pd.Series(equity, index=df.index)
