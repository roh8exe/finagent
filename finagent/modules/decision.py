# finagent/modules/decision.py
import json
from ..indicators.ta import MACD, RSI, KDJ, zscore_mean_reversion

class Decision:
    def __init__(self, llm, prompts, trader_pref="balanced"):
        self.llm = llm
        self.prompts = prompts
        self.pref = trader_pref

    def tool_signals(self, df):
        macd = MACD(df["Close"])
        rsi  = RSI(df["Close"])
        kdj  = KDJ(df[["High","Low","Close"]])
        z    = zscore_mean_reversion(df["Close"])
        latest = {
            "macd_cross": "bull" if macd["macd"].iloc[-1] > macd["signal"].iloc[-1] else "bear",
            "rsi": float(rsi.iloc[-1]),
            "kdj_j": float(kdj["J"].iloc[-1]),
            "z": float(z["z"].iloc[-1]),
        }
        return latest

    def run(self, ticker, dt, latest_mi_summary, llr_reasoning, hlr_summary, df, use_tools=True):
        tools = self.tool_signals(df) if use_tools else {}
        user = f"""Ticker: {ticker}
Date: {dt.date()}

MI summary:
{latest_mi_summary}

LLR reasoning:
{llr_reasoning}

HLR lessons:
{hlr_summary}

Trader preference: {self.pref}
Tool signals: {json.dumps(tools)}

Return JSON {{analysis, reasoning, action}} with action in ["BUY","SELL","HOLD"].
"""
        system = open(self.prompts["decision"]).read()
        out = self.llm.chat(system, user)
        try:
            js = json.loads(out)
        except:
            js = {"analysis": out, "reasoning": out, "action": "HOLD"}
        return js, tools
