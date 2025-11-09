# finagent/modules/high_reflection.py
import json, numpy as np
from ..embed.store import MemoryItem

class HighReflection:
    def __init__(self, llm, encoder, mem_store, prompts):
        self.llm = llm
        self.enc = encoder
        self.mem = mem_store
        self.prompts = prompts

    def run(self, ticker, dt, recent_decisions, trading_chart_img, equity_curve_stats):
        # Build compact action log for last N days
        action_log = "\n".join([f"{d['date']}: {d['action']} - {d.get('reason','')}" for d in recent_decisions])
        user = f"""Ticker: {ticker}
Date: {dt.date()}
Recent decisions:
{action_log}

Equity curve: {equity_curve_stats}
Attached trading chart image: {trading_chart_img}
Return JSON {{evaluation, improvement, summary, query}}.
"""
        system = open(self.prompts["hlr"]).read()
        out = self.llm.chat(system, user, images=[trading_chart_img])
        try:
            js = json.loads(out)
        except:
            js = {"evaluation": out, "improvement": "", "summary": out[:300], "query": f"{ticker} decision lessons {dt.date()}"}
        vec = np.array(self.enc.embed([js.get("query","")]))
        item = MemoryItem(kind="HLR", date=str(dt.date()), ticker=ticker,
                          summary=js.get("summary",""),
                          query=js.get("query",""), extras={"evaluation":js.get("evaluation",""),
                                                            "improvement":js.get("improvement","")})
        self.mem.add(vec, [item])
        return js, item
