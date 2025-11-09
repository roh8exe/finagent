# finagent/modules/low_reflection.py
import json, numpy as np
from ..embed.store import MemoryItem

class LowReflection:
    def __init__(self, llm, encoder, mem_store, prompts):
        self.llm = llm
        self.enc = encoder
        self.mem = mem_store
        self.prompts = prompts

    def run(self, ticker, dt, latest_mi_summary, retrieved_mis, kline_img_path):
        # Build user context: latest summary + N retrieved summaries
        past = "\n\n".join([f"- {m.date} {m.ticker}: {m.summary}" for m in retrieved_mis[:5]])
        user = f"""Ticker: {ticker}
Date: {dt.date()}
Latest MI summary:
{latest_mi_summary}

Retrieved MI (past patterns):
{past}

Attached: K-line image URL -> {kline_img_path}
Return JSON with keys: reasoning, query
"""
        system = open(self.prompts["llr"]).read()
        out = self.llm.chat(system, user, images=[kline_img_path])
        try:
            js = json.loads(out)
        except:
            js = {"reasoning": out, "query": f"{ticker} price-move reflection {dt.date()}"}
        # Write a retrievable item
        vec = np.array(self.enc.embed([js.get("query","")]))
        item = MemoryItem(kind="LLR", date=str(dt.date()), ticker=ticker,
                          summary=js.get("reasoning",""),
                          query=js.get("query",""), extras={})
        self.mem.add(vec, [item])
        return js, item
