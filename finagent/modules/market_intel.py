# finagent/modules/market_intel.py
import json, numpy as np
from datetime import datetime
from ..embed.encoder import TextEncoder
from ..embed.store import MemoryItem
from ..dataio.news import fetch_news
from ..dataio.charts import render_kline
from ..indicators.ta import MACD, RSI, KDJ, zscore_mean_reversion

class MarketIntelligence:
    def __init__(self, llm, encoder, mem_store, prompts, chart_dir="data/charts"):
        self.llm = llm
        self.enc = encoder
        self.mem = mem_store
        self.prompts = prompts
        self.chart_dir = chart_dir

    def latest(self, ticker, dt, df_window):
        # Build basic context = last prices + few news lines
        news = fetch_news(ticker, dt)
        user = f"Ticker: {ticker}\nDate: {dt.date()}\nNews:\n- " + "\n- ".join(news)
        system = open(self.prompts["mi"]).read()
        out = self.llm.chat(system, user)
        # Expect 'analysis:', 'summary:', 'query:' blocks (or JSON if you prefer).
        return out

    def diversified_retrieve(self, ticker, mi_out):
        # Extract the 'query' line(s) into a small list
        queries = []
        for line in mi_out.splitlines():
            if line.lower().startswith("query"):
                q = line.split(":",1)[-1].strip()
                queries = [x.strip() for x in q.split(",") if x.strip()]
        if not queries: return []
        vecs = self.enc.embed(queries)
        res  = self.mem.search(vecs, k=3)
        # Flatten & dedupe by date/ticker/kind
        flat = []
        seen = set()
        for group in res:
            for item in group:
                key = (item.date, item.ticker, item.kind, item.summary[:40])
                if key not in seen:
                    seen.add(key); flat.append(item)
        return flat

    def write_to_memory(self, ticker, dt, mi_out):
        # Derive compact 'summary' and 'query' strings from output
        summary = []
        query = ""
        for line in mi_out.splitlines():
            ll = line.strip()
            if ll.lower().startswith("summary"):
                summary.append(ll.split(":",1)[-1].strip())
            if ll.lower().startswith("query"):
                query = ll.split(":",1)[-1].strip()
        text_for_embed = [query or ""]  # retrieval-oriented
        vecs = np.array(self.enc.embed(text_for_embed))
        item = MemoryItem(kind="MI", date=str(dt.date()), ticker=ticker,
                          summary="\n".join(summary), query=query, extras={})
        self.mem.add(vecs, [item])
        return item
