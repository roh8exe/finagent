from finagent.embed.encoder import TextEncoder
from finagent.embed.store import VectorStore, MemoryItem
import numpy as np

enc = TextEncoder("sentence-transformers/all-MiniLM-L6-v2")
vec = enc.embed("test query")
store = VectorStore(dim=vec.shape[1])

item = MemoryItem(kind="MI", date="2023-01-01", ticker="AAPL",
                  summary="dummy", query="test query", extras={})
store.add(vec, [item])

res = store.search(vec, k=1)
print(res[0][0])
