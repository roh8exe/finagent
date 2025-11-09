import faiss
import numpy as np
from dataclasses import dataclass

@dataclass
class MemoryItem:
    kind: str      # "MI" | "LLR" | "HLR"
    date: str
    ticker: str
    summary: str
    query: str
    extras: dict

class VectorStore:
    def __init__(self, dim):
        self.index = faiss.IndexFlatIP(dim)
        self.items = []
        self.vecs = None

    def add(self, vectors: np.ndarray, items):
        if self.vecs is None:
            self.vecs = vectors.astype("float32")
        else:
            self.vecs = np.vstack([self.vecs, vectors.astype("float32")])
        self.index.add(vectors.astype("float32"))
        self.items.extend(items)

    def search(self, vectors: np.ndarray, k=5):
        D, I = self.index.search(vectors.astype("float32"), k)
        results = []
        for ids in I:
            results.append([self.items[i] for i in ids if i >= 0])
        return results
