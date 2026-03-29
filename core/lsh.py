import numpy as np
from collections import defaultdict

class LSHIndex:
    def __init__(self, dim=64, n_tables=10, n_funcs=4, w=1.0):
        self.w = w
        self.tables = []
        for _ in range(n_tables):
            a = np.random.randn(n_funcs, dim)
            b = np.random.uniform(0, w, n_funcs)
            self.tables.append((a, b, defaultdict(list)))

    def _hash(self, a, b, z):
        return tuple(((a @ z + b) / self.w).astype(int))

    def index(self, embeddings):
        for i, z in enumerate(embeddings):
            for a, b, table in self.tables:
                table[self._hash(a, b, z)].append(i)

    def query(self, z_q, embeddings, k=5):
        candidates = set()
        for a, b, table in self.tables:
            bucket = self._hash(a, b, z_q)
            candidates.update(table.get(bucket, []))

        candidates = list(candidates)
        if len(candidates) == 0:
            return []

        dists = np.linalg.norm(embeddings[candidates] - z_q, axis=1)
        top_k = np.array(candidates)[np.argsort(dists)[:k]]
        return top_k
