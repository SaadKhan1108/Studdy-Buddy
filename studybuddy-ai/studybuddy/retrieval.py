"""Keyword + phrase search over chunks (TF-IDF). Light enough for free hosting."""
from __future__ import annotations

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer


class Index:
    def __init__(self, chunks):
        self.chunks = [c for c in chunks if c.searchable] or list(chunks)  # covers, indexes etc. are skipped
        self.pos = {c.id: j for j, c in enumerate(self.chunks)}
        self.vec = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), sublinear_tf=True)
        self.matrix = self.vec.fit_transform([f"{c.heading} {c.text}" for c in self.chunks])

    def search(self, query: str, k: int = 6, allowed=None):
        """Return the k most relevant chunks. `allowed` = optional set of chunk ids."""
        q = self.vec.transform([query])
        scores = (self.matrix @ q.T).toarray().ravel()  # rows are L2-normalised -> cosine
        if allowed is not None:
            mask = np.zeros(len(scores), dtype=bool)
            mask[[self.pos[i] for i in allowed if i in self.pos]] = True
            scores = np.where(mask, scores, 0.0)
        order = np.argsort(-scores)[:k]
        return [self.chunks[i] for i in order if scores[i] > 0]
