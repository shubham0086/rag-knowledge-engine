"""
Reranker: cross-encoder scoring of (query, passage) pairs.

Uses ms-marco-MiniLM-L6-v2 — lightweight, fast, strong at passage ranking.
After hybrid retrieval fetches ~50 candidates, the cross-encoder rescores
each (query, passage) pair jointly and returns the highest-quality top_k.
Unlike bi-encoder cosine similarity, cross-encoder attention sees both
query and passage together, giving much better relevance judgement.
"""
from typing import List, Dict

from sentence_transformers import CrossEncoder

RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"


class Reranker:
    def __init__(self, model: str = RERANK_MODEL):
        self.model = CrossEncoder(model)

    def rerank(self, query: str, candidates: List[Dict], top_k: int = 5) -> List[Dict]:
        """Score each (query, passage) pair; return top_k sorted by cross-encoder score."""
        if not candidates:
            return []
        pairs = [(query, c["text"]) for c in candidates]
        scores = self.model.predict(pairs)
        ranked = sorted(zip(scores, candidates), key=lambda x: x[0], reverse=True)
        result = []
        for score, candidate in ranked[:top_k]:
            out = candidate.copy()
            out["rerank_score"] = round(float(score), 4)
            result.append(out)
        return result
