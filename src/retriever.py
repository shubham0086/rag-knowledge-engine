"""
Retriever: takes a query and fetches the most relevant chunks from Qdrant.

v2 upgrade:
  - Hybrid search: vector (Qdrant cosine) + BM25 keyword, fused via RRF.
    BM25 catches exact-keyword queries that pure vector search misses.
    Vector catches semantic matches that keyword search misses.
    RRF (Reciprocal Rank Fusion) combines both ranked lists without needing
    score normalisation — only rank positions matter.
  - Reranking: cross-encoder rescores the top-50 candidate pool to top_k.
    Cross-encoder sees query + passage jointly, giving far better relevance
    judgement than the bi-encoder cosine used during retrieval.

BM25 index is built lazily on first hybrid search by scrolling all Qdrant
points. Rebuild by setting retriever._bm25 = None.
"""
from typing import Dict, List

from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer

from .ingestor import COLLECTION, EMBED_MODEL
from .reranker import Reranker

TOP_K_DEFAULT  = 5
FETCH_K_DEFAULT = 50   # candidate pool size before reranking


class Retriever:
    def __init__(
        self,
        qdrant_url: str = "http://localhost:6333",
        enable_rerank: bool = True,
    ):
        self.client   = QdrantClient(url=qdrant_url)
        self.embedder = SentenceTransformer(EMBED_MODEL)
        self.reranker = Reranker() if enable_rerank else None
        self._bm25              = None
        self._bm25_corpus: List[Dict] = []

    # ------------------------------------------------------------------
    # BM25 index — built lazily by scrolling all Qdrant payloads
    # ------------------------------------------------------------------

    def _build_bm25(self) -> None:
        from rank_bm25 import BM25Okapi
        self._bm25_corpus = []
        offset = None
        while True:
            records, offset = self.client.scroll(
                collection_name=COLLECTION,
                offset=offset,
                limit=200,
                with_payload=True,
            )
            for r in records:
                self._bm25_corpus.append({
                    "text":  r.payload.get("text", ""),
                    "file":  r.payload.get("file", "unknown"),
                    "chunk": r.payload.get("chunk_idx", 0),
                    "score": 0.0,
                })
            if offset is None:
                break
        tokenized = [doc["text"].lower().split() for doc in self._bm25_corpus]
        self._bm25 = BM25Okapi(tokenized)

    # ------------------------------------------------------------------
    # Reciprocal Rank Fusion — no score normalisation needed
    # ------------------------------------------------------------------

    @staticmethod
    def _rrf(vec_hits: List[Dict], bm25_hits: List[Dict], rrf_k: int = 60) -> List[Dict]:
        """Fuse two ranked lists via RRF. A doc appearing in both lists scores higher."""
        def key(doc: Dict):
            return (doc["file"], doc["chunk"])

        rrf_scores: Dict = {}
        for rank, doc in enumerate(vec_hits):
            k = key(doc)
            rrf_scores[k] = rrf_scores.get(k, 0.0) + 1.0 / (rrf_k + rank + 1)
        for rank, doc in enumerate(bm25_hits):
            k = key(doc)
            rrf_scores[k] = rrf_scores.get(k, 0.0) + 1.0 / (rrf_k + rank + 1)

        seen: Dict = {}
        for doc in vec_hits + bm25_hits:
            k = key(doc)
            if k not in seen:
                seen[k] = doc

        return sorted(seen.values(), key=lambda d: rrf_scores.get(key(d), 0.0), reverse=True)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def search(
        self,
        query: str,
        top_k: int = TOP_K_DEFAULT,
        hybrid: bool = True,
        rerank: bool = True,
        fetch_k: int = FETCH_K_DEFAULT,
    ) -> List[Dict]:
        """
        Retrieve top_k chunks for query.

        hybrid=True  → fuse BM25 keyword + vector results via RRF
        rerank=True  → cross-encoder rescore the candidate pool
        fetch_k      → candidate pool size fed into the reranker
        """
        vector = self.embedder.encode(query).tolist()
        raw = self.client.search(
            collection_name=COLLECTION,
            query_vector=vector,
            limit=fetch_k if rerank else top_k,
            with_payload=True,
        )
        vec_hits = [
            {
                "score": round(hit.score, 4),
                "file":  hit.payload.get("file", "unknown"),
                "chunk": hit.payload.get("chunk_idx", 0),
                "text":  hit.payload.get("text", ""),
            }
            for hit in raw
        ]

        if hybrid:
            if self._bm25 is None:
                self._build_bm25()
            bm25_index = self._bm25
            assert bm25_index is not None
            bm25_scores = bm25_index.get_scores(query.lower().split())
            bm25_ranked = sorted(
                enumerate(bm25_scores), key=lambda x: x[1], reverse=True
            )[:fetch_k]
            bm25_hits = [
                {**self._bm25_corpus[i], "score": round(float(s), 4)}
                for i, s in bm25_ranked if s > 0
            ]
            candidates = self._rrf(vec_hits, bm25_hits)[:fetch_k]
        else:
            candidates = vec_hits

        if rerank and self.reranker and candidates:
            return self.reranker.rerank(query, candidates, top_k=top_k)

        return candidates[:top_k]

    def format_context(self, results: List[Dict]) -> str:
        parts = []
        for i, r in enumerate(results, 1):
            score = r.get("rerank_score", r.get("score", 0))
            parts.append(
                f"[{i}] {r['file']} (chunk {r['chunk']}, score {score}):\n{r['text']}"
            )
        return "\n\n".join(parts)
