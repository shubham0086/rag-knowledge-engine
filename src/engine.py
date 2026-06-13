"""
RAGEngine: the public interface.
Wraps ingestor, retriever, answerer, and evaluator into a single clean API.
"""
import os
from typing import Dict, List, Optional

from .ingestor  import Ingestor
from .retriever import Retriever
from .answerer  import Answerer
from .evaluator import Evaluator


class RAGEngine:
    def __init__(
        self,
        qdrant_url: str  = "http://localhost:6333",
        llm_model:  str  = "claude-opus-4-8-20260528",
        top_k:      int  = 5,
        hybrid:     bool = True,
        rerank:     bool = True,
        fetch_k:    int  = 50,
    ):
        self.ingestor  = Ingestor(qdrant_url=qdrant_url)
        self.retriever = Retriever(qdrant_url=qdrant_url, enable_rerank=rerank)
        self.answerer  = Answerer(model=llm_model)
        self.evaluator = Evaluator()
        self.top_k     = top_k
        self.hybrid    = hybrid
        self.rerank    = rerank
        self.fetch_k   = fetch_k

    def ingest(self, path: str, extensions: Optional[List[str]] = None) -> int:
        """Ingest a file or directory. Returns number of chunks stored."""
        if os.path.isdir(path):
            return self.ingestor.ingest_directory(path, extensions)
        return self.ingestor.ingest_file(path)

    def ask(self, query: str, top_k: Optional[int] = None, evaluate: bool = False) -> Dict:
        """
        Ask a question against the knowledge base.
        Returns: { answer, sources, model, tokens, chunks_retrieved, eval? }

        evaluate=True adds RAGAS-style metrics (faithfulness, answer_relevance,
        context_precision). Costs one extra Haiku call for faithfulness scoring.
        """
        k = top_k or self.top_k
        chunks = self.retriever.search(
            query,
            top_k=k,
            hybrid=self.hybrid,
            rerank=self.rerank,
            fetch_k=self.fetch_k,
        )
        result = self.answerer.answer(query, chunks)
        result["chunks_retrieved"] = len(chunks)
        if evaluate:
            result["eval"] = self.evaluator.evaluate(query, result["answer"], chunks)
        return result

    def search(self, query: str, top_k: Optional[int] = None) -> List[Dict]:
        """Raw retrieval — returns chunks without LLM synthesis."""
        return self.retriever.search(
            query,
            top_k=top_k or self.top_k,
            hybrid=self.hybrid,
            rerank=self.rerank,
        )
