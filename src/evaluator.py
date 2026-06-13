"""
Evaluator: RAGAS-style quality metrics for the RAG pipeline.

Three metrics:
  faithfulness      — fraction of answer claims grounded in context (LLM judge)
  answer_relevance  — semantic similarity between question and answer embeddings
  context_precision — mean retrieval/rerank score of returned chunks

Using Claude Haiku for faithfulness judgement keeps eval costs low.
Cosine sim is computed with numpy; no sklearn dependency needed.
"""
import json
import os
from typing import List, Dict

import anthropic
import numpy as np
from sentence_transformers import SentenceTransformer

from .ingestor import EMBED_MODEL

_FAITHFULNESS_PROMPT = """You are evaluating a RAG system's answer for faithfulness to its sources.

Context chunks:
{context}

Answer:
{answer}

List all factual claims in the answer. For each, judge whether it is directly supported by the context.
Respond with JSON only:
{{
  "total_claims": <int>,
  "supported_claims": <int>,
  "faithfulness_score": <float 0.0-1.0>,
  "unsupported": [<strings>]
}}"""


def _cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    return float(np.dot(a, b) / (denom + 1e-10))


class Evaluator:
    def __init__(self, model: str = "claude-haiku-4-5-20251001"):
        self.client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))
        self.model = model
        self.embedder = SentenceTransformer(EMBED_MODEL)

    def evaluate(self, query: str, answer: str, context_chunks: List[Dict]) -> Dict:
        """Return all three quality metrics for a single RAG response."""
        return {
            "faithfulness":      self._faithfulness(answer, context_chunks),
            "answer_relevance":  self._answer_relevance(query, answer),
            "context_precision": self._context_precision(context_chunks),
        }

    def _faithfulness(self, answer: str, chunks: List[Dict]) -> float:
        if not chunks or not answer.strip():
            return 0.0
        context = "\n\n".join(f"[{i+1}] {c['text']}" for i, c in enumerate(chunks))
        prompt = _FAITHFULNESS_PROMPT.format(context=context, answer=answer)
        try:
            resp = self.client.messages.create(
                model=self.model,
                max_tokens=512,
                messages=[{"role": "user", "content": prompt}],
            )
            data = json.loads(resp.content[0].text)
            return float(data.get("faithfulness_score", 0.0))
        except Exception:
            return 0.0

    def _answer_relevance(self, query: str, answer: str) -> float:
        if not query.strip() or not answer.strip():
            return 0.0
        q_emb, a_emb = self.embedder.encode([query, answer])
        return round(_cosine_sim(q_emb, a_emb), 4)

    def _context_precision(self, chunks: List[Dict]) -> float:
        if not chunks:
            return 0.0
        score_key = "rerank_score" if "rerank_score" in chunks[0] else "score"
        return round(float(np.mean([c.get(score_key, 0.0) for c in chunks])), 4)
