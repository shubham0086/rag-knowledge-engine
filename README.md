# RAG Knowledge Engine

Ask questions about any codebase or document collection and get cited answers in seconds.

Ingests files into Qdrant (vector database), retrieves the most relevant chunks via hybrid search, reranks with a cross-encoder, and generates a grounded answer via Claude with source citations. Refuses to hallucinate — if the answer is not in the documents, it says so.

## What it does

```
Your files → chunk → embed → Qdrant
                                 ↓
Query → [vector search + BM25 keyword] → RRF fusion → cross-encoder rerank → Claude → cited answer
```

Four stages, each independently testable:

| Stage | File | What it does |
|-------|------|-------------|
| **Ingest** | `src/ingestor.py` | Read files, chunk text (512 words + 64 overlap), embed with `all-MiniLM-L6-v2`, upsert to Qdrant |
| **Retrieve** | `src/retriever.py` | Hybrid search (vector + BM25) fused via RRF, optional cross-encoder reranking |
| **Answer** | `src/answerer.py` | Pass ranked chunks to Claude, require citations, refuse if out-of-scope |
| **Evaluate** | `src/evaluator.py` | RAGAS-style metrics: faithfulness, answer relevance, context precision |

## Quick start

**1. Start Qdrant locally**
```bash
docker run -p 6333:6333 qdrant/qdrant
```

**2. Install dependencies**
```bash
pip install -r requirements.txt
```

**3. Set API key**
```bash
export ANTHROPIC_API_KEY=sk-ant-...
```

**4. Ingest and ask**
```python
from src import RAGEngine

engine = RAGEngine()
engine.ingest("./my-codebase")
result = engine.ask("How does the authentication flow work?")

print(result["answer"])   # grounded, cited answer
print(result["sources"])  # source files
```

## API reference

```python
engine = RAGEngine(
    qdrant_url="http://localhost:6333",
    llm_model="claude-opus-4-8-20260528",
    top_k=5,
    hybrid=True,   # vector + BM25 fused via RRF
    rerank=True,   # cross-encoder rescores candidate pool
    fetch_k=50,    # candidates to fetch before reranking
)

# Ingest a file or directory
chunks_stored = engine.ingest("./docs", extensions=[".md", ".py", ".txt"])

# Ask a question (returns grounded answer + sources + token usage)
result = engine.ask("What does the circuit breaker do?")
# {
#   "answer": "The circuit breaker tracks failure rate... [1][2]",
#   "sources": ["src/circuit_breaker.py", "docs/architecture.md"],
#   "model": "claude-opus-4-8-20260528",
#   "tokens": {"input": 1240, "output": 180},
#   "chunks_retrieved": 5
# }

# Ask with RAGAS-style quality metrics
result = engine.ask("What does the circuit breaker do?", evaluate=True)
# result["eval"] = {
#   "faithfulness": 0.95,       # fraction of claims grounded in context
#   "answer_relevance": 0.88,   # semantic similarity: question ↔ answer
#   "context_precision": 0.81,  # mean rerank score of returned chunks
# }

# Raw retrieval (no LLM — just ranked chunks)
chunks = engine.search("rate limiting implementation", top_k=3)
```

## Tests

```bash
pytest tests/ -v
```

20 tests. All mock Qdrant, Anthropic, and the cross-encoder — no live services needed.

```
tests/test_ingestor.py   — 6 tests  (chunking, doc IDs, file ingestion)
tests/test_retriever.py  — 6 tests  (vector search, RRF fusion, hybrid, rerank)
tests/test_reranker.py   — 3 tests  (score sorting, top-k, empty input)
tests/test_answerer.py   — 2 tests  (citation answer, not-found refusal)
tests/test_evaluator.py  — 3 tests  (faithfulness, relevance, precision)
```

## Design decisions

**Why hybrid search (BM25 + vector)?** Each retrieval method catches different failures. Vector search misses exact keyword queries ("RFC 2119 MUST NOT"). BM25 misses semantic paraphrases ("how does auth work?" ≠ "authentication logic"). Combining both via RRF gets both. No score normalisation needed — RRF only uses rank positions.

**Why cross-encoder reranking?** The bi-encoder used during retrieval encodes query and passage separately, then scores cosine similarity. A cross-encoder sees both together via full attention — far better relevance judgement. We fetch 50 candidates and rerank to 5, so the LLM sees the highest-quality passages.

**Why Qdrant?** Best price-performance in the 2026 vector DB market. Runs locally in Docker, has a generous free cloud tier, and the Python client is the cleanest of the major options.

**Why `all-MiniLM-L6-v2`?** 384-dimension embeddings, CPU inference, fast enough for local development. Swap to `text-embedding-3-large` for production by changing one constant in `ingestor.py`.

**Why Claude for answering?** Citation compliance. Claude reliably follows "only answer from context, always cite." The `Answerer` returns "Not found in the provided documents." when context doesn't support a claim — hallucinated answers in a knowledge base are worse than silence.

**Why RAGAS-style eval?** Retrieval quality improvements need measurement, not vibes. The three metrics cover different failure modes: faithfulness catches hallucination, answer relevance catches off-topic answers, context precision catches bad retrieval.

## Stack

- **Qdrant** — vector database, local Docker or cloud
- **sentence-transformers** — `all-MiniLM-L6-v2` (bi-encoder) + `ms-marco-MiniLM-L6-v2` (cross-encoder)
- **rank-bm25** — BM25Okapi for keyword retrieval
- **Anthropic SDK** — Claude for answer generation and faithfulness evaluation
- **pytest** — 20 tests, all mocked

## Related repos

- [research-agent](https://github.com/shubham0086/research-agent) — web research pipeline using the same LLM router
- [equilibrium](https://github.com/shubham0086/equilibrium) — AgentKernel including the retriever engine this was extracted from
