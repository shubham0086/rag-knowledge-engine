"""
Pluggable embedding backends.

The engine originally embedded only with sentence-transformers (local, torch, 384-dim).
Serverless / edge deployment can't carry torch, so embeddings are now pluggable:

  - LocalEmbedder  : sentence-transformers all-MiniLM-L6-v2 (default, original behavior)
  - GeminiEmbedder : Google text-embedding-004 via REST (no torch, serverless-friendly)

Both expose:
  - encode(texts) -> List[List[float]]   (or List[float] if given a single str)
  - .dim                                  (embedding dimensionality)

The SAME embedder must be used to build an index and to query it (vectors must be
comparable). For serverless mode, build the static index with GeminiEmbedder and
query with the same model in the runtime function.
"""
import os
from typing import List, Union

Texts = Union[str, List[str]]


class LocalEmbedder:
    """sentence-transformers backend. Heavy (torch); the original default."""

    dim = 384
    MODEL = "all-MiniLM-L6-v2"

    def __init__(self, model_name: str = None):
        # Imported lazily so this module is importable without torch installed.
        from sentence_transformers import SentenceTransformer
        self.MODEL = model_name or self.MODEL
        self.model = SentenceTransformer(self.MODEL)

    def encode(self, texts: Texts):
        single = isinstance(texts, str)
        arr = self.model.encode([texts] if single else list(texts))
        out = arr.tolist()
        return out[0] if single else out


class GeminiEmbedder:
    """
    Google Gemini embeddings via REST (uses httpx, already a dependency).
    No torch, no model download -> works inside a serverless function and at build time.
    Reads GEMINI_API_KEY from the environment unless passed explicitly.

    Uses gemini-embedding-001 at 768 dims (requested via outputDimensionality). Google
    normalizes the default 3072-dim output but NOT reduced dims, so we L2-normalize here
    to keep cosine similarity correct. The SAME model + dim must be used to query.
    """

    dim = 768
    MODEL = "gemini-embedding-001"
    _ENDPOINT = "https://generativelanguage.googleapis.com/v1beta/models"

    def __init__(self, api_key: str = None, model: str = None, dim: int = None,
                 throttle: float = 0.0):
        self.api_key = api_key or os.environ.get("GEMINI_API_KEY")
        if not self.api_key:
            raise ValueError("GEMINI_API_KEY is not set (env var or api_key arg required)")
        self.MODEL = model or self.MODEL
        self.dim = dim or self.dim
        self.throttle = throttle  # seconds to wait between requests (smooths bursts)

    @staticmethod
    def _normalize(vec: List[float]) -> List[float]:
        norm = sum(x * x for x in vec) ** 0.5
        return [x / norm for x in vec] if norm else vec

    def _post_with_retry(self, client, url, payload, max_retries: int = 6):
        import time
        delay = 2.0
        resp = None
        for _ in range(max_retries):
            resp = client.post(url, json=payload)
            if resp.status_code in (429, 503):
                ra = resp.headers.get("retry-after")
                wait = float(ra) if (ra and ra.replace(".", "", 1).isdigit()) else delay
                time.sleep(wait)
                delay = min(delay * 2, 60)
                continue
            resp.raise_for_status()
            return resp
        resp.raise_for_status()  # exhausted retries -> surface the last error
        return resp

    def encode(self, texts: Texts):
        import time
        import httpx

        single = isinstance(texts, str)
        items = [texts] if single else list(texts)
        url = f"{self._ENDPOINT}/{self.MODEL}:embedContent?key={self.api_key}"
        out: List[List[float]] = []
        with httpx.Client(timeout=60) as client:
            for t in items:
                resp = self._post_with_retry(
                    client, url,
                    {
                        "model": f"models/{self.MODEL}",
                        "content": {"parts": [{"text": t}]},
                        "outputDimensionality": self.dim,
                    },
                )
                out.append(self._normalize(resp.json()["embedding"]["values"]))
                if self.throttle:
                    time.sleep(self.throttle)
        return out[0] if single else out


def get_embedder(name: str = "local", **kwargs):
    """Factory. name in {local, gemini}. kwargs passed to the backend constructor."""
    key = (name or "local").lower()
    if key in ("local", "minilm", "sentence-transformers", "st"):
        return LocalEmbedder(**kwargs)
    if key in ("gemini", "google"):
        return GeminiEmbedder(**kwargs)
    raise ValueError(f"unknown embedder backend: {name!r} (expected 'local' or 'gemini')")
