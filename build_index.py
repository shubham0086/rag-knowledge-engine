"""
Build the static RAG index for the portfolio chatbot (serverless mode).

Reads GEMINI_API_KEY from .env, chunks + embeds the PUBLIC portfolio content with
Gemini text-embedding-004, and writes index.json. PUBLIC CONTENT ONLY -- this points
at the deployed portfolio site, never at private docs.

Usage:
    python build_index.py                      # uses the defaults below
    python build_index.py <corpus_dir> <out>   # override corpus dir / output path
"""
import sys
from pathlib import Path

from dotenv import load_dotenv

from src.embeddings import GeminiEmbedder
from src.static_index import build_static_index

# Load GEMINI_API_KEY from the .env beside this script.
load_dotenv(Path(__file__).parent / ".env")

# The deployed portfolio site IS the public corpus.
DEFAULT_CORPUS = r"D:\dev\Shubham-Portfolio-Analysis\MyPortfolio.github.io"
DEFAULT_OUT = str(Path(__file__).parent / "index.json")

# Citation base URL. UPDATE to the Vercel domain once it's live.
LIVE_BASE_URL = "https://shubham0086.github.io/MyPortfolio.github.io"

# Skip embedded repos, asset/build noise -- keep only curated public pages + blog posts.
EXCLUDE = {
    ".git", "node_modules", "__pycache__", ".pytest_cache",
    "assets", "scratch",
    "agent-anatomy", "ai-systems-evolution",  # embedded repos (code, not portfolio prose)
}


def main() -> int:
    corpus = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_CORPUS
    out = sys.argv[2] if len(sys.argv) > 2 else DEFAULT_OUT

    if not Path(corpus).is_dir():
        print(f"ERROR: corpus dir not found: {corpus}")
        return 1

    print(f"Corpus : {corpus}")
    print(f"Output : {out}")
    print(f"Base URL: {LIVE_BASE_URL}")
    print("Embedding with Gemini text-embedding-004 (requires GEMINI_API_KEY in .env)...")

    try:
        # Throttle smooths bursts so we stay under the free-tier rate limit;
        # the embedder also retries on 429 with backoff.
        embedder = GeminiEmbedder(throttle=0.4)
        summary = build_static_index(
            corpus,
            out_path=out,
            embedder=embedder,
            extensions=[".md", ".html", ".txt"],
            source_base_url=LIVE_BASE_URL,
            exclude_dirs=EXCLUDE,
        )
    except ValueError as e:
        # GeminiEmbedder raises this when the key is missing.
        print(f"\nERROR: {e}")
        print("Add your key to .env:  GEMINI_API_KEY=...")
        return 1

    print(f"\nDone. {summary['count']} chunks | dim {summary['dim']} | model {summary['model']}")
    print(f"Wrote {summary['out_path']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
