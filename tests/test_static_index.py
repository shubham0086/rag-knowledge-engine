"""
Offline tests for the serverless static-index builder.

Uses a fake embedder so the suite runs with no network, no torch, and no Qdrant.
Verifies chunking -> embedding -> index.json shape, HTML stripping, and that an
injected embedder is used (the path serverless mode depends on).
"""
import json

from src.static_index import build_static_index, _strip_html
from src.chunking import _chunk_text


class FakeEmbedder:
    """Deterministic, dependency-free embedder for tests."""
    dim = 3
    MODEL = "fake-test-embed"

    def encode(self, texts):
        single = isinstance(texts, str)
        items = [texts] if single else list(texts)
        out = [[float(len(t)), float(t.count(" ")), 1.0] for t in items]
        return out[0] if single else out


def test_strip_html_removes_tags_and_scripts():
    html = "<html><head><style>.x{}</style></head><body><h1>Hi</h1><script>bad()</script><p>World</p></body></html>"
    assert _strip_html(html) == "Hi World"


def test_build_static_index_shape(tmp_path):
    (tmp_path / "a.md").write_text("hello world from shubham portfolio", encoding="utf-8")
    (tmp_path / "b.html").write_text("<p>edge deployable rag engine</p>", encoding="utf-8")

    out = tmp_path / "index.json"
    summary = build_static_index(
        str(tmp_path),
        out_path=str(out),
        embedder=FakeEmbedder(),
        source_base_url="https://example.com",
    )

    assert summary["count"] >= 2
    assert summary["dim"] == 3
    assert summary["model"] == "fake-test-embed"

    index = json.loads(out.read_text(encoding="utf-8"))
    assert index["source_base_url"] == "https://example.com"
    assert index["count"] == len(index["records"])
    for rec in index["records"]:
        assert set(("file", "chunk_idx", "text", "vector")).issubset(rec)
        assert len(rec["vector"]) == 3
    # HTML record must be tag-free
    html_recs = [r for r in index["records"] if r["file"] == "b.html"]
    assert html_recs and "<p>" not in html_recs[0]["text"]


def test_only_listed_extensions_are_indexed(tmp_path):
    (tmp_path / "keep.md").write_text("keep me", encoding="utf-8")
    (tmp_path / "skip.png").write_text("binary-ish", encoding="utf-8")

    out = tmp_path / "index.json"
    build_static_index(str(tmp_path), out_path=str(out), embedder=FakeEmbedder())
    index = json.loads(out.read_text(encoding="utf-8"))
    files = {r["file"] for r in index["records"]}
    assert "keep.md" in files
    assert "skip.png" not in files


def test_exclude_dirs_skips_matching_folders(tmp_path):
    (tmp_path / "keep.md").write_text("top level keep", encoding="utf-8")
    embedded = tmp_path / "agent-anatomy"
    embedded.mkdir()
    (embedded / "README.md").write_text("embedded repo readme", encoding="utf-8")

    out = tmp_path / "index.json"
    build_static_index(
        str(tmp_path),
        out_path=str(out),
        embedder=FakeEmbedder(),
        exclude_dirs={"agent-anatomy"},
    )
    index = json.loads(out.read_text(encoding="utf-8"))
    files = {r["file"] for r in index["records"]}
    assert "keep.md" in files
    assert not any("agent-anatomy" in f for f in files)


def test_chunking_still_importable_from_chunking_module():
    # Backward-compat: chunking primitives work standalone (no torch import).
    assert _chunk_text("one two three") == ["one two three"]
