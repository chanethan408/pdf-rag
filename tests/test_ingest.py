"""Level 2 tests: idempotency, updates, failure recovery. Fake embeddings => no model download, no API."""
import pytest

from app import config, ingest, rag
from tests.helpers import fake_embed, make_pdf


def all_ids():
    return rag.get_client().get_collection(config.COLLECTION).get(include=[])["ids"]


def test_idempotent(env):
    first = ingest.ingest()
    assert {r["status"] for r in first} == {"added"}
    ids1, count1 = all_ids(), rag.index_count()
    second = ingest.ingest()
    assert {r["status"] for r in second} == {"unchanged"}
    assert sorted(all_ids()) == sorted(ids1)          # stable IDs, nothing new
    assert len(set(ids1)) == len(ids1)                # no duplicates
    assert rag.index_count() == count1
    assert ingest.check_consistency() == []


def test_update_replaces_old_version(env):
    ingest.ingest()
    old_hash = ingest.load_registry()["lending"]["active_hash"]
    make_pdf(env / "lending.pdf", ["The minimum DSCR for commercial loans is 1.40x.\n\nLoans need 24 months history."])
    results = {r["file"]: r["status"] for r in ingest.ingest()}
    assert results == {"docs.pdf": "unchanged", "lending.pdf": "updated"}

    reg = ingest.load_registry()["lending"]
    assert reg["version"] == 2 and reg["active_hash"] != old_hash   # hash changed with content
    hits = rag.retrieve("minimum DSCR commercial loans", k=5)
    texts = " ".join(h["text"] for h in hits)
    assert "1.40x" in texts and "1.25x" not in texts                # old threshold never returned
    assert all(h["doc_hash"] != old_hash for h in hits if h["doc_id"] == "lending")
    assert len(all_ids()) > rag.index_count()                       # superseded chunks kept but inactive
    assert ingest.check_consistency() == []


def test_corrupt_and_empty_pdfs_fail_gracefully(env):
    (env / "corrupt.pdf").write_bytes(b"this is not a pdf")
    (env / "empty.pdf").write_bytes(b"")
    make_pdf(env / "blank.pdf", [""])
    results = {r["file"]: r for r in ingest.ingest()}
    for bad in ("corrupt.pdf", "empty.pdf", "blank.pdf"):
        assert results[bad]["status"] == "failed" and results[bad]["reason"]
    assert results["lending.pdf"]["status"] == "added"              # good docs unaffected
    assert "corrupt" not in ingest.load_registry()


def test_failed_replacement_keeps_previous_index(env):
    ingest.ingest()
    before = ingest.load_registry()["lending"]
    count = rag.index_count()
    (env / "lending.pdf").write_bytes(b"garbage")                   # replacement is corrupt
    res = {r["file"]: r["status"] for r in ingest.ingest()}
    assert res["lending.pdf"] == "failed"
    assert ingest.load_registry()["lending"] == before              # active version unchanged
    assert rag.index_count() == count
    hits = rag.retrieve("minimum DSCR", k=3)
    assert any("1.25x" in h["text"] for h in hits)                  # old content still served
    assert ingest.check_consistency() == []


def test_embedding_failure_keeps_previous_index(env, monkeypatch):
    ingest.ingest()
    make_pdf(env / "lending.pdf", ["The minimum DSCR for commercial loans is 1.50x."])

    def boom(texts):
        raise RuntimeError("embedder down")
    monkeypatch.setattr(rag, "embed", boom)
    res = {r["file"]: r for r in ingest.ingest()}
    assert res["lending.pdf"]["status"] == "failed"
    monkeypatch.setattr(rag, "embed", fake_embed)
    assert any("1.25x" in h["text"] for h in rag.retrieve("minimum DSCR", k=3))


def test_chunks_have_valid_metadata(env):
    ingest.ingest()
    metas = rag.get_client().get_collection(config.COLLECTION).get(where=rag.ACTIVE, include=["metadatas", "documents"])
    assert metas["ids"]
    for m, doc in zip(metas["metadatas"], metas["documents"]):
        assert doc.strip()
        assert m["doc_id"] and m["filename"].endswith(".pdf") and m["page"] >= 1 and m["chunk_id"]
