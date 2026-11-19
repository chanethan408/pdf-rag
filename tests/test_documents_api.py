"""Upload / remove document endpoints. Fake embeddings, temp folders: never touches the real index or an API."""
import pytest
from fastapi.testclient import TestClient

from app import config, ingest, rag
from app.main import app
from tests.helpers import make_pdf

client = TestClient(app, raise_server_exceptions=False)


def pdf_bytes(tmp_path, text, name="up.pdf"):
    p = tmp_path / name
    make_pdf(p, [text])
    return p.read_bytes()


def put(name, data):
    return client.put(f"/documents/{name}", content=data, headers={"content-type": "application/pdf"})


def test_upload_new_document_is_searchable(env, tmp_path):
    ingest.ingest()
    before = rag.index_count()
    r = put("travel_policy.pdf", pdf_bytes(tmp_path, "Business class flights need VP approval above 6 hours."))
    assert r.status_code == 200 and r.json()["status"] == "added" and r.json()["chunks"] >= 1
    assert (env / "travel_policy.pdf").exists()
    assert not list((env / ".staging").glob("*"))                  # staging cleaned up
    assert rag.index_count() > before
    assert "VP approval" in rag.retrieve("business class flights approval", k=1)[0]["text"]
    listed = {d["filename"]: d for d in client.get("/documents").json()}
    assert listed["travel_policy.pdf"]["indexed"] is True


def test_reupload_unchanged_then_modified(env, tmp_path):
    ingest.ingest()
    v1 = pdf_bytes(tmp_path, "The flight limit is 6 hours.")
    assert put("travel.pdf", v1).json()["status"] == "added"
    assert put("travel.pdf", v1).json()["status"] == "unchanged"
    r = put("travel.pdf", pdf_bytes(tmp_path, "The flight limit is 8 hours.", "v2.pdf"))
    assert r.json()["status"] == "updated" and r.json()["version"] == 2
    text = " ".join(h["text"] for h in rag.retrieve("flight limit hours", k=5))
    assert "8 hours" in text and "6 hours" not in text
    assert ingest.check_consistency() == []


def test_bad_uploads_rejected_and_existing_doc_survives(env, tmp_path):
    ingest.ingest()
    good = pdf_bytes(tmp_path, "The flight limit is 6 hours.")
    put("travel.pdf", good)
    count = rag.index_count()

    assert put("travel.pdf", b"not a pdf at all").status_code == 400                      # wrong magic bytes
    assert put("travel.pdf", b"%PDF-1.4 truncated garbage").status_code == 422           # looks like PDF, isn't
    assert (env / "travel.pdf").read_bytes() == good                                      # previous file untouched
    assert rag.index_count() == count
    assert any("6 hours" in h["text"] for h in rag.retrieve("flight limit", k=3))


@pytest.mark.parametrize("name", ["notes.txt", "a$b.pdf", "..\\evil.pdf", "x" * 200 + ".pdf", ".hidden.pdf"])
def test_invalid_filenames_rejected(env, name):
    assert put(name, b"%PDF-1.4").status_code == 400


def test_upload_size_limit(env, monkeypatch):
    monkeypatch.setattr(config, "MAX_UPLOAD_BYTES", 50)
    assert put("big.pdf", b"%PDF-" + b"x" * 500).status_code == 413


def test_remove_document(env):
    ingest.ingest()
    assert any(h["doc_id"] == "docs" for h in rag.retrieve("loan files retained years", k=3))
    r = client.delete("/documents/docs.pdf")
    assert r.status_code == 200
    assert not (env / "docs.pdf").exists()
    assert all(h["doc_id"] != "docs" for h in rag.retrieve("loan files retained years", k=5))
    assert "docs" not in ingest.load_registry()
    ids = rag.get_client().get_collection(config.COLLECTION).get(include=["metadatas"])["metadatas"]
    assert all(m["doc_id"] != "docs" for m in ids)                 # chunks (all versions) really deleted
    assert ingest.check_consistency() == []
    assert client.delete("/documents/docs.pdf").status_code == 404
    assert client.delete("/documents/bad$name.pdf").status_code == 400


def test_removing_last_document_gives_clear_error(env):
    ingest.ingest()
    client.delete("/documents/docs.pdf")
    client.delete("/documents/lending.pdf")
    assert client.get("/health").status_code == 503
    assert client.post("/ask", json={"question": "anything"}).status_code == 503


# ---------- default sample restore ----------
@pytest.fixture
def samples(env, tmp_path, monkeypatch):
    d = tmp_path / "samples"
    d.mkdir()
    make_pdf(d / "sample_a.pdf", ["Default policy A: the review period is 12 months."])
    monkeypatch.setattr(config, "SAMPLES_DIR", str(d))
    return d


def test_restore_samples_adds_missing_default(env, samples):
    assert ingest.restore_samples() == ["sample_a.pdf"]
    assert (env / "sample_a.pdf").exists()
    assert "review period" in rag.retrieve("review period months", k=1)[0]["text"]
    assert ingest.restore_samples() == []                       # already present: nothing to do


def test_removed_default_comes_back_on_next_start(env, samples):
    ingest.restore_samples()
    assert client.delete("/documents/sample_a.pdf").status_code == 200
    assert not (env / "sample_a.pdf").exists() and "sample_a" not in ingest.load_registry()
    assert ingest.restore_samples() == ["sample_a.pdf"]         # simulated restart
    assert any(h["doc_id"] == "sample_a" for h in rag.retrieve("review period months", k=3))
    assert ingest.check_consistency() == []


def test_restore_does_not_overwrite_user_version(env, samples, tmp_path):
    ingest.restore_samples()
    newer = pdf_bytes(tmp_path, "Default policy A: the review period is 6 months.", "newer.pdf")
    assert put("sample_a.pdf", newer).json()["status"] == "updated"
    assert ingest.restore_samples() == []
    assert "6 months" in rag.retrieve("review period months", k=1)[0]["text"]


def test_pristine_samples_exist_in_repo():
    from pathlib import Path
    names = {p.name for p in Path("data/samples").glob("*.pdf")}
    assert names >= {"lending_requirements.pdf", "credit_risk_review.pdf", "loan_documentation.pdf"}
