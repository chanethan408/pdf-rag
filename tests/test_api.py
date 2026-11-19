"""Level 1 API tests. LLM/retrieval are mocked, so no API key or index is needed."""
from pathlib import Path

from fastapi.testclient import TestClient

from app import rag
from app.main import app

client = TestClient(app, raise_server_exceptions=False)

FAKE = {
    "question": "q",
    "answer": "DSCR is 1.25x [1]",
    "abstained": False,
    "sources": [{"doc_id": "d", "filename": "d.pdf", "page": 1, "chunk_id": "d-p1-c0", "score": 0.1, "text": "t"}],
}


def test_demo_ui_and_pdfs_are_served():
    r = client.get("/")
    assert r.status_code == 200 and "text/html" in r.headers["content-type"]
    assert "Credit Policy Assistant" in r.text
    assert client.get("/pdfs/../.env").status_code in (400, 404)  # cannot escape the PDF folder
    existing = sorted(Path("data/pdfs").glob("*.pdf"))  # working folder; contents depend on UI use, so don't assume
    if existing:
        pdf = client.get(f"/pdfs/{existing[0].name}")
        assert pdf.status_code == 200 and pdf.headers["content-type"] == "application/pdf"


def test_documents_list(env):
    r = client.get("/documents")
    assert r.status_code == 200
    docs = r.json()
    assert {d["filename"] for d in docs} == {"lending.pdf", "docs.pdf"}
    assert all(d["pages"] >= 1 and d["title"] and d["indexed"] is False for d in docs)


def test_ask_valid(monkeypatch):
    monkeypatch.setattr(rag, "answer_question", lambda q: {**FAKE, "question": q})
    r = client.post("/ask", json={"question": "min DSCR?"})
    assert r.status_code == 200
    assert r.json()["sources"][0]["page"] == 1


def test_empty_missing_wrongtype_malformed():
    assert client.post("/ask", json={"question": ""}).status_code == 422
    assert client.post("/ask", json={"question": "   "}).status_code == 422
    assert client.post("/ask", json={}).status_code == 422
    assert client.post("/ask", json={"question": 123}).status_code == 422
    r = client.post("/ask", content="{bad", headers={"content-type": "application/json"})
    assert 400 <= r.status_code < 500


def test_llm_error(monkeypatch):
    def boom(q):
        raise rag.LLMError("LLM provider call failed")
    monkeypatch.setattr(rag, "answer_question", boom)
    r = client.post("/ask", json={"question": "x"})
    assert r.status_code == 502 and r.json()["error"] == "llm_error"


def test_index_missing(monkeypatch):
    def boom(q):
        raise rag.IndexNotReadyError("Vector index not found")
    monkeypatch.setattr(rag, "answer_question", boom)
    assert client.post("/ask", json={"question": "x"}).status_code == 503


def test_unexpected_hides_details(monkeypatch):
    def boom(q):
        raise ValueError("secret sk-123")
    monkeypatch.setattr(rag, "answer_question", boom)
    r = client.post("/ask", json={"question": "x"})
    assert r.status_code == 500 and "sk-123" not in r.text


def test_health(monkeypatch):
    monkeypatch.setattr(rag, "index_count", lambda: 5)
    assert client.get("/health").status_code == 200
    monkeypatch.setattr(rag, "index_count", lambda: 0)
    assert client.get("/health").status_code == 503
