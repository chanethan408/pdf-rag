"""Level 3 pipeline tests: retrieval metadata, evidence check, mocked LLM, failures, logging. No live API calls."""
import logging

import openai
import pytest
from fastapi.testclient import TestClient

from app import config, ingest, rag
from app.main import app

client = TestClient(app, raise_server_exceptions=False)
SECRET = "sk-secret-test-key"


@pytest.fixture
def indexed(env):
    ingest.ingest()
    return env


# ---------- retrieval / citation metadata ----------
def test_retrieval_returns_citation_metadata(indexed):
    hits = rag.retrieve("minimum DSCR commercial loans", k=3)
    assert hits
    for h in hits:
        assert h["doc_id"] and h["filename"].endswith(".pdf") and h["page"] >= 1
        assert h["chunk_id"] and h["text"].strip() and isinstance(h["score"], float)
    assert "1.25x" in hits[0]["text"] and hits[0]["filename"] == "lending.pdf"


# ---------- empty results / evidence check ----------
def test_missing_index_raises(env):
    with pytest.raises(rag.IndexNotReadyError):
        rag.retrieve("anything")


def test_evidence_check_abstains_without_calling_llm(indexed, monkeypatch):
    monkeypatch.setattr(config, "MAX_DISTANCE", 0.5)

    def must_not_call(*a, **k):
        raise AssertionError("LLM must not be called when evidence is insufficient")
    monkeypatch.setattr(rag, "generate", must_not_call)
    res = rag.answer_question("zebra quantum xylophone")
    assert res["abstained"] and res["abstain_reason"] == "no_relevant_evidence" and res["sources"] == []


def test_relevant_question_passes_evidence_check(indexed, monkeypatch):
    monkeypatch.setattr(config, "MAX_DISTANCE", 0.9)
    monkeypatch.setattr(rag, "generate", lambda q, hits: ("DSCR is 1.25x [1]", False))
    res = rag.answer_question("minimum DSCR commercial loans")
    assert not res["abstained"] and res["sources"] and res["abstain_reason"] is None


# ---------- mocked LLM ----------
class FakeOpenAI:
    reply = "INSUFFICIENT_EVIDENCE"
    error = None

    def __init__(self, api_key=None):
        self.chat = self
        self.completions = self

    def create(self, **kwargs):
        if FakeOpenAI.error:
            raise FakeOpenAI.error
        msg = type("M", (), {"content": FakeOpenAI.reply})
        return type("R", (), {"choices": [type("C", (), {"message": msg})]})


@pytest.fixture
def fake_llm(monkeypatch):
    monkeypatch.setattr(config, "OPENAI_API_KEY", SECRET)
    monkeypatch.setattr(openai, "OpenAI", FakeOpenAI)
    FakeOpenAI.reply, FakeOpenAI.error = "INSUFFICIENT_EVIDENCE", None
    return FakeOpenAI


def test_llm_abstention_returns_no_sources(indexed, fake_llm):
    res = rag.answer_question("minimum DSCR commercial loans")
    assert res["abstained"] and res["abstain_reason"] == "llm_abstained" and res["sources"] == []


def test_llm_answer_is_returned_with_sources(indexed, fake_llm):
    fake_llm.reply = "The minimum DSCR is 1.25x [1]."
    res = rag.answer_question("minimum DSCR commercial loans")
    assert not res["abstained"] and "1.25x" in res["answer"] and res["sources"]


def test_llm_failure_is_controlled_and_leaks_no_secret(indexed, fake_llm):
    fake_llm.error = openai.OpenAIError(f"auth failed for {SECRET}")
    r = client.post("/ask", json={"question": "minimum DSCR commercial loans"})
    assert r.status_code == 502 and r.json()["error"] == "llm_error"
    assert SECRET not in r.text and "Traceback" not in r.text


def test_missing_api_key_is_controlled(indexed, monkeypatch):
    monkeypatch.setattr(config, "OPENAI_API_KEY", "")
    r = client.post("/ask", json={"question": "minimum DSCR commercial loans"})
    assert r.status_code == 502


# ---------- logging ----------
def test_request_logging_has_ids_scores_latency_but_no_secrets(indexed, fake_llm, caplog):
    fake_llm.reply = "The minimum DSCR is 1.25x [1]."
    caplog.set_level(logging.INFO)
    question = "zq-unique-question minimum DSCR commercial loans"
    r = client.post("/ask", json={"question": question})
    assert r.status_code == 200
    rid = r.headers["X-Request-ID"]
    assert rid
    logs = caplog.text
    assert f"request_id={rid}" in logs
    assert "latency_ms=" in logs and "status=200" in logs
    assert "lending-" in logs                      # retrieved chunk IDs are logged
    assert SECRET not in logs                      # no secrets
    assert "zq-unique-question" not in logs        # no raw user question
    assert "retained for 7 years" not in logs and "1.25x." not in logs  # no raw document text


def test_error_status_is_logged(indexed, monkeypatch, caplog):
    caplog.set_level(logging.INFO)
    r = client.post("/ask", json={"question": ""})
    assert r.status_code == 422
    assert "status=422" in caplog.text


def test_each_request_gets_distinct_id(indexed):
    ids = {client.get("/health").headers["X-Request-ID"] for _ in range(3)}
    assert len(ids) == 3
