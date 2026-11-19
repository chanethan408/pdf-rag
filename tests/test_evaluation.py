"""Level 3 evaluation tests: metric math, dataset validity, reproducibility."""
from pathlib import Path

from app import evaluation as ev
from app import ingest, rag

A = {"doc_id": "a", "page": 1, "contains": "alpha value"}
B = {"doc_id": "b", "page": 2, "contains": "beta value"}


def chunk(doc, page, text, score=0.1):
    return {"doc_id": doc, "page": page, "text": text, "score": score, "chunk_id": f"{doc}-{page}"}


# ---------- metric math ----------
def test_hit_and_recall_at_k():
    chunks = [chunk("x", 1, "noise"), chunk("a", 1, "The  Alpha\nValue is 5"), chunk("y", 1, "noise"), chunk("b", 2, "beta value")]
    assert ev.hit_at_k(chunks, [A, B], k=3) is True
    assert ev.recall_at_k(chunks, [A, B], k=3) == 0.5      # B is rank 4, outside top-3
    assert ev.recall_at_k(chunks, [A, B], k=4) == 1.0
    assert ev.hit_at_k(chunks[:1], [A], k=3) is False
    assert ev.recall_at_k(chunks[:1], [A], k=3) == 0.0


def test_match_requires_doc_page_and_phrase():
    assert not ev.matches(chunk("a", 2, "alpha value"), A)   # wrong page
    assert not ev.matches(chunk("z", 1, "alpha value"), A)   # wrong doc
    assert not ev.matches(chunk("a", 1, "something else"), A)


def test_citation_scores():
    sources = [chunk("a", 1, "alpha value"), chunk("x", 1, "noise"), chunk("b", 2, "beta value")]
    both = ev.citation_scores("It is X [1] and Y [3].", sources, [A, B])
    assert both == {"has_valid_citations": True, "precision": 1.0, "recall": 1.0}
    partial = ev.citation_scores("It is X [1][2].", sources, [A, B])
    assert partial["precision"] == 0.5 and partial["recall"] == 0.5
    assert ev.citation_scores("No citation here", sources, [A])["has_valid_citations"] is False
    assert ev.citation_scores("Bad cite [9]", sources, [A])["has_valid_citations"] is False


def test_numbers_grounded():
    src = [chunk("a", 1, "The DSCR is 1.25x and files kept 7 years. Limit USD 5,000,000.")]
    assert ev.numbers_grounded("DSCR is 1.25x [1]. Kept 7 years. USD 5,000,000.", src)
    assert not ev.numbers_grounded("DSCR is 1.35x [1]", src)  # hallucinated number
    assert ev.numbers_grounded("Cited [1] only", src)         # citation markers are not claims


def test_must_include():
    assert ev.must_include_ok("Reviewed every 6 months [1]", ["6 months"])
    assert not ev.must_include_ok("Reviewed annually", ["6 months"])


# ---------- dataset ----------
def test_dataset_shape_and_labels():
    data = ev.load_dataset()
    assert len(data) == 10
    types = [q["type"] for q in data]
    assert types.count("direct") == 6 and types.count("multi_passage") == 1
    assert types.count("cross_document") == 1 and types.count("unsupported") == 2
    for q in data:
        assert q["question"].strip()
        if q["type"] == "unsupported":
            assert q["expected_abstain"] is True and q["evidence"] == []
        else:
            assert q["expected_abstain"] is False and q["evidence"] and q["must_include"]
    multi = next(q for q in data if q["type"] == "multi_passage")
    assert len(multi["evidence"]) >= 2 and len({e["doc_id"] for e in multi["evidence"]}) == 1
    cross = next(q for q in data if q["type"] == "cross_document")
    assert len({e["doc_id"] for e in cross["evidence"]}) == 2


def test_dataset_labels_exist_in_real_pdfs():
    """Guards against label drift: every labeled phrase must be on its labeled page of the real policy PDFs."""
    pdf_dir = Path("data/samples")  # pristine defaults, unaffected by UI uploads/removals
    for q in ev.load_dataset():
        for e in q["evidence"]:
            pages = dict(ingest.extract_pages(pdf_dir / f"{e['doc_id']}.pdf"))
            assert ev.norm(e["contains"]) in ev.norm(pages[e["page"]]), f"{q['id']}: label not found in PDF"


# ---------- reproducibility ----------
def test_eval_is_reproducible(env):
    ingest.ingest()
    data = [
        {"id": "t1", "type": "direct", "question": "minimum DSCR commercial loans",
         "evidence": [{"doc_id": "lending", "page": 1, "contains": "1.25x"}], "must_include": ["1.25"], "expected_abstain": False},
        {"id": "t2", "type": "unsupported", "question": "zebra quantum xylophone",
         "evidence": [], "must_include": [], "expected_abstain": True},
    ]
    r1 = ev.run_eval(data, retrieval_only=True)
    r2 = ev.run_eval(data, retrieval_only=True)
    assert r1["records"] == r2["records"] and r1["summary"] == r2["summary"]
    assert r1["summary"]["hit_at_k"] == 1.0
    # an incorrect retrieval must be recorded, not hidden
    bad = [{**data[0], "evidence": [{"doc_id": "lending", "page": 1, "contains": "not in the document"}]}]
    assert ev.run_eval(bad, retrieval_only=True)["records"][0]["hit_at_k"] is False
