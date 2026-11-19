"""Evaluation of retrieval, answers, citations and abstention on a fixed labeled question set.

Evidence labels are (doc_id, page, contains-phrase) rather than chunk IDs, because chunk IDs change when a
document version changes. A retrieved chunk "matches" evidence if doc_id and page agree and the phrase occurs
in its text (whitespace/case-normalised).
"""
import json
import re
import time
from pathlib import Path

from . import config, rag

DATASET = Path("data/eval/questions.json")
CITE = re.compile(r"\[(\d+)\]")


def norm(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip().lower()


def load_dataset(path: Path = DATASET) -> list[dict]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


# ---------- retrieval metrics ----------
def matches(chunk: dict, ev: dict) -> bool:
    return chunk["doc_id"] == ev["doc_id"] and chunk["page"] == ev["page"] and norm(ev["contains"]) in norm(chunk["text"])


def covered(chunks: list[dict], evidence: list[dict]) -> list[bool]:
    return [any(matches(c, e) for c in chunks) for e in evidence]


def hit_at_k(chunks: list[dict], evidence: list[dict], k: int = 3) -> bool:
    """1 if ANY labeled evidence item appears in the top-k chunks."""
    return any(covered(chunks[:k], evidence))


def recall_at_k(chunks: list[dict], evidence: list[dict], k: int = 3) -> float:
    """Fraction of labeled evidence items appearing in the top-k chunks."""
    cov = covered(chunks[:k], evidence)
    return sum(cov) / len(cov)


# ---------- answer / citation metrics ----------
def cited_indices(answer: str) -> list[int]:
    return sorted({int(m) for m in CITE.findall(answer)})


def citation_scores(answer: str, sources: list[dict], evidence: list[dict]) -> dict:
    """precision: share of cited passages that are labeled evidence. recall: share of labeled evidence that is cited."""
    idx = cited_indices(answer)
    valid = bool(idx) and all(1 <= i <= len(sources) for i in idx)
    if not valid:
        return {"has_valid_citations": False, "precision": 0.0, "recall": 0.0}
    cited = [sources[i - 1] for i in idx]
    precision = sum(any(matches(c, e) for e in evidence) for c in cited) / len(cited)
    recall = sum(any(matches(c, e) for c in cited) for e in evidence) / len(evidence)
    return {"has_valid_citations": True, "precision": precision, "recall": recall}


def numbers_grounded(answer: str, sources: list[dict]) -> bool:
    """Cheap deterministic groundedness proxy: every number in the answer appears in the cited-context text.
    Misses non-numeric claims and number words ('five million'), so it is a floor, not a full check."""
    text = norm(" ".join(s["text"] for s in sources))
    nums = [n.rstrip(".,") for n in re.findall(r"\d[\d,]*\.?\d*", CITE.sub("", answer))]
    return all(n in text for n in nums)


def must_include_ok(answer: str, must_include: list[str]) -> bool:
    a = norm(answer)
    return all(norm(m) in a for m in must_include)


def llm_judge_grounded(answer: str, sources: list[dict]) -> bool | None:
    """Optional LLM-as-judge: is every factual claim supported by the passages? None if the judge call fails."""
    if not config.OPENAI_API_KEY:
        return None
    from openai import OpenAI, OpenAIError
    context = "\n\n".join(f"[{i}] {s['text']}" for i, s in enumerate(sources, 1))
    try:
        r = OpenAI(api_key=config.OPENAI_API_KEY).chat.completions.create(
            model=config.OPENAI_MODEL, temperature=0,
            messages=[
                {"role": "system", "content": "You are a strict fact-checker. Reply with exactly SUPPORTED if every factual "
                 "claim in the ANSWER is supported by the PASSAGES, otherwise reply with exactly UNSUPPORTED."},
                {"role": "user", "content": f"PASSAGES:\n{context}\n\nANSWER:\n{answer}"},
            ],
        )
    except OpenAIError:
        return None
    return (r.choices[0].message.content or "").strip().upper().startswith("SUPPORTED")


# ---------- runner ----------
def _mean(xs: list[float]) -> float | None:
    return round(sum(xs) / len(xs), 3) if xs else None


def run_eval(dataset: list[dict] | None = None, k: int = 3, retrieval_only: bool = False, judge: bool = False) -> dict:
    dataset = dataset if dataset is not None else load_dataset()
    records = []
    for q in dataset:
        chunks = rag.retrieve(q["question"], k)  # raw retrieval, before the evidence threshold
        rec = {
            "id": q["id"], "type": q["type"], "question": q["question"], "expected_abstain": q["expected_abstain"],
            "retrieved": [{"chunk_id": c["chunk_id"], "doc_id": c["doc_id"], "page": c["page"],
                           "score": round(c["score"], 4)} for c in chunks],
            "best_score": round(chunks[0]["score"], 4) if chunks else None,
        }
        if q["evidence"]:
            rec["hit_at_k"] = hit_at_k(chunks, q["evidence"], k)
            rec["recall_at_k"] = recall_at_k(chunks, q["evidence"], k)
        if not retrieval_only:
            res = rag.answer_question(q["question"])
            rec.update(answer=res["answer"], abstained=res["abstained"], abstain_reason=res["abstain_reason"])
            if q["evidence"] and not res["abstained"]:
                rec["answer_correct"] = must_include_ok(res["answer"], q["must_include"])
                rec["numbers_grounded"] = numbers_grounded(res["answer"], res["sources"])
                rec["citations"] = citation_scores(res["answer"], res["sources"], q["evidence"])
                if judge:
                    rec["llm_grounded"] = llm_judge_grounded(res["answer"], res["sources"])
        records.append(rec)
    return {
        "run_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "config": {"k": k, "embed_model": config.EMBED_MODEL, "llm": None if retrieval_only else config.OPENAI_MODEL,
                   "max_distance": config.MAX_DISTANCE, "retrieval_only": retrieval_only},
        "summary": summarize(records, retrieval_only),
        "records": records,
    }


def summarize(records: list[dict], retrieval_only: bool) -> dict:
    answerable = [r for r in records if not r["expected_abstain"]]
    unsupported = [r for r in records if r["expected_abstain"]]
    s = {
        "n_answerable": len(answerable), "n_unsupported": len(unsupported),
        "hit_at_k": _mean([float(r["hit_at_k"]) for r in answerable]),
        "recall_at_k": _mean([r["recall_at_k"] for r in answerable]),
        "best_score_answerable": [r["best_score"] for r in answerable],
        "best_score_unsupported": [r["best_score"] for r in unsupported],
    }
    if retrieval_only:
        return s
    answered = [r for r in answerable if not r["abstained"]]
    s.update(
        false_abstention_rate=_mean([float(r["abstained"]) for r in answerable]),
        abstention_rate_on_unsupported=_mean([float(r["abstained"]) for r in unsupported]),
        answer_correct_rate=_mean([float(r["answer_correct"]) for r in answered]),
        numbers_grounded_rate=_mean([float(r["numbers_grounded"]) for r in answered]),
        citation_precision=_mean([r["citations"]["precision"] for r in answered]),
        citation_recall=_mean([r["citations"]["recall"] for r in answered]),
    )
    judged = [r["llm_grounded"] for r in answered if r.get("llm_grounded") is not None]
    if judged:
        s["llm_grounded_rate"] = _mean([float(j) for j in judged])
    return s
