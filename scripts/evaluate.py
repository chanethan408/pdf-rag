import argparse
import json
from pathlib import Path

from app import evaluation

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Evaluate retrieval, answers, citations and abstention.")
    ap.add_argument("--retrieval-only", action="store_true", help="no LLM calls (free); retrieval metrics + scores")
    ap.add_argument("--judge", action="store_true", help="also run an LLM groundedness judge (extra API calls)")
    ap.add_argument("--out", default="data/eval/results.json")
    args = ap.parse_args()

    result = evaluation.run_eval(retrieval_only=args.retrieval_only, judge=args.judge)
    Path(args.out).write_text(json.dumps(result, indent=2), encoding="utf-8")

    for r in result["records"]:
        line = f"{r['id']} {r['type']:<14} best={r['best_score']}"
        if "hit_at_k" in r:
            line += f" hit={int(r['hit_at_k'])} recall={r['recall_at_k']:.2f}"
        if "abstained" in r:
            line += f" abstained={r['abstained']}"
            if "answer_correct" in r:
                c = r["citations"]
                line += f" correct={r['answer_correct']} nums_ok={r['numbers_grounded']} cite_p={c['precision']:.2f} cite_r={c['recall']:.2f}"
        print(line)
    print("\nSUMMARY")
    for k, v in result["summary"].items():
        print(f"  {k}: {v}")
    print(f"\nFull results (incl. every answer, for manual review): {args.out}")
