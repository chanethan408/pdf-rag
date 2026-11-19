import argparse
import sys

from app import ingest, rag

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Ingest policy PDFs into the vector index.")
    ap.add_argument("--rebuild", action="store_true", help="drop the index and re-ingest everything")
    ap.add_argument("--check", action="store_true", help="only verify index/registry consistency")
    args = ap.parse_args()

    if args.check:
        problems = ingest.check_consistency()
        print("consistent" if not problems else "\n".join(problems))
        sys.exit(1 if problems else 0)

    for name in ingest.restore_samples():
        print(f"restored   {name}  (default document)")
    results = ingest.ingest(rebuild=args.rebuild)
    for r in results:
        print(f"{r['status']:<10} {r['file']}" + (f"  ({r['reason']})" if r["reason"] else ""))
    print(f"Active chunks: {rag.index_count()}")
    sys.exit(1 if any(r["status"] == "failed" for r in results) else 0)
