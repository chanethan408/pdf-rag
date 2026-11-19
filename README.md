# Credit Policy Intelligence API

A small RAG service for banking credit policy PDFs. You ask a question, it answers from the documents and
cites the page. If the documents don't cover it, it says so instead of guessing.

Stack: FastAPI, ChromaDB, sentence-transformers, OpenAI. The PDFs are synthetic.

## Setup

Needs Python 3.10+ and an OpenAI API key.

```
python -m venv .venv
.venv\Scripts\activate          # macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
copy .env.example .env          # macOS/Linux: cp .env.example .env
```

Put your key in `.env` as `OPENAI_API_KEY=...`, then start the server:

```
uvicorn app.main:app --reload
```

The first start downloads the embedding model (~90 MB) and indexes the three sample policies, so it takes a
minute.

- UI: http://127.0.0.1:8000/
- API docs: http://127.0.0.1:8000/docs

## Usage

Click an example question or type your own. Click a citation like `[1]` to see the source page. The sidebar
lets you upload or remove PDFs.

Try "What is the minimum DSCR for commercial loans?" (answers with a citation) and "What is the current
30-year mortgage rate?" (declines).

The sample PDFs are in `data/samples/` and get restored on every startup, so removing one only lasts until
the next restart. Working copies live in `data/pdfs/`.

```
curl -X POST http://127.0.0.1:8000/ask -H "Content-Type: application/json" -d "{\"question\": \"What is the minimum DSCR?\"}"
```

Other endpoints: `GET /documents`, `PUT /documents/{filename}`, `DELETE /documents/{filename}`, `GET /health`.
Upload and delete have no authentication, so keep the server on localhost.

## How it works

1. PDFs are chunked and embedded into ChromaDB. Re-ingesting skips unchanged files, and an updated file only
   replaces the old version once it's fully indexed.
2. A question pulls the 3 closest chunks.
3. If the best one is too far away (distance > 0.55), it declines without calling the LLM.
4. Otherwise the LLM answers from those passages only and cites them, or says there isn't enough evidence.

## Tests and evaluation

```
pytest                                        # no API key needed
python -m scripts.evaluate --retrieval-only   # free
python -m scripts.evaluate                    # full run, small API cost
```

On the 10 questions in `data/eval/questions.json`, 8 of 8 answerable ones were answered correctly with the right
citations, and both unanswerable ones were declined. The set is tiny and I picked the 0.55 cutoff using it, so
this is a sanity check, not a real accuracy number.

## Config

Only `OPENAI_API_KEY` is required. Optional `.env` settings: `OPENAI_MODEL`, `EMBED_MODEL`, `TOP_K`,
`EVIDENCE_MAX_DISTANCE`, `CHROMA_DIR`, `PDF_DIR`, `RESTORE_SAMPLES_ON_START`.

If you get a 502, check the API key. If you get a 503, nothing is indexed: restart or run
`python -m scripts.ingest`. After changing the embedding model, run `python -m scripts.ingest --rebuild`.

## Limitations

- One chunk per page, so long pages aren't split.
- The eval set is small and the cutoff is tuned to it. The distance is not a probability of being right.
- Uploaded PDFs go to the LLM as context, so prompt injection is possible. The prompt helps but I haven't
  tested it.
- No auth or per-user access control.
