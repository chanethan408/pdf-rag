# Credit Policy Intelligence API

Ask questions about banking credit policy PDFs and get answers that **cite the exact page**, or a clear
"not enough evidence" when the documents don't cover it.

It's a RAG (retrieval-augmented generation) service built with FastAPI, ChromaDB, sentence-transformers and
OpenAI. The bundled documents are synthetic; no real bank data is included.

## Quick start

You need Python 3.10+ and an OpenAI API key (a few dollars of credit goes a long way with `gpt-4o-mini`).

```
python -m venv .venv
.venv\Scripts\activate          # macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
copy .env.example .env          # macOS/Linux: cp .env.example .env
```

Open `.env` and set `OPENAI_API_KEY=your-key-here`. Then:

```
uvicorn app.main:app --reload
```

The first start downloads a small embedding model (~90 MB) and indexes the three default policies, so give it
a minute. Then open:

- http://127.0.0.1:8000/ for the demo UI
- http://127.0.0.1:8000/docs for the interactive API docs

## Using it

1. Click an example question or type your own.
2. Click a citation like `[1]`, then **Open PDF page** to see the source.
3. Use the sidebar to **upload** a PDF or **remove** one. Changes take effect immediately.

Try: "What is the minimum DSCR for commercial loans?" (answered with a citation) and "What is the current
30-year mortgage rate?" (declined, since the documents don't say).

The three default documents live in `data/samples/` and come back at every startup, so removing one is only
temporary. The working copies are in `data/pdfs/` (gitignored).

## API

```
curl -X POST http://127.0.0.1:8000/ask -H "Content-Type: application/json" -d "{\"question\": \"What is the minimum DSCR?\"}"
```

| Endpoint | Purpose |
|---|---|
| `POST /ask` | Ask a question |
| `GET /documents` | List documents |
| `PUT /documents/{filename}` | Upload or update a PDF (10 MB max) |
| `DELETE /documents/{filename}` | Remove a document |
| `GET /health` | Status |

Upload and remove have **no authentication**, so keep the server on localhost.

## How it works

1. PDFs are split into chunks and embedded into ChromaDB. Re-ingesting is safe: unchanged files are skipped and
   updated files become a new version that only goes live once fully indexed.
2. A question retrieves the 3 closest chunks.
3. If even the best chunk is too far away (distance above 0.55), it declines without calling the LLM.
4. Otherwise the LLM answers only from those passages, cites them, or says there isn't enough evidence.

## Tests and evaluation

```
pytest                                        # 48 tests, no API key needed
python -m scripts.evaluate --retrieval-only   # free retrieval check
python -m scripts.evaluate                    # full run (small API cost)
```

On the 10 labeled questions in `data/eval/questions.json`: 8/8 answered correctly with correct citations, and
2/2 unsupported questions declined. That's a small demo set and the threshold was tuned on it, so treat it as
a sanity check, not proof of accuracy.

## Settings (`.env`)

Only the API key is required. Optional: `OPENAI_MODEL`, `EMBED_MODEL`, `TOP_K` (3),
`EVIDENCE_MAX_DISTANCE` (0.55), `CHROMA_DIR`, `PDF_DIR`, `RESTORE_SAMPLES_ON_START` (true).

## Troubleshooting

- **502 error:** the API key is missing, wrong, or out of credit.
- **503 error:** nothing is indexed. Restart the server or run `python -m scripts.ingest`.
- **Changed the embedding model:** run `python -m scripts.ingest --rebuild`.

## Known limitations

- Chunking is one chunk per page, so long pages aren't split further.
- The evaluation set is tiny and the distance cutoff is a tuned value, not a probability of being right.
- A malicious uploaded PDF could attempt prompt injection. The prompt mitigates this but it isn't tested.
- No authentication or per-user access control.
- Logs never record questions, document text, or keys.

Never commit `.env`.
