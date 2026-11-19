import logging
import os
import re
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

import pymupdf
from fastapi import FastAPI, Request
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from starlette.concurrency import run_in_threadpool

from . import config, ingest, rag
from .logging_utils import request_id_var
from .schemas import AskRequest, AskResponse, ErrorResponse, HealthResponse, UploadResponse

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("api")

@asynccontextmanager
async def lifespan(_: FastAPI):
    """On startup, restore any default sample document that is missing (e.g. removed in the last session)."""
    if config.RESTORE_SAMPLES_ON_START:
        try:
            restored = await run_in_threadpool(ingest.restore_samples)
            if restored:
                log.info("Restored default documents: %s", restored)
        except Exception as e:  # never block startup on this
            log.error("Could not restore default documents: %s", type(e).__name__)
    yield


app = FastAPI(title="Credit Policy Intelligence API", version="0.1.0", lifespan=lifespan)

# Demo UI + the source PDFs (so a citation can be opened at its page). Synthetic documents only.
app.mount("/pdfs", StaticFiles(directory=config.PDF_DIR, check_dir=False), name="pdfs")


NAME_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9 _.\-]{0,100}\.pdf", re.IGNORECASE)


@app.get("/documents")
def documents():
    """The policy PDFs in the document folder, and whether each is in the search index (for the UI sidebar)."""
    registry = ingest.load_registry()
    items = []
    for pdf in sorted(Path(config.PDF_DIR).glob("*.pdf")):
        try:
            with pymupdf.open(pdf) as d:
                pages = len(d)
        except Exception:
            continue  # unreadable file: leave it out of the list
        items.append({"filename": pdf.name, "title": pdf.stem.replace("_", " ").title(),
                      "pages": pages, "indexed": pdf.stem in registry})
    return items


@app.put("/documents/{filename}", response_model=UploadResponse,
         responses={400: {"model": ErrorResponse}, 413: {"model": ErrorResponse}, 422: {"model": ErrorResponse}})
async def upload_document(filename: str, request: Request):
    """Add or replace a PDF. The raw PDF bytes are the request body. Goes through the same versioned ingestion as the CLI."""
    if not NAME_RE.fullmatch(filename):
        return _err(400, "invalid_filename", "Use letters, numbers, spaces, dots, dashes or underscores, ending in .pdf.")
    declared = request.headers.get("content-length", "")
    if declared.isdigit() and int(declared) > config.MAX_UPLOAD_BYTES:
        return _err(413, "file_too_large", f"Maximum upload size is {config.MAX_UPLOAD_BYTES // (1024 * 1024)} MB.")
    data = await request.body()
    if len(data) > config.MAX_UPLOAD_BYTES:
        return _err(413, "file_too_large", f"Maximum upload size is {config.MAX_UPLOAD_BYTES // (1024 * 1024)} MB.")
    if not data.startswith(b"%PDF"):
        return _err(400, "not_a_pdf", "That file is not a PDF.")

    # Stage first: a bad upload must never overwrite a good document that is already indexed.
    staged = Path(config.PDF_DIR) / ".staging" / filename
    staged.parent.mkdir(parents=True, exist_ok=True)
    staged.write_bytes(data)
    try:
        result = await run_in_threadpool(ingest.ingest_one, staged)
        if result["status"] == "failed":
            return _err(422, "ingest_failed", f"Could not index this PDF: {result['reason']}.")
        os.replace(staged, Path(config.PDF_DIR) / filename)
    finally:
        staged.unlink(missing_ok=True)
    return UploadResponse(filename=filename, status=result["status"], version=result["version"], chunks=result["chunks"])


@app.delete("/documents/{filename}", responses={400: {"model": ErrorResponse}, 404: {"model": ErrorResponse}})
def delete_document(filename: str):
    """Remove a document from the search index and delete its file."""
    if not NAME_RE.fullmatch(filename):
        return _err(400, "invalid_filename", "Invalid filename.")
    path = Path(config.PDF_DIR) / filename
    was_indexed = ingest.remove_document(path.stem)
    had_file = path.exists()
    path.unlink(missing_ok=True)
    if not was_indexed and not had_file:
        return _err(404, "not_found", "No such document.")
    return {"filename": filename, "removed": True}


@app.get("/", include_in_schema=False)
def ui():
    return FileResponse(Path(__file__).parent / "static" / "index.html")


@app.middleware("http")
async def request_logging(request: Request, call_next):
    rid = uuid.uuid4().hex[:12]
    request_id_var.set(rid)
    start = time.perf_counter()
    try:
        response = await call_next(request)
    except Exception as exc:  # unhandled: logged here, generic 500 produced by the exception handler
        log.error("request request_id=%s method=%s path=%s status=500 latency_ms=%.1f error=%s",
                  rid, request.method, request.url.path, (time.perf_counter() - start) * 1000, type(exc).__name__)
        raise
    response.headers["X-Request-ID"] = rid
    log.info("request request_id=%s method=%s path=%s status=%d latency_ms=%.1f",
             rid, request.method, request.url.path, response.status_code, (time.perf_counter() - start) * 1000)
    return response


def _err(status: int, error: str, detail: str) -> JSONResponse:
    return JSONResponse(status_code=status, content=ErrorResponse(error=error, detail=detail).model_dump())


@app.exception_handler(rag.IndexNotReadyError)
async def _index(_: Request, exc: rag.IndexNotReadyError):
    return _err(503, "index_not_ready", str(exc))


@app.exception_handler(rag.LLMError)
async def _llm(_: Request, exc: rag.LLMError):
    return _err(502, "llm_error", str(exc))


@app.exception_handler(Exception)
async def _unexpected(_: Request, exc: Exception):
    log.error("Unhandled error: %s", type(exc).__name__)  # no stack trace / secrets to client
    return _err(500, "internal_error", "An unexpected error occurred.")


@app.get("/health", response_model=HealthResponse, responses={503: {"model": HealthResponse}})
def health():
    n = rag.index_count()
    body = HealthResponse(status="ok" if n else "degraded", index_ready=n > 0, indexed_chunks=n)
    return JSONResponse(status_code=200 if n else 503, content=body.model_dump())


@app.post(
    "/ask",
    response_model=AskResponse,
    responses={502: {"model": ErrorResponse}, 503: {"model": ErrorResponse}},
)
def ask(req: AskRequest):
    return rag.answer_question(req.question)
