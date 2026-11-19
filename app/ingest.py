"""Offline ingestion: PDFs -> chunks -> Chroma, with hashing, versioning and safe replacement.

Design:
- Chunk IDs are deterministic: {doc_id}-{hash12}-p{page}-c{n}.
- A registry file (registry.json next to the Chroma data) records each document's active hash and version.
- A changed document is indexed as INACTIVE first. Only after all its chunks are stored are the new chunks
  activated and the old ones deactivated, in a single Chroma update. A failure before that leaves the
  previous version fully usable.
- Retrieval only sees chunks with active=True.
- Documents deleted from the PDF folder by hand are NOT auto-removed from the index; use remove_document()
  (the DELETE /documents/{filename} endpoint) or --rebuild.
"""
import hashlib
import json
import logging
import os
import shutil
import threading
import time
from pathlib import Path

import pymupdf

from . import config, rag

log = logging.getLogger("ingest")
_write_lock = threading.Lock()  # serialises index writes from concurrent upload/remove requests


class DocumentError(ValueError):
    """A PDF that cannot be ingested (unreadable, empty, no text)."""


# ---------- registry ----------
def registry_path() -> Path:
    return Path(config.CHROMA_DIR) / "registry.json"


def load_registry() -> dict:
    p = registry_path()
    return json.loads(p.read_text()) if p.exists() else {}


def save_registry(reg: dict) -> None:
    p = registry_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(".tmp")
    tmp.write_text(json.dumps(reg, indent=2))
    os.replace(tmp, p)  # atomic: registry is never half-written


# ---------- extraction / chunking ----------
def file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def extract_pages(pdf_path: Path) -> list[tuple[int, str]]:
    try:
        with pymupdf.open(pdf_path) as doc:
            pages = [(i + 1, page.get_text().strip()) for i, page in enumerate(doc)]
    except Exception as e:
        raise DocumentError(f"unreadable PDF ({type(e).__name__})") from e
    if not any(text for _, text in pages):
        raise DocumentError("PDF has no extractable text")
    return pages


def chunk_page(text: str, max_chars: int = config.CHUNK_CHARS) -> list[str]:
    """Pack paragraphs into chunks <= max_chars without splitting a paragraph."""
    chunks, cur = [], ""
    for para in (p.strip() for p in text.split("\n\n") if p.strip()):
        if cur and len(cur) + len(para) + 2 > max_chars:
            chunks.append(cur)
            cur = para
        else:
            cur = f"{cur}\n\n{para}" if cur else para
    if cur:
        chunks.append(cur)
    return chunks


def build_chunks(pdf: Path, doc_hash: str, version: int) -> list[dict]:
    doc_id, out, n = pdf.stem, [], 0
    for page_no, text in extract_pages(pdf):
        for piece in chunk_page(text):
            cid = f"{doc_id}-{doc_hash[:12]}-p{page_no}-c{n}"
            out.append({
                "id": cid,
                "text": piece,
                "meta": {
                    "doc_id": doc_id, "filename": pdf.name, "page": page_no, "chunk_id": cid,
                    "doc_hash": doc_hash, "version": version, "active": False,
                },
            })
            n += 1
    if not out:
        raise DocumentError("PDF produced no chunks")
    return out


# ---------- ingestion ----------
def _ingest_document(col, reg: dict, pdf: Path) -> str:
    doc_id, h = pdf.stem, file_hash(pdf)
    entry = reg.get(doc_id)
    if entry and entry["active_hash"] == h:
        return "unchanged"

    version = entry["version"] + 1 if entry else 1
    chunks = build_chunks(pdf, h, version)  # may raise DocumentError -> nothing has been written yet
    ids = [c["id"] for c in chunks]
    embeddings = rag.embed([c["text"] for c in chunks])

    # 1. Store new version as inactive (upsert + deterministic ids => safe to retry).
    col.upsert(ids=ids, documents=[c["text"] for c in chunks], embeddings=embeddings,
               metadatas=[c["meta"] for c in chunks])

    # 2. Switch: activate new chunks and deactivate the old version in ONE update call.
    old = col.get(where={"$and": [{"doc_id": doc_id}, {"active": True}]}, include=["metadatas"])
    new_metas = [{**c["meta"], "active": True} for c in chunks]
    old_metas = [{**m, "active": False} for m in old["metadatas"]]
    col.update(ids=ids + old["ids"], metadatas=new_metas + old_metas)

    # 3. Record the new active version.
    reg[doc_id] = {
        "filename": pdf.name, "active_hash": h, "version": version,
        "chunk_count": len(chunks), "ingested_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    save_registry(reg)
    return "added" if version == 1 else "updated"


def ingest(pdf_dir: str | None = None, rebuild: bool = False) -> list[dict]:
    """Ingest every PDF in pdf_dir. Returns one result dict per file; one bad file never blocks the others."""
    pdfs = sorted(Path(pdf_dir or config.PDF_DIR).glob("*.pdf"))
    if not pdfs:
        raise FileNotFoundError("No PDFs found to ingest")

    client = rag.get_client()
    # No registry => first run, or a pre-Level-2 index with incompatible metadata: start clean.
    if rebuild or not registry_path().exists():
        try:
            client.delete_collection(config.COLLECTION)
        except Exception:
            pass
        if registry_path().exists():
            registry_path().unlink()
    col = client.get_or_create_collection(config.COLLECTION, metadata={"hnsw:space": "cosine"})

    reg = load_registry()
    results = []
    for pdf in pdfs:
        try:
            status, reason = _ingest_document(col, reg, pdf), None
        except DocumentError as e:
            status, reason = "failed", str(e)
        except Exception as e:  # e.g. embedding failure; keep previous version active
            log.error("Ingest of %s failed: %s", pdf.name, type(e).__name__)
            status, reason = "failed", f"unexpected error ({type(e).__name__})"
        results.append({"file": pdf.name, "status": status, "reason": reason})
    return results


def ingest_one(pdf: Path) -> dict:
    """Ingest a single PDF (used by the upload endpoint). Never raises for a bad document."""
    with _write_lock:
        col = rag.get_client().get_or_create_collection(config.COLLECTION, metadata={"hnsw:space": "cosine"})
        reg = load_registry()
        try:
            status, reason = _ingest_document(col, reg, pdf), None
        except DocumentError as e:
            status, reason = "failed", str(e)
        except Exception as e:
            log.error("Ingest of %s failed: %s", pdf.name, type(e).__name__)
            status, reason = "failed", f"unexpected error ({type(e).__name__})"
        entry = reg.get(pdf.stem) or {}
    return {"file": pdf.name, "status": status, "reason": reason,
            "version": entry.get("version"), "chunks": entry.get("chunk_count")}


def restore_samples() -> list[str]:
    """Put back any default sample PDF that is missing from the working folder or the index.

    A sample that is already indexed is left alone, so a newer version the user uploaded under the same
    name is never overwritten. Returns the filenames that were restored."""
    src_dir, dst_dir = Path(config.SAMPLES_DIR), Path(config.PDF_DIR)
    dst_dir.mkdir(parents=True, exist_ok=True)
    restored = []
    for src in sorted(src_dir.glob("*.pdf")):
        dst = dst_dir / src.name
        if dst.exists() and src.stem in load_registry():
            continue
        if not dst.exists():
            shutil.copyfile(src, dst)
        result = ingest_one(dst)
        if result["status"] == "failed":
            log.warning("Could not restore sample %s: %s", src.name, result["reason"])
        else:
            restored.append(src.name)
    return restored


def remove_document(doc_id: str) -> bool:
    """Delete a document's chunks (all versions) and its registry entry. Returns False if it was not indexed."""
    with _write_lock:
        reg = load_registry()
        existed = doc_id in reg
        try:
            col = rag.get_client().get_collection(config.COLLECTION)
            ids = col.get(where={"doc_id": doc_id}, include=[])["ids"]
            if ids:
                col.delete(ids=ids)
                existed = True
        except Exception:
            pass  # no collection yet: nothing to delete from the index
        if doc_id in reg:
            del reg[doc_id]
            save_registry(reg)
        return existed


def check_consistency() -> list[str]:
    """Return a list of problems; empty means active chunks and the registry agree."""
    reg = load_registry()
    col = rag.get_client().get_collection(config.COLLECTION)
    problems = []
    metas = col.get(where=rag.ACTIVE, include=["metadatas"])["metadatas"]
    for m in metas:
        entry = reg.get(m["doc_id"])
        if entry is None:
            problems.append(f"active chunk {m['chunk_id']} references unknown document {m['doc_id']}")
        elif m["doc_hash"] != entry["active_hash"]:
            problems.append(f"active chunk {m['chunk_id']} is from a superseded version")
    for doc_id, entry in reg.items():
        n = sum(1 for m in metas if m["doc_id"] == doc_id)
        if n != entry["chunk_count"]:
            problems.append(f"{doc_id}: {n} active chunks, registry expects {entry['chunk_count']}")
    return problems
