import hashlib
import re

import pymupdf


def fake_embed(texts):
    """Deterministic bag-of-words embedding: no model download needed in tests."""
    out = []
    for t in texts:
        v = [0.0] * 64
        for w in re.findall(r"[a-z0-9.]+", t.lower()):
            v[int(hashlib.md5(w.encode()).hexdigest(), 16) % 64] += 1
        n = sum(x * x for x in v) ** 0.5 or 1
        out.append([x / n for x in v])
    return out


def make_pdf(path, pages):
    doc = pymupdf.open()
    for text in pages:
        doc.new_page().insert_textbox(pymupdf.Rect(50, 50, 550, 750), text, fontsize=11)
    doc.save(path)
