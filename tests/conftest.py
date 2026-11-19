import pytest

from app import config, rag
from tests.helpers import fake_embed, make_pdf


@pytest.fixture
def env(tmp_path, monkeypatch):
    """Temp PDF dir + temp Chroma dir + fake embeddings. Never touches the real index or any API."""
    pdfs = tmp_path / "pdfs"
    pdfs.mkdir()
    monkeypatch.setattr(config, "CHROMA_DIR", str(tmp_path / "chroma"))
    monkeypatch.setattr(config, "PDF_DIR", str(pdfs))
    monkeypatch.setattr(config, "MAX_DISTANCE", None)  # tests set this explicitly when needed
    monkeypatch.setattr(rag, "embed", fake_embed)
    make_pdf(pdfs / "lending.pdf", ["The minimum DSCR for commercial loans is 1.25x.\n\nLoans need 24 months history."])
    make_pdf(pdfs / "docs.pdf", ["Loan files are retained for 7 years after maturity."])
    return pdfs
