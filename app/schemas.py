from pydantic import BaseModel, Field, field_validator


class AskRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=1000)

    @field_validator("question")
    @classmethod
    def not_blank(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("question must not be blank")
        return v


class Source(BaseModel):
    doc_id: str
    filename: str
    page: int
    chunk_id: str
    version: int | None = None
    score: float = Field(..., description="Cosine distance (lower = closer). Not a probability.")
    text: str


class AskResponse(BaseModel):
    question: str
    answer: str
    abstained: bool
    abstain_reason: str | None = None  # "no_relevant_evidence" (retrieval check) | "llm_abstained"
    best_score: float | None = Field(None, description="Cosine distance of the closest chunk (lower = closer).")
    sources: list[Source]


class HealthResponse(BaseModel):
    status: str
    index_ready: bool
    indexed_chunks: int


class UploadResponse(BaseModel):
    filename: str
    status: str  # added | updated | unchanged
    version: int | None = None
    chunks: int | None = None


class ErrorResponse(BaseModel):
    error: str
    detail: str
