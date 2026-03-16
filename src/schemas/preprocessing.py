from pydantic import BaseModel, field_validator
from typing import Optional, List
from uuid import UUID
from datetime import datetime


# ─── Request ─────────────────────────────────────────────────────────────────

class ChatMessage(BaseModel):
    role: str          # "human" or "assistant"
    content: str


class QueryRequest(BaseModel):
    question: str
    document_ids: Optional[List[UUID]] = None   # None = search all user's docs
    chat_history: List[ChatMessage] = []
    include_sources: bool = True

    @field_validator("question")
    @classmethod
    def question_not_empty(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("Question cannot be empty.")
        if len(v) > 2000:
            raise ValueError("Question must be under 2000 characters.")
        return v


# ─── Response ─────────────────────────────────────────────────────────────────

class SourceDocument(BaseModel):
    document_id: str
    original_filename: str
    page: Optional[int] = None
    chunk_index: Optional[int] = None


class QueryResponse(BaseModel):
    answer: str
    sources: List[SourceDocument] = []
    question: str
    model_used: str


class IngestResponse(BaseModel):
    document_id: str
    original_filename: str
    chunk_count: int
    status: str
    message: str