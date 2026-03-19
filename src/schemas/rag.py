from pydantic import BaseModel, Field
from typing import Optional, List, Any
from enum import Enum
from uuid import UUID


# ─── Enums ────────────────────────────────────────────────────────────────────

class ChunkingStrategy(str, Enum):
    RECURSIVE    = "recursive"
    SEMANTIC     = "semantic"
    PARENT_CHILD = "parent_child"


class EmbeddingModelChoice(str, Enum):
    BGE_M3    = "BAAI/bge-m3"
    BGE_LARGE = "BAAI/bge-large-en-v1.5"


class RetrievalStrategy(str, Enum):
    DENSE  = "dense"
    SPARSE = "sparse"
    HYBRID = "hybrid"


# ─── RAG Configuration ────────────────────────────────────────────────────────

class RAGConfig(BaseModel):
    chunking_strategy:             ChunkingStrategy     = ChunkingStrategy.RECURSIVE
    embedding_model:               EmbeddingModelChoice = EmbeddingModelChoice.BGE_M3
    retrieval_strategy:            RetrievalStrategy    = RetrievalStrategy.HYBRID
    chunk_size:                    int = Field(1000, ge=100,  le=4000)
    chunk_overlap:                 int = Field(150,  ge=0,    le=500)
    parent_chunk_size:             int = Field(2000, ge=500,  le=8000)
    child_chunk_size:              int = Field(400,  ge=100,  le=1000)
    child_overlap:                 int = Field(50,   ge=0,    le=200)
    retrieval_k:                   int = Field(6,    ge=1,    le=20)
    semantic_breakpoint_threshold: float = Field(0.35, ge=0.1, le=0.9)


# ─── Request Schemas ──────────────────────────────────────────────────────────

class IngestRequest(BaseModel):
    config: RAGConfig = RAGConfig()


class QueryRequest(BaseModel):
    question:          str
    document_ids:      Optional[List[UUID]] = None
    config:            RAGConfig = RAGConfig()
    include_sources:   bool = True
    max_answer_tokens: int = Field(2048, ge=256, le=8192)
    session_id:        Optional[UUID] = None


# ─── Response Schemas ─────────────────────────────────────────────────────────

class SourceChunk(BaseModel):
    document_id:  str
    filename:     str
    content:      str
    score:        float
    page_number:  Optional[int] = None
    chunk_type:   str = "text"


class QueryResponse(BaseModel):
    answer:             str
    sources:            Optional[List[SourceChunk]] = None
    model_used:         str
    retrieval_strategy: str
    chunks_retrieved:   int
    session_id:         Optional[str] = None


class IngestResponse(BaseModel):
    document_id:       str
    chunks_created:    int
    embedding_model:   str
    chunking_strategy: str
    status:            str
    message:           str


# ─── Task / Async schemas ─────────────────────────────────────────────────────

class TaskStatusResponse(BaseModel):
    """
    Returned immediately from POST /rag/ingest and by GET /rag/tasks/{task_id}.

    States:  PENDING | STARTED | SUCCESS | FAILURE | RETRY | REVOKED
    """
    task_id:     str
    document_id: Optional[str]  = None
    state:       str            = "PENDING"
    message:     str            = ""
    result:      Optional[dict] = None   # populated on SUCCESS
    error:       Optional[str]  = None   # populated on FAILURE
    meta:        Optional[dict] = None   # populated on STARTED (custom progress)