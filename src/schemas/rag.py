from pydantic import BaseModel, Field
from typing import Optional, List
from enum import Enum
from uuid import UUID


# ─── Enums ────────────────────────────────────────────────────────────────────

class ChunkingStrategy(str, Enum):
    RECURSIVE    = "recursive"     # Fast, paragraph→sentence→word splits
    SEMANTIC     = "semantic"      # Embedding-based semantic boundary detection
    PARENT_CHILD = "parent_child"  # Small child chunks retrieved, large parent chunks sent to LLM


class EmbeddingModelChoice(str, Enum):
    BGE_M3    = "BAAI/bge-m3"           # Multilingual | dense + sparse + colbert | 1024-dim
    BGE_LARGE = "BAAI/bge-large-en-v1.5"  # English-only | dense-only | 1024-dim


class RetrievalStrategy(str, Enum):
    DENSE  = "dense"   # ANN vector search — great for conceptual/paraphrase queries
    SPARSE = "sparse"  # Lexical/keyword search — great for exact terms, codes, names (bge-m3 only)
    HYBRID = "hybrid"  # Dense + Sparse fused with RRF — best overall recall (bge-m3 only)


# ─── RAG Configuration ────────────────────────────────────────────────────────

class RAGConfig(BaseModel):
    # Core strategy choices
    chunking_strategy:  ChunkingStrategy    = ChunkingStrategy.RECURSIVE
    embedding_model:    EmbeddingModelChoice = EmbeddingModelChoice.BGE_M3
    retrieval_strategy: RetrievalStrategy   = RetrievalStrategy.HYBRID

    # Recursive / Semantic chunk sizing
    chunk_size:    int = Field(1000, ge=100, le=4000,  description="Max chars per chunk")
    chunk_overlap: int = Field(150,  ge=0,   le=500,   description="Overlap between consecutive chunks")

    # Parent-Child specific
    parent_chunk_size: int = Field(2000, ge=500,  le=8000, description="Parent chunk size (context window)")
    child_chunk_size:  int = Field(400,  ge=100,  le=1000, description="Child chunk size (retrieval unit)")
    child_overlap:     int = Field(50,   ge=0,    le=200)

    # Retrieval
    retrieval_k: int = Field(6, ge=1, le=20, description="Number of chunks to retrieve")

    # Semantic chunker tuning
    semantic_breakpoint_threshold: float = Field(
        0.35, ge=0.1, le=0.9,
        description="Cosine distance threshold for semantic boundary detection"
    )


# ─── Request Schemas ──────────────────────────────────────────────────────────

class IngestRequest(BaseModel):
    config: RAGConfig = RAGConfig()


class QueryRequest(BaseModel):
    question:        str
    document_ids:    Optional[List[UUID]] = None  # None → search all user's ready documents
    config:          RAGConfig = RAGConfig()
    include_sources: bool = True
    max_answer_tokens: int = Field(2048, ge=256, le=8192)


# ─── Response Schemas ─────────────────────────────────────────────────────────

class SourceChunk(BaseModel):
    document_id:  str
    filename:     str
    content:      str
    score:        float
    page_number:  Optional[int] = None
    chunk_type:   str = "text"   # text | table | parent


class QueryResponse(BaseModel):
    answer:             str
    sources:            Optional[List[SourceChunk]] = None
    model_used:         str
    retrieval_strategy: str
    chunks_retrieved:   int


class IngestResponse(BaseModel):
    document_id:       str
    chunks_created:    int
    embedding_model:   str
    chunking_strategy: str
    status:            str
    message:           str