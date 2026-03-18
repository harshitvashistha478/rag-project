"""
RAG API router.

Endpoints
─────────
  POST  /rag/ingest/{document_id}   — chunk + embed + store a document
  POST  /rag/query                  — query across user's documents
  GET   /rag/config/defaults        — browsable strategy reference
  DELETE /rag/document/{document_id} — wipe vectors (keeps S3 / DB record)
"""

import logging
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from src.database import get_db
from src.models.document import Document, DocumentStatus
from src.models.user import User
from src.schemas.rag import (
    RAGConfig,
    IngestRequest,
    IngestResponse,
    QueryRequest,
    QueryResponse,
)
from src.utils.auth import get_current_active_user
from src.rag.pipeline import ingest_document, query_documents
from src.rag import vector_store as vs

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/rag", tags=["RAG"])


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _get_ready_document(document_id: UUID, user: User, db: Session) -> Document:
    """Fetch a non-deleted document owned by the user, or raise 404."""
    doc = (
        db.query(Document)
        .filter(
            Document.id == document_id,
            Document.user_id == user.id,
            Document.is_deleted == False,
        )
        .first()
    )
    if not doc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Document not found.")
    return doc


# ─── Ingest ───────────────────────────────────────────────────────────────────

@router.post(
    "/ingest/{document_id}",
    response_model=IngestResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Ingest a PDF document into the vector store",
)
def ingest(
    document_id: UUID,
    body:         IngestRequest         = IngestRequest(),
    current_user: User                  = Depends(get_current_active_user),
    db:           Session               = Depends(get_db),
):
    """
    Parse → chunk → embed → store a document in Milvus.

    The document must already be uploaded (status PENDING or FAILED).
    Only PDF files are currently supported for RAG ingestion.

    On success, the document status is updated to **READY** and `chunk_count`
    is populated.  On failure the status is set to **FAILED** with an error
    message.
    """
    doc = _get_ready_document(document_id, current_user, db)

    if doc.file_extension.lower() != "pdf":
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=(
                f"RAG ingestion only supports PDF files. "
                f"This document has extension '.{doc.file_extension}'."
            ),
        )

    if doc.status == DocumentStatus.PROCESSING:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Document is already being processed.",
        )

    # Mark as processing immediately
    doc.status = DocumentStatus.PROCESSING
    db.commit()

    try:
        chunk_count = ingest_document(
            document_id=str(doc.id),
            user_id=str(current_user.id),
            s3_key=doc.s3_key,
            original_filename=doc.original_filename,
            config=body.config,
        )

        doc.status          = DocumentStatus.READY
        doc.chunk_count     = chunk_count
        doc.processing_error = None
        db.commit()

        logger.info(
            f"[ingest] document={document_id} user={current_user.email} "
            f"chunks={chunk_count} model={body.config.embedding_model} "
            f"strategy={body.config.chunking_strategy}"
        )

        return IngestResponse(
            document_id=str(document_id),
            chunks_created=chunk_count,
            embedding_model=body.config.embedding_model,
            chunking_strategy=body.config.chunking_strategy,
            status="ready",
            message=f"Successfully ingested '{doc.original_filename}' — {chunk_count} chunks created.",
        )

    except HTTPException:
        # Re-raise S3 / storage errors directly
        doc.status           = DocumentStatus.FAILED
        doc.processing_error = "Storage error during ingestion."
        db.commit()
        raise

    except Exception as exc:
        doc.status           = DocumentStatus.FAILED
        doc.processing_error = str(exc)[:500]
        db.commit()
        logger.exception(f"[ingest] Failed for document={document_id}: {exc}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Ingestion failed: {exc}",
        )


# ─── Query ────────────────────────────────────────────────────────────────────

@router.post(
    "/query",
    response_model=QueryResponse,
    summary="Ask a question over your ingested documents",
)
def query(
    payload:      QueryRequest,
    current_user: User    = Depends(get_current_active_user),
    db:           Session = Depends(get_db),
):
    """
    Retrieve relevant chunks from the vector store and generate an answer
    using Gemini.

    If `document_ids` is omitted, the query searches across **all** of the
    user's READY documents.  Passing a list restricts the search scope.

    The `config` field controls which embedding model, chunking strategy
    (must match what was used during ingestion), and retrieval strategy to use.
    """
    if not payload.question.strip():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Question cannot be empty.",
        )

    # Validate + authorise requested document IDs
    doc_id_strs: list | None = None
    if payload.document_ids:
        doc_id_strs = []
        for doc_uuid in payload.document_ids:
            doc = _get_ready_document(doc_uuid, current_user, db)
            if doc.status != DocumentStatus.READY:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=(
                        f"Document '{doc.original_filename}' is not ready for querying "
                        f"(status: {doc.status.value}).  Please ingest it first."
                    ),
                )
            doc_id_strs.append(str(doc.id))

    try:
        result = query_documents(
            question=payload.question,
            user_id=str(current_user.id),
            document_ids=doc_id_strs,
            config=payload.config,
            max_tokens=payload.max_answer_tokens,
        )

        if not payload.include_sources:
            result.sources = None

        return result

    except Exception as exc:
        logger.exception(f"[query] Failed for user={current_user.email}: {exc}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Query failed: {exc}",
        )


# ─── Config reference ─────────────────────────────────────────────────────────

@router.get(
    "/config/defaults",
    summary="View default RAG configuration and all available options",
)
def get_default_config():
    """
    Returns the default RAGConfig plus human-readable descriptions of every
    option.  Useful for building a settings UI on the frontend.
    """
    return {
        "defaults": RAGConfig().model_dump(),
        "options": {
            "chunking_strategies": {
                "recursive": (
                    "Splits on paragraph → sentence → word boundaries.  "
                    "Fast, reliable, no extra compute.  Best default choice."
                ),
                "semantic": (
                    "Embeds every sentence, then splits where cosine distance "
                    "between adjacent sentences exceeds the threshold.  "
                    "Produces thematically coherent chunks.  ~2× slower than recursive "
                    "due to the extra embedding pass."
                ),
                "parent_child": (
                    "Creates large parent chunks (2 k chars) and smaller child chunks "
                    "(400 chars).  Children are embedded and retrieved; parents are "
                    "returned to the LLM for richer context.  Best precision + recall "
                    "trade-off on long, dense documents."
                ),
            },
            "embedding_models": {
                "BAAI/bge-m3": (
                    "Multilingual (100+ languages).  "
                    "Produces dense vectors + sparse lexical weights in one pass.  "
                    "Required for sparse and hybrid retrieval.  "
                    "~570 M parameters."
                ),
                "BAAI/bge-large-en-v1.5": (
                    "English-only.  Top-tier MTEB dense retrieval score.  "
                    "Faster inference than bge-m3 with lower memory.  "
                    "Only supports dense retrieval.  "
                    "~335 M parameters."
                ),
            },
            "retrieval_strategies": {
                "dense": (
                    "Approximate nearest-neighbour search on 1024-dim cosine space.  "
                    "Excellent for paraphrase and conceptual queries.  "
                    "Works with both embedding models."
                ),
                "sparse": (
                    "Lexical search using SPLADE-style token weights.  "
                    "Excellent for exact terms, acronyms, product codes, proper nouns.  "
                    "Requires BAAI/bge-m3."
                ),
                "hybrid": (
                    "Combines dense ANN + sparse search with Reciprocal Rank Fusion (k=60).  "
                    "Best overall recall across both conceptual and keyword queries.  "
                    "Requires BAAI/bge-m3."
                ),
            },
        },
    }


# ─── Delete vectors ───────────────────────────────────────────────────────────

@router.delete(
    "/document/{document_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Remove all vector embeddings for a document",
)
def delete_document_vectors(
    document_id:  UUID,
    current_user: User    = Depends(get_current_active_user),
    db:           Session = Depends(get_db),
):
    """
    Deletes all Milvus vectors for the document and resets its status to
    PENDING.  The S3 file and PostgreSQL record are kept intact.

    Use this to force a clean re-ingestion with different settings.
    """
    doc = _get_ready_document(document_id, current_user, db)

    vs.delete_document_chunks(str(document_id))

    doc.status      = DocumentStatus.PENDING
    doc.chunk_count = None
    db.commit()

    logger.info(
        f"[vectors] Deleted vectors for document={document_id} "
        f"user={current_user.email}"
    )
    return None