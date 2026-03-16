from fastapi import APIRouter, Depends, HTTPException, status, BackgroundTasks
from langchain_core.messages import HumanMessage, AIMessage
from sqlalchemy.orm import Session
from uuid import UUID
from typing import Optional

from src.database import get_db, SessionLocal
from src.models.document import Document, DocumentStatus
from src.models.user import User
from src.schemas.document import DocumentResponse
from src.schemas.preprocessing import QueryRequest, QueryResponse, SourceDocument, IngestResponse
from src.schemas.user import MessageResponse
from src.utils.auth import get_current_active_user
from src.utils.rag_pipeline import ingest_document, reingest_document
from src.utils.langchain_rag import (
    build_rag_chain_with_sources,
    delete_document_vectors,
)
from src.config import settings
import logging

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/rag", tags=["RAG"])


# ─── Ingest ───────────────────────────────────────────────────────────────────

@router.post(
    "/ingest/{document_id}",
    response_model=MessageResponse,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Ingest a document into the vector store (background)",
)
def trigger_ingest(
    document_id: UUID,
    background_tasks: BackgroundTasks,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db),
):
    """
    Full ingestion pipeline (background):
      1. Download from S3
      2. Parse with LangChain loaders (PDF/DOCX/TXT/CSV/XLSX)
      3. Split with RecursiveCharacterTextSplitter (tiktoken)
      4. Embed with embeddings model (HuggingFace)
      5. Store in pgvector with user_id + document_id metadata
    """
    document = _get_user_doc_or_404(document_id, current_user.id, db)

    if document.status == DocumentStatus.PROCESSING:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Document is already being processed.",
        )
    if document.status == DocumentStatus.READY:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"Already ingested ({document.chunk_count} chunks). "
                   "Use /rag/reingest/{id} to reprocess.",
        )

    background_tasks.add_task(_bg_ingest, str(document_id))

    return MessageResponse(
        message=f"Ingestion started for '{document.original_filename}'. "
                f"Poll GET /api/v1/rag/status/{document_id} for progress."
    )


@router.post(
    "/reingest/{document_id}",
    response_model=MessageResponse,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Delete existing vectors and re-ingest",
)
def trigger_reingest(
    document_id: UUID,
    background_tasks: BackgroundTasks,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db),
):
    document = _get_user_doc_or_404(document_id, current_user.id, db)

    if document.status == DocumentStatus.PROCESSING:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Document is currently being processed.",
        )

    background_tasks.add_task(_bg_reingest, str(document_id))
    return MessageResponse(message=f"Re-ingestion started for '{document.original_filename}'.")


# ─── Status ───────────────────────────────────────────────────────────────────

@router.get(
    "/status/{document_id}",
    response_model=DocumentResponse,
    summary="Poll ingestion status",
)
def get_status(
    document_id: UUID,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db),
):
    document = _get_user_doc_or_404(document_id, current_user.id, db)
    return DocumentResponse.model_validate(document)


# ─── Query / Chat ─────────────────────────────────────────────────────────────

@router.post(
    "/query",
    response_model=QueryResponse,
    summary="Ask a question over your ingested documents",
)
def query_documents(
    payload: QueryRequest,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db),
):
    """
    Full RAG query:
      - Multi-Query: rewrites question 3 ways for better recall
      - MMR retrieval: diverse results, filtered to this user's docs
      - Contextual Compression: strips irrelevant sentences from chunks
      - GPT-4o-mini: answers strictly from retrieved context
      - Returns answer + source citations (filename, page, chunk)
    """
    # Validate document_ids belong to this user (if provided)
    doc_id_strs: Optional[list[str]] = None
    if payload.document_ids:
        doc_id_strs = []
        for doc_id in payload.document_ids:
            doc = (
                db.query(Document)
                .filter(
                    Document.id == doc_id,
                    Document.user_id == current_user.id,
                    Document.status == DocumentStatus.READY,
                    Document.is_deleted == False,
                )
                .first()
            )
            if not doc:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail=f"Document {doc_id} not found or not yet ingested.",
                )
            doc_id_strs.append(str(doc_id))

    # Convert chat history to LangChain messages
    lc_history = []
    for msg in payload.chat_history:
        if msg.role == "human":
            lc_history.append(HumanMessage(content=msg.content))
        elif msg.role == "assistant":
            lc_history.append(AIMessage(content=msg.content))

    try:
        chain = build_rag_chain_with_sources(
            user_id=str(current_user.id),
            document_ids=doc_id_strs,
        )
        result = chain.invoke({
            "question": payload.question,
            "chat_history": lc_history,
        })
    except Exception as e:
        logger.error(f"RAG query failed for user {current_user.id}: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to process query. Please try again.",
        )

    sources = [SourceDocument(**s) for s in result.get("sources", [])]

    return QueryResponse(
        answer=result["answer"],
        sources=sources if payload.include_sources else [],
        question=payload.question,
        model_used=settings.OPENAI_CHAT_MODEL,
    )


# ─── Delete vectors ───────────────────────────────────────────────────────────

@router.delete(
    "/vectors/{document_id}",
    response_model=MessageResponse,
    summary="Delete all vectors for a document from pgvector",
)
def delete_vectors(
    document_id: UUID,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db),
):
    document = _get_user_doc_or_404(document_id, current_user.id, db)
    delete_document_vectors(str(document_id))
    document.status = DocumentStatus.PENDING
    document.chunk_count = None
    db.commit()
    return MessageResponse(
        message=f"Vectors deleted for '{document.original_filename}'. Re-ingest to search again."
    )


# ─── Background tasks ─────────────────────────────────────────────────────────

def _bg_ingest(document_id: str):
    db = SessionLocal()
    try:
        document = db.query(Document).filter(Document.id == document_id).first()
        if document:
            ingest_document(document, db)
    except Exception as e:
        logger.error(f"BG ingest error for {document_id}: {e}", exc_info=True)
    finally:
        db.close()


def _bg_reingest(document_id: str):
    db = SessionLocal()
    try:
        document = db.query(Document).filter(Document.id == document_id).first()
        if document:
            reingest_document(document, db)
    except Exception as e:
        logger.error(f"BG reingest error for {document_id}: {e}", exc_info=True)
    finally:
        db.close()


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _get_user_doc_or_404(document_id: UUID, user_id, db: Session) -> Document:
    doc = (
        db.query(Document)
        .filter(
            Document.id == document_id,
            Document.user_id == user_id,
            Document.is_deleted == False,
        )
        .first()
    )
    if not doc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Document not found.")
    return doc