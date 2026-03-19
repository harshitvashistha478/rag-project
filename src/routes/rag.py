"""
RAG API router — Celery-backed async ingestion + chat session management.

Ingest flow (async)
───────────────────
  POST /rag/ingest/{document_id}
    → validates document
    → dispatches ingest_document_task to Celery
    → returns {task_id, document_id, status: "queued"} immediately

  GET  /rag/tasks/{task_id}
    → polls Celery result backend (Redis) for task state
    → returns {state, progress, result, error}

All other endpoints (query, sessions, config, vectors) are unchanged.
"""

import json
import logging
from uuid import UUID
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy.orm import Session
from celery.result import AsyncResult

from src.database import get_db
from src.models.document import Document, DocumentStatus
from src.models.user import User
from src.models.chat import ChatSession, ChatMessage
from src.schemas.rag import (
    RAGConfig,
    IngestRequest,
    IngestResponse,
    TaskStatusResponse,
    QueryRequest,
    QueryResponse,
)
from src.schemas.chat import (
    ChatSessionResponse,
    ChatSessionDetailResponse,
    ChatSessionListResponse,
    ChatMessageResponse,
    RenameChatSessionRequest,
)
from src.utils.auth import get_current_active_user
from src.tasks.rag_tasks import ingest_document_task
from src.rag.pipeline import query_documents
from src.rag import vector_store as vs
from src.celery_app import celery_app

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/rag", tags=["RAG"])

N_HISTORY_TURNS = 6


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _get_ready_document(document_id: UUID, user: User, db: Session) -> Document:
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


def _get_session_or_404(session_id: UUID, user_id: UUID, db: Session) -> ChatSession:
    sess = db.query(ChatSession).filter(
        ChatSession.id == session_id,
        ChatSession.user_id == user_id,
    ).first()
    if not sess:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Chat session not found.")
    return sess


def _session_to_response(sess: ChatSession, db: Session) -> ChatSessionResponse:
    return ChatSessionResponse(
        id=sess.id,
        user_id=sess.user_id,
        title=sess.title,
        created_at=sess.created_at,
        updated_at=sess.updated_at,
        message_count=sess.messages.count(),
    )


# ─── Ingest (async) ───────────────────────────────────────────────────────────

@router.post(
    "/ingest/{document_id}",
    response_model=TaskStatusResponse,
    status_code=status.HTTP_202_ACCEPTED,   # 202 = accepted for processing
    summary="Queue a PDF document for background RAG ingestion",
)
def ingest(
    document_id:  UUID,
    body:         IngestRequest         = IngestRequest(),
    current_user: User                  = Depends(get_current_active_user),
    db:           Session               = Depends(get_db),
):
    """
    Dispatches a background Celery task to ingest the document.

    Returns immediately with a ``task_id``.
    Poll ``GET /rag/tasks/{task_id}`` to track progress.

    The document ``status`` field in the Documents API will also reflect
    the current state (PENDING → PROCESSING → READY | FAILED).
    """
    doc = _get_ready_document(document_id, current_user, db)

    if doc.file_extension.lower() != "pdf":
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"RAG ingestion only supports PDF files. Extension: '.{doc.file_extension}'.",
        )

    if doc.status == DocumentStatus.PROCESSING:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Document is already being processed.",
        )

    # Mark as PENDING so the UI can show the queued state immediately
    doc.status = DocumentStatus.PENDING
    db.commit()

    # Dispatch to Celery — all args must be JSON-serialisable
    task = ingest_document_task.apply_async(
        kwargs={
            "document_id":       str(doc.id),
            "user_id":           str(current_user.id),
            "s3_key":            doc.s3_key,
            "original_filename": doc.original_filename,
            "config_dict":       body.config.model_dump(),
        },
        queue="ingest",
    )

    logger.info(
        f"[ingest] Queued task={task.id} "
        f"document={document_id} user={current_user.email}"
    )

    return TaskStatusResponse(
        task_id=task.id,
        document_id=str(document_id),
        state="PENDING",
        message="Document queued for ingestion. Poll /rag/tasks/{task_id} for progress.",
    )


# ─── Task status polling ──────────────────────────────────────────────────────

@router.get(
    "/tasks/{task_id}",
    response_model=TaskStatusResponse,
    summary="Poll the status of a background ingestion task",
)
def get_task_status(
    task_id:      str,
    current_user: User = Depends(get_current_active_user),
):
    """
    Returns the current state of a Celery task.

    States
    ------
    PENDING  — queued, not yet started
    STARTED  — worker picked it up, currently running
    SUCCESS  — completed successfully
    FAILURE  — failed after all retries
    RETRY    — transient error, waiting to retry
    REVOKED  — cancelled

    The ``result`` field is populated on SUCCESS.
    The ``error`` field is populated on FAILURE.
    """
    result = AsyncResult(task_id, app=celery_app)

    response = TaskStatusResponse(
        task_id=task_id,
        document_id=None,
        state=result.state,
        message=_state_message(result.state),
    )

    if result.state == "SUCCESS":
        data              = result.result or {}
        response.document_id = data.get("document_id")
        response.result   = data
        response.message  = data.get("message", "Ingestion complete.")

    elif result.state == "FAILURE":
        response.error   = str(result.result)   # result holds the exception on failure
        response.message = "Ingestion failed."

    elif result.state == "STARTED":
        # task.info can carry custom progress metadata if we call
        # self.update_state(state="STARTED", meta={...}) from inside the task
        response.meta    = result.info if isinstance(result.info, dict) else {}

    return response


def _state_message(state: str) -> str:
    return {
        "PENDING": "Queued — waiting for an available worker.",
        "STARTED": "In progress — downloading, parsing, and embedding…",
        "SUCCESS": "Ingestion complete.",
        "FAILURE": "Ingestion failed.",
        "RETRY":   "Transient error — retrying shortly.",
        "REVOKED": "Task was cancelled.",
    }.get(state, state)


# ─── Query ────────────────────────────────────────────────────────────────────

@router.post(
    "/query",
    response_model=QueryResponse,
    summary="Ask a question — creates or continues a chat session",
)
def query(
    payload:      QueryRequest,
    current_user: User    = Depends(get_current_active_user),
    db:           Session = Depends(get_db),
):
    if not payload.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    doc_id_strs = None
    if payload.document_ids:
        doc_id_strs = []
        for doc_uuid in payload.document_ids:
            doc = _get_ready_document(doc_uuid, current_user, db)
            if doc.status != DocumentStatus.READY:
                raise HTTPException(
                    status_code=400,
                    detail=f"Document '{doc.original_filename}' is not ready (status: {doc.status.value}).",
                )
            doc_id_strs.append(str(doc.id))

    # Resolve or create session
    if payload.session_id:
        session = _get_session_or_404(payload.session_id, current_user.id, db)
    else:
        title = payload.question.strip()[:80]
        if len(payload.question.strip()) > 80:
            title += "…"
        session = ChatSession(
            user_id=current_user.id,
            title=title,
            rag_config_json=payload.config.model_dump_json(),
        )
        db.add(session)
        db.flush()

    past_messages = session.messages.order_by(ChatMessage.turn_index.asc()).all()
    history       = [
        {"role": m.role, "content": m.content}
        for m in past_messages[-(N_HISTORY_TURNS * 2):]
    ]
    turn_index = len(past_messages)

    user_msg = ChatMessage(
        session_id=session.id,
        role="user",
        content=payload.question,
        turn_index=turn_index,
    )
    db.add(user_msg)

    try:
        result = query_documents(
            question=payload.question,
            user_id=str(current_user.id),
            document_ids=doc_id_strs,
            config=payload.config,
            max_tokens=payload.max_answer_tokens,
            history=history,
        )
    except Exception as exc:
        db.rollback()
        logger.exception(f"[query] Failed for user={current_user.email}: {exc}")
        raise HTTPException(status_code=500, detail=f"Query failed: {exc}")

    sources_json = None
    if result.sources:
        sources_json = json.dumps([s.model_dump() for s in result.sources])

    assistant_msg = ChatMessage(
        session_id=session.id,
        role="assistant",
        content=result.answer,
        sources_json=sources_json,
        metadata_json=json.dumps({
            "model_used":         result.model_used,
            "retrieval_strategy": result.retrieval_strategy,
            "chunks_retrieved":   result.chunks_retrieved,
        }),
        turn_index=turn_index + 1,
    )
    db.add(assistant_msg)
    db.commit()

    if not payload.include_sources:
        result.sources = None

    result.session_id = str(session.id)
    return result


# ─── Config reference ─────────────────────────────────────────────────────────

@router.get("/config/defaults")
def get_default_config():
    return {
        "defaults": RAGConfig().model_dump(),
        "options": {
            "chunking_strategies": {
                "recursive":    "Splits on paragraph → sentence → word. Fast, reliable.",
                "semantic":     "Embedding-guided splits at topic boundaries.",
                "parent_child": "Small child chunks retrieved, large parent chunks sent to LLM.",
            },
            "embedding_models": {
                "BAAI/bge-m3":            "Multilingual. Dense + sparse. Required for hybrid.",
                "BAAI/bge-large-en-v1.5": "English-only. Fastest dense-only baseline.",
            },
            "retrieval_strategies": {
                "dense":  "ANN vector search. Best for conceptual queries.",
                "sparse": "Lexical search. Best for exact terms. Requires bge-m3.",
                "hybrid": "Dense + sparse fused with RRF. Best overall. Requires bge-m3.",
            },
        },
    }


# ─── Delete vectors ───────────────────────────────────────────────────────────

@router.delete("/document/{document_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_document_vectors(
    document_id:  UUID,
    current_user: User    = Depends(get_current_active_user),
    db:           Session = Depends(get_db),
):
    doc = _get_ready_document(document_id, current_user, db)
    vs.delete_document_chunks(str(document_id))
    doc.status      = DocumentStatus.PENDING
    doc.chunk_count = None
    db.commit()
    return None


# ─── Chat Session endpoints ───────────────────────────────────────────────────

@router.get("/sessions", response_model=ChatSessionListResponse)
def list_sessions(
    page:         int     = Query(1, ge=1),
    page_size:    int     = Query(20, ge=1, le=100),
    current_user: User    = Depends(get_current_active_user),
    db:           Session = Depends(get_db),
):
    q       = db.query(ChatSession).filter(ChatSession.user_id == current_user.id).order_by(ChatSession.updated_at.desc())
    total   = q.count()
    sessions = q.offset((page - 1) * page_size).limit(page_size).all()
    return ChatSessionListResponse(items=[_session_to_response(s, db) for s in sessions], total=total)


@router.get("/sessions/{session_id}", response_model=ChatSessionDetailResponse)
def get_session(
    session_id:   UUID,
    current_user: User    = Depends(get_current_active_user),
    db:           Session = Depends(get_db),
):
    session  = _get_session_or_404(session_id, current_user.id, db)
    messages = session.messages.order_by(ChatMessage.turn_index.asc()).all()
    return ChatSessionDetailResponse(
        id=session.id, user_id=session.user_id, title=session.title,
        created_at=session.created_at, updated_at=session.updated_at,
        message_count=len(messages),
        messages=[ChatMessageResponse.from_orm_with_json(m) for m in messages],
    )


@router.patch("/sessions/{session_id}", response_model=ChatSessionResponse)
def rename_session(
    session_id:   UUID,
    payload:      RenameChatSessionRequest,
    current_user: User    = Depends(get_current_active_user),
    db:           Session = Depends(get_db),
):
    session       = _get_session_or_404(session_id, current_user.id, db)
    session.title = payload.title[:100]
    db.commit()
    db.refresh(session)
    return _session_to_response(session, db)


@router.delete("/sessions/{session_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_session(
    session_id:   UUID,
    current_user: User    = Depends(get_current_active_user),
    db:           Session = Depends(get_db),
):
    session = _get_session_or_404(session_id, current_user.id, db)
    db.delete(session)
    db.commit()
    return None