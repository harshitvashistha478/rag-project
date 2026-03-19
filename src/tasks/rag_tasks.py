"""
RAG background tasks.

ingest_document_task
────────────────────
Runs the full RAG ingestion pipeline in a Celery worker process:
  S3 download → PDF parse → chunk → embed → Milvus insert

The task updates the Document.status column in PostgreSQL at each stage
so the FastAPI route can poll progress via GET /rag/tasks/{task_id}.

Task lifecycle
──────────────
  PENDING   → task queued, not yet picked up
  STARTED   → worker picked it up (we update doc status → processing)
  SUCCESS   → ingestion complete (doc status → ready, chunk_count set)
  FAILURE   → unhandled exception (doc status → failed, error stored)
"""

import logging
from celery import Task
from celery.utils.log import get_task_logger

from src.celery_app import celery_app
from src.database import SessionLocal
from src.models.document import Document, DocumentStatus
from src.rag.pipeline import ingest_document
from src.schemas.rag import RAGConfig

logger = get_task_logger(__name__)


class BaseTaskWithRetry(Task):
    """
    Base class that adds structured error handling and DB status updates.
    All RAG tasks should inherit from this.
    """
    abstract = True

    def on_failure(self, exc, task_id, args, kwargs, einfo):
        """Called by Celery when the task raises an unhandled exception."""
        document_id = kwargs.get("document_id") or (args[0] if args else None)
        if document_id:
            _mark_document_failed(document_id, str(exc)[:500])
        logger.error(f"Task {task_id} failed for document={document_id}: {exc}")
        super().on_failure(exc, task_id, args, kwargs, einfo)


@celery_app.task(
    bind=True,
    base=BaseTaskWithRetry,
    name="src.tasks.rag_tasks.ingest_document_task",
    max_retries=2,
    default_retry_delay=30,    # seconds between retries
    soft_time_limit=600,       # 10 min — raises SoftTimeLimitExceeded
    time_limit=660,            # 11 min — hard kill
    track_started=True,        # allows STARTED state to be tracked
)
def ingest_document_task(
    self,
    document_id:       str,
    user_id:           str,
    s3_key:            str,
    original_filename: str,
    config_dict:       dict,
) -> dict:
    """
    Background task: ingest a PDF document into the RAG vector store.

    Args
    ----
    document_id:       UUID string of the Document record in PostgreSQL.
    user_id:           UUID string of the owning user.
    s3_key:            Full S3 object key for the PDF file.
    original_filename: Display name (for logging / error messages).
    config_dict:       RAGConfig serialised as a plain dict (JSON-safe).

    Returns
    -------
    dict with keys: document_id, chunks_created, embedding_model,
                    chunking_strategy, status, message
    """
    logger.info(
        f"[task:{self.request.id}] Starting ingestion for "
        f"document={document_id} user={user_id}"
    )

    # ── Mark document as PROCESSING in DB ────────────────────────
    _set_document_status(document_id, DocumentStatus.PROCESSING)

    # ── Deserialise RAGConfig ─────────────────────────────────────
    try:
        config = RAGConfig(**config_dict)
    except Exception as exc:
        raise ValueError(f"Invalid RAGConfig: {exc}") from exc

    # ── Run the pipeline ──────────────────────────────────────────
    try:
        chunk_count = ingest_document(
            document_id=document_id,
            user_id=user_id,
            s3_key=s3_key,
            original_filename=original_filename,
            config=config,
        )
    except Exception as exc:
        # Retry transient errors (S3 timeouts, Milvus blips).
        # Permanent errors (bad PDF, corrupt file) will exhaust retries
        # and trigger on_failure → mark document FAILED.
        logger.warning(
            f"[task:{self.request.id}] Ingestion error (attempt "
            f"{self.request.retries + 1}/{self.max_retries + 1}): {exc}"
        )
        try:
            raise self.retry(exc=exc)
        except self.MaxRetriesExceededError:
            _mark_document_failed(document_id, str(exc)[:500])
            raise

    # ── Mark document as READY in DB ─────────────────────────────
    _set_document_ready(document_id, chunk_count)

    result = {
        "document_id":       document_id,
        "chunks_created":    chunk_count,
        "embedding_model":   config.embedding_model,
        "chunking_strategy": config.chunking_strategy,
        "status":            "ready",
        "message": (
            f"Successfully ingested '{original_filename}' — "
            f"{chunk_count} chunks created."
        ),
    }
    logger.info(
        f"[task:{self.request.id}] Completed: "
        f"document={document_id} chunks={chunk_count}"
    )
    return result


# ─── DB helpers (each opens its own short-lived session) ──────────────────────

def _set_document_status(document_id: str, status: DocumentStatus) -> None:
    db = SessionLocal()
    try:
        doc = db.query(Document).filter(Document.id == document_id).first()
        if doc:
            doc.status = status
            db.commit()
    except Exception as exc:
        logger.error(f"Failed to update document status: {exc}")
        db.rollback()
    finally:
        db.close()


def _set_document_ready(document_id: str, chunk_count: int) -> None:
    db = SessionLocal()
    try:
        doc = db.query(Document).filter(Document.id == document_id).first()
        if doc:
            doc.status           = DocumentStatus.READY
            doc.chunk_count      = chunk_count
            doc.processing_error = None
            db.commit()
    except Exception as exc:
        logger.error(f"Failed to mark document ready: {exc}")
        db.rollback()
    finally:
        db.close()


def _mark_document_failed(document_id: str, error_message: str) -> None:
    db = SessionLocal()
    try:
        doc = db.query(Document).filter(Document.id == document_id).first()
        if doc:
            doc.status           = DocumentStatus.FAILED
            doc.processing_error = error_message
            db.commit()
    except Exception as exc:
        logger.error(f"Failed to mark document as failed: {exc}")
        db.rollback()
    finally:
        db.close()