"""
RAG Pipeline Entry Point
========================
Called from background tasks to process a Document:
  1. Load from S3 via LangChain loaders
  2. Split with tiktoken-aware RecursiveCharacterTextSplitter
  3. Embed with OpenAI + store in pgvector
  4. Update Document status → READY
"""

import logging
from sqlalchemy.orm import Session

from src.models.document import Document, DocumentStatus
from src.utils.langchain_rag import (
    load_document_from_s3,
    split_documents,
    embed_and_store,
    delete_document_vectors,
)

logger = logging.getLogger(__name__)


def ingest_document(document: Document, db: Session) -> int:
    """
    Full ingestion pipeline for a single Document.
    Marks status PROCESSING → READY (or FAILED on error).

    Returns:
        Number of chunks embedded and stored.
    """
    # ── Mark processing ───────────────────────────────────────────────────────
    document.status = DocumentStatus.PROCESSING
    document.processing_error = None
    db.commit()

    try:
        # ── 1. Load from S3 ───────────────────────────────────────────────────
        lc_docs = load_document_from_s3(
            s3_key=document.s3_key,
            file_extension=document.file_extension,
            original_filename=document.original_filename,
        )

        # ── 2. Split ──────────────────────────────────────────────────────────
        chunks = split_documents(
            docs=lc_docs,
            document_id=str(document.id),
            user_id=str(document.user_id),
            original_filename=document.original_filename,
            s3_key=document.s3_key,
            file_extension=document.file_extension,
        )

        # ── 3. Embed + store in pgvector ──────────────────────────────────────
        chunk_count = embed_and_store(chunks, document_id=str(document.id))

        # ── 4. Mark ready ─────────────────────────────────────────────────────
        document.status = DocumentStatus.READY
        document.chunk_count = chunk_count
        db.commit()

        logger.info(
            f"Ingestion complete: document {document.id} "
            f"({document.original_filename}) → {chunk_count} chunks"
        )
        return chunk_count

    except Exception as e:
        document.status = DocumentStatus.FAILED
        document.processing_error = str(e)[:1000]
        db.commit()
        logger.error(f"Ingestion failed for document {document.id}: {e}", exc_info=True)
        raise


def reingest_document(document: Document, db: Session) -> int:
    """
    Delete existing vectors for a document and re-ingest from scratch.
    Useful when re-uploading or changing chunk settings.
    """
    logger.info(f"Re-ingesting document {document.id} — deleting old vectors first")
    delete_document_vectors(str(document.id))
    return ingest_document(document, db)