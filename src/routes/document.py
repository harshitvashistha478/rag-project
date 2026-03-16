from fastapi import (
    APIRouter, Depends, HTTPException, UploadFile, File,
    Query, status
)
from sqlalchemy.orm import Session
from sqlalchemy import func
from datetime import datetime, timezone
from typing import Optional
from uuid import UUID
import math

from src.database import get_db
from src.models.document import Document, DocumentStatus
from src.models.user import User
from src.schemas.document import (
    DocumentResponse,
    DocumentWithURLResponse,
    DocumentListResponse,
    DocumentUploadResponse,
)
from src.utils.auth import get_current_active_user
from src.utils.s3_functions import (
    upload_file_to_s3,
    generate_presigned_url,
    delete_file_from_s3,
)
from src.config import settings
import logging

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/documents", tags=["Documents"])



@router.post(
    "/upload",
    response_model=DocumentUploadResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Upload a document to S3",
)
async def upload_document(
    file: UploadFile = File(..., description="File to upload"),
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db),
):
    """
    Uploads a file to S3 and persists a Document record in the database.

    - Validates file type and size before uploading.
    - Returns the document metadata + a pre-signed download URL.
    """
    s3_result = await upload_file_to_s3(file, str(current_user.id))

    document = Document(
        user_id=current_user.id,
        original_filename=file.filename or s3_result.stored_filename,
        stored_filename=s3_result.stored_filename,
        file_extension=s3_result.stored_filename.rsplit(".", 1)[-1],
        content_type=s3_result.content_type,
        file_size=s3_result.file_size,
        s3_bucket=s3_result.s3_bucket,
        s3_key=s3_result.s3_key,
        s3_region=s3_result.s3_region,
        status=DocumentStatus.PENDING,
    )
    db.add(document)
    db.commit()
    db.refresh(document)

    download_url = generate_presigned_url(document.s3_key)

    logger.info(
        f"User {current_user.email} uploaded document {document.id} "
        f"({document.original_filename}, {document.file_size_human})"
    )

    return DocumentUploadResponse(
        document=DocumentResponse.model_validate(document),
        download_url=download_url,
        url_expires_in=settings.S3_PRESIGNED_URL_EXPIRY,
    )



@router.get(
    "/",
    response_model=DocumentListResponse,
    summary="List all documents for the current user",
)
def list_documents(
    page: int = Query(1, ge=1, description="Page number"),
    page_size: int = Query(20, ge=1, le=100, description="Items per page"),
    status: Optional[DocumentStatus] = Query(None, description="Filter by status"),
    file_extension: Optional[str] = Query(None, description="Filter by extension, e.g. pdf"),
    search: Optional[str] = Query(None, description="Search by filename"),
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db),
):
    query = (
        db.query(Document)
        .filter(Document.user_id == current_user.id, Document.is_deleted == False)
    )

    if status:
        query = query.filter(Document.status == status)
    if file_extension:
        query = query.filter(Document.file_extension == file_extension.lower().strip("."))
    if search:
        query = query.filter(Document.original_filename.ilike(f"%{search}%"))

    total = query.count()
    total_pages = math.ceil(total / page_size) if total else 1
    items = (
        query.order_by(Document.created_at.desc())
        .offset((page - 1) * page_size)
        .limit(page_size)
        .all()
    )

    return DocumentListResponse(
        items=[DocumentResponse.model_validate(d) for d in items],
        total=total,
        page=page,
        page_size=page_size,
        total_pages=total_pages,
    )



@router.get(
    "/{document_id}",
    response_model=DocumentWithURLResponse,
    summary="Get a document by ID with a fresh download URL",
)
def get_document(
    document_id: UUID,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db),
):
    document = _get_user_document_or_404(document_id, current_user.id, db)
    download_url = generate_presigned_url(document.s3_key)

    return DocumentWithURLResponse(
        **DocumentResponse.model_validate(document).model_dump(),
        download_url=download_url,
        url_expires_in=settings.S3_PRESIGNED_URL_EXPIRY,
    )



@router.get(
    "/{document_id}/download-url",
    summary="Get a fresh pre-signed S3 download URL for a document",
)
def get_download_url(
    document_id: UUID,
    expiry: int = Query(
        default=None,
        ge=60,
        le=604800,     
        description="URL lifetime in seconds (60-604800). Defaults to server setting.",
    ),
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db),
):
    document = _get_user_document_or_404(document_id, current_user.id, db)
    expiry_seconds = expiry or settings.S3_PRESIGNED_URL_EXPIRY
    url = generate_presigned_url(document.s3_key, expiry=expiry_seconds)

    return {
        "document_id": str(document_id),
        "download_url": url,
        "expires_in": expiry_seconds,
        "original_filename": document.original_filename,
    }



@router.delete(
    "/{document_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Soft-delete a document (marks as deleted, removes from S3)",
)
def delete_document(
    document_id: UUID,
    hard_delete: bool = Query(
        False,
        description="If true, immediately deletes from S3. Default is soft delete only.",
    ),
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db),
):
    document = _get_user_document_or_404(document_id, current_user.id, db)

    delete_file_from_s3(document.s3_key)

    if hard_delete:
        db.delete(document)
    else:
        document.is_deleted = True
        document.deleted_at = datetime.now(timezone.utc)

    db.commit()
    logger.info(
        f"User {current_user.email} {'hard' if hard_delete else 'soft'}-deleted "
        f"document {document_id}"
    )
    return None



@router.get(
    "/admin/all",
    response_model=DocumentListResponse,
    summary="[Admin] List all documents across all users",
)
def admin_list_all_documents(
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100),
    status: Optional[DocumentStatus] = Query(None),
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db),
):
    from src.models.user import UserRole
    if current_user.role != UserRole.ADMIN:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Admin access required.",
        )

    query = db.query(Document).filter(Document.is_deleted == False)
    if status:
        query = query.filter(Document.status == status)

    total = query.count()
    items = (
        query.order_by(Document.created_at.desc())
        .offset((page - 1) * page_size)
        .limit(page_size)
        .all()
    )

    return DocumentListResponse(
        items=[DocumentResponse.model_validate(d) for d in items],
        total=total,
        page=page,
        page_size=page_size,
        total_pages=math.ceil(total / page_size) if total else 1,
    )



def _get_user_document_or_404(
    document_id: UUID,
    user_id: UUID,
    db: Session,
) -> Document:
    """Fetch a non-deleted document that belongs to the given user, or raise 404."""
    document = (
        db.query(Document)
        .filter(
            Document.id == document_id,
            Document.user_id == user_id,
            Document.is_deleted == False,
        )
        .first()
    )
    if not document:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Document not found.",
        )
    return document