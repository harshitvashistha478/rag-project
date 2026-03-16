from pydantic import BaseModel, ConfigDict, field_validator
from typing import Optional, List
from datetime import datetime
from uuid import UUID
from src.models.document import DocumentStatus


# ─── Response Schemas ────────────────────────────────────────────────────────

class DocumentResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: UUID
    user_id: UUID
    original_filename: str
    file_extension: str
    content_type: str
    file_size: int
    file_size_human: str              # "2.4 MB" — from model @property
    s3_bucket: str
    s3_key: str
    s3_region: str
    status: DocumentStatus
    processing_error: Optional[str]
    chunk_count: Optional[int]
    is_deleted: bool
    created_at: datetime
    updated_at: datetime


class DocumentWithURLResponse(DocumentResponse):
    """DocumentResponse + a pre-signed S3 download URL."""
    download_url: str
    url_expires_in: int               # seconds


class DocumentListResponse(BaseModel):
    items: List[DocumentResponse]
    total: int
    page: int
    page_size: int
    total_pages: int


class DocumentUploadResponse(BaseModel):
    """Returned immediately after a successful upload."""
    document: DocumentResponse
    download_url: str
    url_expires_in: int
    message: str = "File uploaded successfully."


# ─── Query / Filter Schemas ──────────────────────────────────────────────────

class DocumentFilterParams(BaseModel):
    """Query parameters for listing documents."""
    page: int = 1
    page_size: int = 20
    status: Optional[DocumentStatus] = None
    file_extension: Optional[str] = None
    search: Optional[str] = None      # searches original_filename

    @field_validator("page")
    @classmethod
    def page_must_be_positive(cls, v: int) -> int:
        if v < 1:
            raise ValueError("page must be >= 1")
        return v

    @field_validator("page_size")
    @classmethod
    def page_size_range(cls, v: int) -> int:
        if not (1 <= v <= 100):
            raise ValueError("page_size must be between 1 and 100")
        return v


# ─── Internal / Utility Schemas ──────────────────────────────────────────────

class S3UploadResult(BaseModel):
    """Internal schema — result of a successful S3 put_object call."""
    s3_key: str
    s3_bucket: str
    s3_region: str
    stored_filename: str
    content_type: str
    file_size: int