from sqlalchemy import (
    Column, String, BigInteger, Boolean,
    DateTime, Text, ForeignKey, Enum as SAEnum, Integer
)
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from src.database import Base
import uuid
import enum


class DocumentStatus(str, enum.Enum):
    PENDING    = "pending"
    PROCESSING = "processing"
    READY      = "ready"
    FAILED     = "failed"


class Document(Base):
    __tablename__ = "documents"

    id = Column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
        index=True,
    )

    # ── Ownership ──────────────────────────────────────────────────────────────
    user_id = Column(
        UUID(as_uuid=True),
        ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )

    # ── File metadata ──────────────────────────────────────────────────────────
    original_filename = Column(String(255), nullable=False)   # original name from upload
    stored_filename   = Column(String(512), nullable=False)   # UUID-based name in S3
    file_extension    = Column(String(20),  nullable=False)
    content_type      = Column(String(100), nullable=False)
    file_size         = Column(BigInteger,  nullable=False)   # bytes

    # ── S3 location ───────────────────────────────────────────────────────────
    s3_bucket     = Column(String(255), nullable=False)
    s3_key        = Column(String(1024), nullable=False, unique=True)  # full S3 object key
    s3_region     = Column(String(50),  nullable=False)

    # ── Processing state ──────────────────────────────────────────────────────
    status = Column(
        SAEnum(
            DocumentStatus,
            name="documentstatus",
            values_callable=lambda x: [e.value for e in x],  # store lowercase values
        ),
        default=DocumentStatus.PENDING,
        nullable=False,
        index=True,
    )
    processing_error  = Column(Text, nullable=True)           # error message if FAILED
    chunk_count       = Column(Integer, nullable=True)        # filled after processing

    # ── Soft delete ───────────────────────────────────────────────────────────
    is_deleted    = Column(Boolean, default=False, nullable=False, index=True)
    deleted_at    = Column(DateTime(timezone=True), nullable=True)

    # ── Timestamps ────────────────────────────────────────────────────────────
    created_at = Column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    updated_at = Column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False,
    )

    # ── Relationships ─────────────────────────────────────────────────────────
    owner = relationship("User", back_populates="documents", lazy="select")

    # ── Helpers ───────────────────────────────────────────────────────────────
    @property
    def file_size_human(self) -> str:
        """Returns human-readable file size (e.g. '2.4 MB')."""
        size = self.file_size
        for unit in ("B", "KB", "MB", "GB"):
            if size < 1024:
                return f"{size:.1f} {unit}"
            size /= 1024
        return f"{size:.1f} TB"

    def __repr__(self):
        return f"<Document id={self.id} filename={self.original_filename} status={self.status}>"