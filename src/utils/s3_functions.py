import boto3
import uuid
import logging
from botocore.exceptions import ClientError, BotoCoreError
from botocore.config import Config
from fastapi import HTTPException, status, UploadFile
from typing import Optional

from src.config import settings
from src.schemas.document import S3UploadResult

logger = logging.getLogger(__name__)

# ─── MIME type → extension map ───────────────────────────────────────────────
MIME_TO_EXT: dict[str, str] = {
    "application/pdf":                                                    "pdf",
    "application/msword":                                                 "doc",
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document": "docx",
    "text/plain":                                                         "txt",
    "text/markdown":                                                      "md",
    "text/csv":                                                           "csv",
    "application/vnd.ms-excel":                                           "xls",
    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet":  "xlsx",
    "image/png":                                                          "png",
    "image/jpeg":                                                         "jpg",
    "image/jpg":                                                          "jpg",
}


def _get_s3_client():
    """Create and return a boto3 S3 client using settings credentials."""
    return boto3.client(
        "s3",
        region_name=settings.AWS_REGION,
        aws_access_key_id=settings.AWS_ACCESS_KEY_ID,
        aws_secret_access_key=settings.AWS_SECRET_ACCESS_KEY,
        endpoint_url=f"https://s3.{settings.AWS_REGION}.amazonaws.com",  # ← force regional endpoint
        config=Config(
            retries={"max_attempts": 3, "mode": "standard"},
            connect_timeout=10,
            read_timeout=30,
            signature_version="s3v4",          # ← required for regional buckets
            s3={"addressing_style": "virtual"}, # ← bucket-name.s3.region.amazonaws.com style
        ),
    )


# ─── Validation helpers ───────────────────────────────────────────────────────

def validate_file(file: UploadFile, content: bytes) -> str:
    """
    Validates the uploaded file against size and extension rules.
    Returns the resolved file extension.
    Raises HTTP 400/413 on failure.
    """
    # Size check
    if len(content) > settings.S3_MAX_FILE_SIZE_BYTES:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"File exceeds the maximum allowed size of {settings.S3_MAX_FILE_SIZE_MB} MB.",
        )

    if len(content) == 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Uploaded file is empty.",
        )

    # Resolve extension from filename first, then content-type
    original_name = file.filename or ""
    ext = original_name.rsplit(".", 1)[-1].lower() if "." in original_name else ""

    if not ext and file.content_type:
        ext = MIME_TO_EXT.get(file.content_type, "")

    if not ext or ext not in settings.S3_ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail=(
                f"File type '.{ext}' is not allowed. "
                f"Allowed types: {', '.join(settings.S3_ALLOWED_EXTENSIONS)}"
            ),
        )

    return ext


def _build_s3_key(user_id: str, stored_filename: str) -> str:
    """
    Constructs the S3 object key.
    Pattern: documents/{user_id}/{stored_filename}
    Keeps each user's files in their own prefix for easy IAM scoping.
    """
    return f"documents/{user_id}/{stored_filename}"


def _resolve_content_type(file: UploadFile, ext: str) -> str:
    """Return a clean content-type string, falling back to octet-stream."""
    if file.content_type and file.content_type != "application/octet-stream":
        return file.content_type
    reverse = {v: k for k, v in MIME_TO_EXT.items()}
    return reverse.get(ext, "application/octet-stream")


# ─── Core S3 operations ───────────────────────────────────────────────────────

async def upload_file_to_s3(
    file: UploadFile,
    user_id: str,
) -> S3UploadResult:
    """
    Reads, validates, and uploads a file to S3.

    Returns S3UploadResult with all metadata needed to persist a Document row.
    """
    content = await file.read()
    ext = validate_file(file, content)

    stored_filename = f"{uuid.uuid4()}.{ext}"
    s3_key = _build_s3_key(user_id, stored_filename)
    content_type = _resolve_content_type(file, ext)

    s3 = _get_s3_client()
    try:
        s3.put_object(
            Bucket=settings.S3_BUCKET_NAME,
            Key=s3_key,
            Body=content,
            ContentType=content_type,
            Metadata={
                "user_id":           user_id,
                "original_filename": file.filename or stored_filename,
            },
            # Server-side encryption at rest
            ServerSideEncryption="AES256",
        )
        logger.info(f"Uploaded s3://{settings.S3_BUCKET_NAME}/{s3_key} ({len(content)} bytes)")
    except ClientError as e:
        error_code = e.response["Error"]["Code"]
        logger.error(f"S3 ClientError [{error_code}] uploading {s3_key}: {e}")
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"Failed to upload file to storage: {error_code}",
        )
    except BotoCoreError as e:
        logger.error(f"BotoCoreError uploading {s3_key}: {e}")
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="Storage service is unavailable. Please try again.",
        )

    return S3UploadResult(
        s3_key=s3_key,
        s3_bucket=settings.S3_BUCKET_NAME,
        s3_region=settings.AWS_REGION,
        stored_filename=stored_filename,
        content_type=content_type,
        file_size=len(content),
    )


def generate_presigned_url(
    s3_key: str,
    expiry: Optional[int] = None,
) -> str:
    """
    Generate a pre-signed GET URL for a private S3 object.

    Args:
        s3_key:  Full S3 object key (e.g. documents/user-id/file.pdf)
        expiry:  URL lifetime in seconds. Defaults to settings.S3_PRESIGNED_URL_EXPIRY.

    Returns:
        Pre-signed HTTPS URL string.
    """
    expiry = expiry or settings.S3_PRESIGNED_URL_EXPIRY
    s3 = _get_s3_client()
    try:
        url = s3.generate_presigned_url(
            "get_object",
            Params={"Bucket": settings.S3_BUCKET_NAME, "Key": s3_key},
            ExpiresIn=expiry,
        )
        return url
    except ClientError as e:
        logger.error(f"Failed to generate presigned URL for {s3_key}: {e}")
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="Could not generate download URL.",
        )


def delete_file_from_s3(s3_key: str) -> bool:
    """
    Permanently delete an object from S3.

    Returns True on success, raises HTTPException on hard failure.
    Note: S3 delete_object does NOT error if the key doesn't exist.
    """
    s3 = _get_s3_client()
    try:
        s3.delete_object(Bucket=settings.S3_BUCKET_NAME, Key=s3_key)
        logger.info(f"Deleted s3://{settings.S3_BUCKET_NAME}/{s3_key}")
        return True
    except ClientError as e:
        logger.error(f"S3 ClientError deleting {s3_key}: {e}")
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="Failed to delete file from storage.",
        )


def check_s3_object_exists(s3_key: str) -> bool:
    """Returns True if the S3 object exists, False otherwise."""
    s3 = _get_s3_client()
    try:
        s3.head_object(Bucket=settings.S3_BUCKET_NAME, Key=s3_key)
        return True
    except ClientError as e:
        if e.response["Error"]["Code"] in ("404", "NoSuchKey"):
            return False
        logger.error(f"S3 head_object error for {s3_key}: {e}")
        return False


def download_file_from_s3(s3_key: str) -> bytes:
    """
    Download a file from S3 and return its raw bytes.
    Used internally by the RAG pipeline — never exposes a URL.
 
    Args:
        s3_key: Full S3 object key (e.g. documents/user-id/uuid.pdf)
 
    Returns:
        Raw file bytes ready for parsing.
    """
    s3 = _get_s3_client()
    try:
        response = s3.get_object(Bucket=settings.S3_BUCKET_NAME, Key=s3_key)
        content = response["Body"].read()
        logger.info(f"Downloaded s3://{settings.S3_BUCKET_NAME}/{s3_key} ({len(content)} bytes)")
        return content
    except ClientError as e:
        error_code = e.response["Error"]["Code"]
        if error_code in ("NoSuchKey", "404"):
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"File not found in storage: {s3_key}",
            )
        logger.error(f"S3 ClientError [{error_code}] downloading {s3_key}: {e}")
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"Failed to download file from storage: {error_code}",
        )
    except BotoCoreError as e:
        logger.error(f"BotoCoreError downloading {s3_key}: {e}")
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="Storage service is unavailable. Please try again.",
        )