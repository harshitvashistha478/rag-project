from pydantic_settings import BaseSettings
from pydantic import field_validator
from typing import List
import json


class Settings(BaseSettings):
    # Application
    APP_NAME: str = "AI Document Search"
    APP_VERSION: str = "1.0.0"
    DEBUG: bool = True

    # JWT
    SECRET_KEY: str
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 30
    REFRESH_TOKEN_EXPIRE_DAYS: int = 7

    # Database
    DATABASE_URL: str
    DB_POOL_SIZE: int = 10
    DB_MAX_OVERFLOW: int = 20

    # CORS
    ALLOWED_ORIGINS: List[str] = ["http://localhost:3000", "http://localhost:8000"]

    # AWS S3
    AWS_ACCESS_KEY_ID: str
    AWS_SECRET_ACCESS_KEY: str
    AWS_REGION: str = "ap-south-1"
    S3_BUCKET_NAME: str
    S3_PRESIGNED_URL_EXPIRY: int = 3600
    S3_MAX_FILE_SIZE_MB: int = 50
    S3_ALLOWED_EXTENSIONS: List[str] = [
        "pdf", "doc", "docx", "txt", "md",
        "png", "jpg", "jpeg", "csv", "xlsx",
    ]

    # Gemini
    GOOGLE_API_KEY: str
    GEMINI_CHAT_MODEL: str = "gemini-2.5-flash"
    GEMINI_TEMPERATURE: float = 0.1

    EMBEDDING_MODEL_NAME: str = "BAAI/bge-m3"
    EMBEDDING_DEVICE: str = "cpu"     

    # RAG tuning
    RAG_CHUNK_SIZE: int = 1000
    RAG_CHUNK_OVERLAP: int = 150
    RAG_RETRIEVAL_K: int = 6
    RAG_MULTIQUERY_COUNT: int = 3

    MILVUS_URI: str = "tcp://localhost:19530"     # Milvus default server URI
    MILVUS_ALIAS: str = "default"
    MILVUS_TOKEN: str = ""                  # only needed for Zilliz Cloud
    MILVUS_COLLECTION: str = "rag_documents"
    MILVUS_INDEX_TYPE: str = "HNSW"         # best recall for local Milvus
    MILVUS_METRIC_TYPE: str = "COSINE"      # bge-m3 uses cosine similarity

    @field_validator("ALLOWED_ORIGINS", mode="before")
    @classmethod
    def parse_allowed_origins(cls, v):
        if isinstance(v, str):
            return json.loads(v)
        return v

    @field_validator("S3_ALLOWED_EXTENSIONS", mode="before")
    @classmethod
    def parse_allowed_extensions(cls, v):
        if isinstance(v, str):
            return json.loads(v)
        return v

    @field_validator("SECRET_KEY")
    @classmethod
    def validate_secret_key(cls, v):
        if len(v) < 32:
            raise ValueError("SECRET_KEY must be at least 32 characters long")
        return v

    @property
    def S3_MAX_FILE_SIZE_BYTES(self) -> int:
        return self.S3_MAX_FILE_SIZE_MB * 1024 * 1024

    @property
    def PGVECTOR_CONNECTION_STRING(self) -> str:
        """psycopg3-style connection string required by langchain-postgres."""
        return self.DATABASE_URL.replace(
            "postgresql://", "postgresql+psycopg://"
        ).replace(
            "postgresql+psycopg2://", "postgresql+psycopg://"
        )

    model_config = {"env_file": ".env", "case_sensitive": True}


settings = Settings()