"""
Celery application — configured with Redis as both broker and result backend.

Import this module wherever you need to dispatch tasks or read results.
Never import tasks directly here to avoid circular imports — tasks are
autodiscovered from src.tasks.* at worker startup.
"""

from celery import Celery
from src.config import settings

celery_app = Celery(
    "docmind",
    broker=settings.REDIS_URL,
    backend=settings.REDIS_URL,
    include=["src.tasks.rag_tasks"],
)

celery_app.conf.update(
    # ── Serialisation ────────────────────────────────────────────────────────
    task_serializer="json",
    result_serializer="json",
    accept_content=["json"],

    # ── Reliability ──────────────────────────────────────────────────────────
    # Acknowledge the task only after it has finished (not just when received).
    # This prevents task loss if a worker crashes mid-execution.
    task_acks_late=True,

    # If a worker is killed mid-task, requeue it once for another worker.
    task_reject_on_worker_lost=True,

    # ── Result expiry ────────────────────────────────────────────────────────
    # Keep task results in Redis for 24 hours — enough for the UI to poll.
    result_expires=86_400,

    # ── Concurrency ──────────────────────────────────────────────────────────
    # Each worker process handles one task at a time (RAG ingestion is CPU/IO
    # heavy — prefetching more would starve the embedding model).
    worker_prefetch_multiplier=1,

    # ── Queues ───────────────────────────────────────────────────────────────
    # Separate queues let you scale ingest workers independently from other tasks.
    task_default_queue="default",
    task_queues={
        "ingest":  {"exchange": "ingest",  "routing_key": "ingest"},
        "default": {"exchange": "default", "routing_key": "default"},
    },
    task_routes={
        "src.tasks.rag_tasks.ingest_document_task": {"queue": "ingest"},
    },

    # ── Timezone ─────────────────────────────────────────────────────────────
    timezone="UTC",
    enable_utc=True,
)