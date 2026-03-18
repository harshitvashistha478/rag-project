"""
Milvus vector-store layer.

Manages one collection that supports:
  • Dense ANN search (HNSW + COSINE) — always available
  • Sparse IP search (SPARSE_INVERTED_INDEX) — available when bge-m3 is used
  • Hybrid search (dense + sparse fused via RRF) — available when both are indexed

Collection schema
─────────────────
  id             VARCHAR(64)   PK  — chunk UUID
  document_id    VARCHAR(64)       — foreign key to PostgreSQL documents.id
  user_id        VARCHAR(64)       — owner (enforced in every query filter)
  content        VARCHAR(65535)    — raw chunk text (capped)
  dense_vector   FLOAT_VECTOR(1024)
  sparse_vector  SPARSE_FLOAT_VECTOR        — optional; present when bge-m3 used
  page_number    INT32
  chunk_type     VARCHAR(32)       — "text" | "table" | "parent"
  parent_id      VARCHAR(64)       — set for parent-child children
  is_parent      BOOL              — True for parent chunks (not retrieved directly)
  metadata_json  VARCHAR(1024)     — JSON blob for extra fields

Design decisions
────────────────
  • user_id is filtered in EVERY query so one user can't see another's data.
  • Parent chunks get a zero dense vector — they are stored for context fetch
    but excluded from retrieval by the ``is_parent == false`` filter.
  • The collection is created with sparse support unconditionally; records that
    don't provide a sparse vector receive a minimal placeholder {0: 0.0} so
    Milvus doesn't reject the insert.
"""

import json
import logging
import numpy as np
from typing import List, Dict, Any, Optional

from pymilvus import (
    connections,
    Collection,
    CollectionSchema,
    FieldSchema,
    DataType,
    utility,
    AnnSearchRequest,
    RRFRanker,
)

from src.config import settings

logger = logging.getLogger(__name__)

# ── Field name constants ──────────────────────────────────────────────────────
F_ID         = "id"
F_DOC_ID     = "document_id"
F_USER_ID    = "user_id"
F_CONTENT    = "content"
F_DENSE      = "dense_vector"
F_SPARSE     = "sparse_vector"
F_PAGE       = "page_number"
F_CHUNK_TYPE = "chunk_type"
F_PARENT_ID  = "parent_id"
F_IS_PARENT  = "is_parent"
F_METADATA   = "metadata_json"

DENSE_DIM      = 1024
MAX_CONTENT    = 65_535
MAX_METADATA   = 1_024
_SPARSE_PLACEHOLDER: Dict[int, float] = {0: 0.0}   # for non-sparse records


# ─── Connection ───────────────────────────────────────────────────────────────

def _connect():
    """Open (or reuse) a named connection to Milvus."""
    try:
        connections.connect(
            alias=settings.MILVUS_ALIAS,
            uri=settings.MILVUS_URI,
            token=settings.MILVUS_TOKEN or None,
        )
    except Exception as exc:
        logger.error(f"Milvus connection failed: {exc}")
        raise


# ─── Collection bootstrap ─────────────────────────────────────────────────────

def ensure_collection(name: Optional[str] = None) -> Collection:
    """
    Return the Milvus collection, creating and indexing it if it doesn't exist.
    Safe to call on every request (cheap no-op when collection already exists).
    """
    name = name or settings.MILVUS_COLLECTION
    _connect()

    if utility.has_collection(name, using=settings.MILVUS_ALIAS):
        col = Collection(name, using=settings.MILVUS_ALIAS)
        col.load()
        return col

    logger.info(f"Creating Milvus collection '{name}' …")

    schema = CollectionSchema(
        fields=[
            FieldSchema(F_ID,         DataType.VARCHAR,          max_length=64,           is_primary=True),
            FieldSchema(F_DOC_ID,     DataType.VARCHAR,          max_length=64),
            FieldSchema(F_USER_ID,    DataType.VARCHAR,          max_length=64),
            FieldSchema(F_CONTENT,    DataType.VARCHAR,          max_length=MAX_CONTENT),
            FieldSchema(F_DENSE,      DataType.FLOAT_VECTOR,     dim=DENSE_DIM),
            FieldSchema(F_SPARSE,     DataType.SPARSE_FLOAT_VECTOR),
            FieldSchema(F_PAGE,       DataType.INT32),
            FieldSchema(F_CHUNK_TYPE, DataType.VARCHAR,          max_length=32),
            FieldSchema(F_PARENT_ID,  DataType.VARCHAR,          max_length=64),
            FieldSchema(F_IS_PARENT,  DataType.BOOL),
            FieldSchema(F_METADATA,   DataType.VARCHAR,          max_length=MAX_METADATA),
        ],
        description="RAG document chunks — dense + sparse vectors",
        enable_dynamic_field=False,
    )

    col = Collection(
        name=name,
        schema=schema,
        using=settings.MILVUS_ALIAS,
        consistency_level="Strong",
    )

    # Dense HNSW index
    col.create_index(
        field_name=F_DENSE,
        index_params={
            "index_type":  settings.MILVUS_INDEX_TYPE,   # HNSW
            "metric_type": settings.MILVUS_METRIC_TYPE,  # COSINE
            "params":      {"M": 16, "efConstruction": 200},
        },
    )

    # Sparse inverted index (Milvus 2.4+)
    col.create_index(
        field_name=F_SPARSE,
        index_params={
            "index_type":  "SPARSE_INVERTED_INDEX",
            "metric_type": "IP",
            "params":      {"drop_ratio_build": 0.2},
        },
    )

    col.load()
    logger.info(f"Collection '{name}' created and loaded.")
    return col


# ─── Write operations ─────────────────────────────────────────────────────────

def insert_chunks(
    records:         List[Dict[str, Any]],
    collection_name: Optional[str] = None,
) -> int:
    """
    Insert a list of chunk records into Milvus.

    Expected keys per record:
        id, document_id, user_id, content,
        dense_vector   (np.ndarray shape [1024]),
        sparse_vector  (Dict[int, float], optional — placeholder used if absent),
        page_number, chunk_type, parent_id, is_parent, metadata_json

    Returns:
        Number of records inserted.
    """
    if not records:
        return 0

    col = ensure_collection(collection_name)

    # Build column-oriented data (Milvus bulk insert format)
    data = {
        F_ID:         [r[F_ID]                                       for r in records],
        F_DOC_ID:     [r[F_DOC_ID]                                   for r in records],
        F_USER_ID:    [r[F_USER_ID]                                  for r in records],
        F_CONTENT:    [r[F_CONTENT][:MAX_CONTENT]                    for r in records],
        F_DENSE:      [r[F_DENSE].tolist()                           for r in records],
        F_SPARSE:     [r.get(F_SPARSE, _SPARSE_PLACEHOLDER)         for r in records],
        F_PAGE:       [int(r.get(F_PAGE, 0))                        for r in records],
        F_CHUNK_TYPE: [r.get(F_CHUNK_TYPE, "text")                  for r in records],
        F_PARENT_ID:  [r.get(F_PARENT_ID, "")                       for r in records],
        F_IS_PARENT:  [bool(r.get(F_IS_PARENT, False))              for r in records],
        F_METADATA:   [r.get(F_METADATA, "{}")[:MAX_METADATA]        for r in records],
    }

    col.insert(data)
    col.flush()
    logger.info(f"Inserted {len(records)} records into '{col.name}'.")
    return len(records)


def delete_document_chunks(
    document_id:     str,
    collection_name: Optional[str] = None,
) -> None:
    """Delete all chunks belonging to a document (called before re-ingestion)."""
    col = ensure_collection(collection_name)
    col.delete(expr=f'{F_DOC_ID} == "{document_id}"')
    col.flush()
    logger.info(f"Deleted chunks for document_id='{document_id}'.")


# ─── Search helpers ───────────────────────────────────────────────────────────

_OUTPUT_FIELDS = [F_ID, F_DOC_ID, F_CONTENT, F_PAGE, F_CHUNK_TYPE, F_PARENT_ID, F_METADATA]


def _user_filter(user_id: str, document_ids: Optional[List[str]]) -> str:
    """Build the Milvus boolean expression for ownership + optional doc filter."""
    expr = f'{F_USER_ID} == "{user_id}" && {F_IS_PARENT} == false'
    if document_ids:
        ids_str = ", ".join(f'"{d}"' for d in document_ids)
        expr += f' && {F_DOC_ID} in [{ids_str}]'
    return expr


def _hits_to_dicts(hits) -> List[Dict[str, Any]]:
    """Convert a Milvus result set to plain dicts."""
    out = []
    for hit in hits:
        e = hit.entity
        out.append({
            "id":          e.get(F_ID,         ""),
            "document_id": e.get(F_DOC_ID,     ""),
            "content":     e.get(F_CONTENT,    ""),
            "score":       float(hit.score),
            "page_number": e.get(F_PAGE,        0),
            "chunk_type":  e.get(F_CHUNK_TYPE, "text"),
            "parent_id":   e.get(F_PARENT_ID,  ""),
            "metadata":    e.get(F_METADATA,   "{}"),
        })
    return out


# ─── Search operations ────────────────────────────────────────────────────────

def dense_search(
    query_vector:    np.ndarray,
    user_id:         str,
    document_ids:    Optional[List[str]] = None,
    top_k:           int                 = 10,
    collection_name: Optional[str]       = None,
) -> List[Dict[str, Any]]:
    """
    Approximate nearest-neighbour search on the dense vector field.
    Best for semantic / paraphrase queries.
    """
    col  = ensure_collection(collection_name)
    expr = _user_filter(user_id, document_ids)

    results = col.search(
        data=[query_vector.tolist()],
        anns_field=F_DENSE,
        param={
            "metric_type": settings.MILVUS_METRIC_TYPE,
            "params":      {"ef": max(top_k * 4, 64)},
        },
        limit=top_k,
        expr=expr,
        output_fields=_OUTPUT_FIELDS,
    )
    return _hits_to_dicts(results[0] if results else [])


def sparse_search(
    query_sparse:    Dict[int, float],
    user_id:         str,
    document_ids:    Optional[List[str]] = None,
    top_k:           int                 = 10,
    collection_name: Optional[str]       = None,
) -> List[Dict[str, Any]]:
    """
    Sparse lexical search on the SPARSE_FLOAT_VECTOR field.
    Best for exact-term queries (codes, proper nouns, acronyms).
    Requires bge-m3 embeddings; falls back to dense if sparse not available.
    """
    col  = ensure_collection(collection_name)
    expr = _user_filter(user_id, document_ids)

    results = col.search(
        data=[query_sparse],
        anns_field=F_SPARSE,
        param={
            "metric_type": "IP",
            "params":      {"drop_ratio_search": 0.2},
        },
        limit=top_k,
        expr=expr,
        output_fields=_OUTPUT_FIELDS,
    )
    return _hits_to_dicts(results[0] if results else [])


def hybrid_search(
    query_dense:     np.ndarray,
    query_sparse:    Dict[int, float],
    user_id:         str,
    document_ids:    Optional[List[str]] = None,
    top_k:           int                 = 10,
    collection_name: Optional[str]       = None,
) -> List[Dict[str, Any]]:
    """
    Hybrid search: dense ANN + sparse IP combined with Reciprocal Rank Fusion.

    RRF (k=60) provides robust score fusion without needing manual weight tuning.
    Consistently outperforms either dense or sparse alone on most benchmarks.
    Requires bge-m3 (provides both vector types).
    """
    col  = ensure_collection(collection_name)
    expr = _user_filter(user_id, document_ids)

    dense_req = AnnSearchRequest(
        data=[query_dense.tolist()],
        anns_field=F_DENSE,
        param={
            "metric_type": settings.MILVUS_METRIC_TYPE,
            "params":      {"ef": max(top_k * 4, 64)},
        },
        limit=top_k * 2,
        expr=expr,
    )

    sparse_req = AnnSearchRequest(
        data=[query_sparse],
        anns_field=F_SPARSE,
        param={
            "metric_type": "IP",
            "params":      {"drop_ratio_search": 0.2},
        },
        limit=top_k * 2,
        expr=expr,
    )

    results = col.hybrid_search(
        reqs=[dense_req, sparse_req],
        rerank=RRFRanker(k=60),
        limit=top_k,
        output_fields=_OUTPUT_FIELDS,
    )
    return _hits_to_dicts(results[0] if results else [])


# ─── Parent chunk fetch ───────────────────────────────────────────────────────

def fetch_parent_chunks(
    parent_ids:      List[str],
    collection_name: Optional[str] = None,
) -> Dict[str, str]:
    """
    Fetch full parent chunk content by chunk IDs.
    Used in parent-child retrieval to replace child content with richer context.

    Returns:
        {parent_id: content} mapping.
    """
    if not parent_ids:
        return {}

    col     = ensure_collection(collection_name)
    ids_str = ", ".join(f'"{pid}"' for pid in parent_ids)
    rows    = col.query(
        expr=f'{F_ID} in [{ids_str}] && {F_IS_PARENT} == true',
        output_fields=[F_ID, F_CONTENT],
    )
    return {r[F_ID]: r[F_CONTENT] for r in rows}