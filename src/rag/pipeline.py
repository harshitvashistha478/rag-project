"""
RAG Pipeline — orchestrates the full ingestion and query workflows.

Ingestion flow
──────────────
  S3 download → PDF parse → chunk → embed → Milvus insert
                                     ↑
                         strategy & model from RAGConfig

Query flow
──────────
  embed question → retrieve (dense|sparse|hybrid)
       → [parent-child: expand to parents]
       → build context → Gemini → answer
"""

import json
import re
import uuid
import logging
import numpy as np
from typing import List, Optional, Tuple

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import SystemMessage, HumanMessage

from src.config import settings
from src.schemas.rag import (
    RAGConfig,
    ChunkingStrategy,
    RetrievalStrategy,
    SourceChunk,
    QueryResponse,
)
from src.rag.loader import load_pdf_from_bytes, ParsedDocument
from src.rag.chunker import (
    Chunk,
    RecursiveChunker,
    SemanticChunker,
    ParentChildChunker,
)
from src.rag.embedder import get_embedder, BGEM3Embedder
from src.rag import vector_store as vs
from src.utils.s3_functions import download_file_from_s3

logger = logging.getLogger(__name__)

BATCH_SIZE = 64   # embedding batch size


# ─── Ingestion ────────────────────────────────────────────────────────────────

def ingest_document(
    document_id:       str,
    user_id:           str,
    s3_key:            str,
    original_filename: str,
    config:            RAGConfig,
) -> int:
    """
    Full ingestion pipeline for a single PDF document.

    Steps
    -----
    1. Download raw bytes from S3.
    2. Parse PDF → text + tables + image metadata.
    3. Chunk according to ``config.chunking_strategy``.
    4. Embed chunks in batches.
    5. Delete any existing vectors for this document (idempotent re-ingestion).
    6. Insert new vectors into Milvus.

    Returns
    -------
    Number of **child** chunks stored (what gets retrieved).
    """
    logger.info(f"[ingest] Starting document={document_id}")

    # ── 1. Download ───────────────────────────────────────────────
    content = download_file_from_s3(s3_key)
    logger.info(f"[ingest] Downloaded {len(content):,} bytes from S3")

    # ── 2. Parse ──────────────────────────────────────────────────
    parsed = load_pdf_from_bytes(content, original_filename)
    logger.info(
        f"[ingest] Parsed: {len(parsed.pages)} pages | "
        f"tables={parsed.has_tables} | images={parsed.has_images}"
    )

    # ── 3. Chunk ──────────────────────────────────────────────────
    embedder = get_embedder(config.embedding_model, settings.EMBEDDING_DEVICE)
    child_chunks, parent_chunks = _chunk(parsed, config, embedder)
    logger.info(
        f"[ingest] Chunked → {len(child_chunks)} children, "
        f"{len(parent_chunks)} parents"
    )

    if not child_chunks:
        logger.warning(f"[ingest] No chunks produced for document={document_id}")
        return 0

    # ── 4. Embed ──────────────────────────────────────────────────
    records = _embed_chunks(
        chunks=child_chunks,
        parent_chunks=parent_chunks,
        document_id=document_id,
        user_id=user_id,
        original_filename=original_filename,
        config=config,
        embedder=embedder,
    )

    # ── 5. Delete old vectors (idempotent) ────────────────────────
    vs.delete_document_chunks(document_id)

    # ── 6. Insert ─────────────────────────────────────────────────
    vs.insert_chunks(records)

    child_count = sum(1 for r in records if not r[vs.F_IS_PARENT])
    logger.info(f"[ingest] Done — {child_count} retrieval chunks stored.")
    return child_count


# ─── Chunking helper ──────────────────────────────────────────────────────────

def _chunk(
    parsed:  ParsedDocument,
    config:  RAGConfig,
    embedder,
) -> Tuple[List[Chunk], List[Chunk]]:
    """
    Dispatch to the configured chunking strategy.
    Returns (child_chunks, parent_chunks).
    parent_chunks is empty for recursive and semantic strategies.
    """
    # Annotate text with page markers so chunker metadata can surface page numbers.
    text_parts = []
    for page in parsed.pages:
        if page.markdown.strip():
            text_parts.append(f"<!-- Page {page.page_number} -->\n{page.markdown.strip()}")
    full_text = "\n\n---\n\n".join(text_parts)

    base_meta = {
        "filename":    parsed.metadata.get("filename", ""),
        "page_count":  parsed.metadata.get("page_count", 0),
    }

    if config.chunking_strategy == ChunkingStrategy.RECURSIVE:
        splitter = RecursiveChunker(
            chunk_size=config.chunk_size,
            chunk_overlap=config.chunk_overlap,
        )
        return splitter.split(full_text, base_meta), []

    elif config.chunking_strategy == ChunkingStrategy.SEMANTIC:
        splitter = SemanticChunker(
            embedder=embedder,
            breakpoint_threshold=config.semantic_breakpoint_threshold,
            max_chunk_size=config.chunk_size,
        )
        return splitter.split(full_text, base_meta), []

    elif config.chunking_strategy == ChunkingStrategy.PARENT_CHILD:
        splitter = ParentChildChunker(
            parent_chunk_size=config.parent_chunk_size,
            parent_overlap=100,
            child_chunk_size=config.child_chunk_size,
            child_overlap=config.child_overlap,
        )
        parents, children = splitter.split(full_text, base_meta)
        return children, parents

    raise ValueError(f"Unknown chunking strategy: {config.chunking_strategy}")


# ─── Embedding helper ─────────────────────────────────────────────────────────

def _embed_chunks(
    chunks:            List[Chunk],
    parent_chunks:     List[Chunk],
    document_id:       str,
    user_id:           str,
    original_filename: str,
    config:            RAGConfig,
    embedder,
) -> List[dict]:
    """
    Embed child chunks in batches and build Milvus-ready record dicts.
    Parent chunks receive a zero-vector (they are stored for context, not retrieval).
    """
    needs_sparse = (
        isinstance(embedder, BGEM3Embedder)
        and config.retrieval_strategy in (RetrievalStrategy.SPARSE, RetrievalStrategy.HYBRID)
    )
    zero_vec = np.zeros(embedder.dense_dim, dtype=np.float32)

    records: List[dict] = []

    # ── Child chunks (embedded for retrieval) ─────────────────────
    for batch_start in range(0, len(chunks), BATCH_SIZE):
        batch = chunks[batch_start: batch_start + BATCH_SIZE]
        texts = [c.content for c in batch]

        if needs_sparse:
            dense_vecs, sparse_vecs = embedder.encode_both(texts)
        else:
            dense_vecs  = embedder.encode_dense(texts)
            sparse_vecs = None

        for i, chunk in enumerate(batch):
            page_num = _parse_page_number(chunk.content)
            meta_str = json.dumps({
                "filename": original_filename,
                **chunk.metadata,
            })[:vs.MAX_METADATA]

            rec = {
                vs.F_ID:         chunk.chunk_id,
                vs.F_DOC_ID:     document_id,
                vs.F_USER_ID:    user_id,
                vs.F_CONTENT:    chunk.content,
                vs.F_DENSE:      dense_vecs[i],
                vs.F_PAGE:       page_num,
                vs.F_CHUNK_TYPE: chunk.chunk_type,
                vs.F_PARENT_ID:  chunk.parent_id or "",
                vs.F_IS_PARENT:  False,
                vs.F_METADATA:   meta_str,
            }
            if needs_sparse and sparse_vecs:
                rec[vs.F_SPARSE] = sparse_vecs[i]

            records.append(rec)

    # ── Parent chunks (zero-vector; context-only) ─────────────────
    for parent in parent_chunks:
        meta_str = json.dumps({
            "filename": original_filename,
            **parent.metadata,
        })[:vs.MAX_METADATA]

        records.append({
            vs.F_ID:         parent.chunk_id,
            vs.F_DOC_ID:     document_id,
            vs.F_USER_ID:    user_id,
            vs.F_CONTENT:    parent.content,
            vs.F_DENSE:      zero_vec,
            vs.F_PAGE:       0,
            vs.F_CHUNK_TYPE: "parent",
            vs.F_PARENT_ID:  "",
            vs.F_IS_PARENT:  True,
            vs.F_METADATA:   meta_str,
        })

    return records


def _parse_page_number(text: str) -> int:
    """Extract the page number from a `<!-- Page N -->` marker, or return 0."""
    m = re.search(r'<!--\s*Page\s+(\d+)\s*-->', text)
    return int(m.group(1)) if m else 0


# ─── Query ────────────────────────────────────────────────────────────────────

def query_documents(
    question:     str,
    user_id:      str,
    document_ids: Optional[List[str]],
    config:       RAGConfig,
    max_tokens:   int = 2048,
) -> QueryResponse:
    """
    Query the RAG system.

    Steps
    -----
    1. Embed the question (dense + sparse if bge-m3 and hybrid/sparse).
    2. Retrieve top-k chunks via configured strategy.
    3. Parent-child: replace child content with parent content for richer context.
    4. Build a structured prompt and call Gemini.

    Returns
    -------
    QueryResponse with answer, sources, and metadata.
    """
    embedder = get_embedder(config.embedding_model, settings.EMBEDDING_DEVICE)
    is_bge_m3 = isinstance(embedder, BGEM3Embedder)

    # ── Embed question ────────────────────────────────────────────
    use_sparse = (
        is_bge_m3
        and config.retrieval_strategy != RetrievalStrategy.DENSE
    )

    if use_sparse:
        dense_q_arr, sparse_q_list = embedder.encode_both([question])
        dense_q  = dense_q_arr[0]
        sparse_q = sparse_q_list[0]
    else:
        dense_q  = embedder.encode_dense([question])[0]
        sparse_q = None

    # ── Retrieve ──────────────────────────────────────────────────
    strategy = config.retrieval_strategy

    if strategy == RetrievalStrategy.DENSE or not use_sparse:
        if strategy != RetrievalStrategy.DENSE:
            logger.warning(
                f"Strategy '{strategy}' requested but embedder '{embedder.model_name}' "
                "does not support sparse vectors.  Falling back to dense."
            )
        hits = vs.dense_search(dense_q, user_id, document_ids, config.retrieval_k)

    elif strategy == RetrievalStrategy.SPARSE:
        hits = vs.sparse_search(sparse_q, user_id, document_ids, config.retrieval_k)

    else:  # HYBRID
        hits = vs.hybrid_search(dense_q, sparse_q, user_id, document_ids, config.retrieval_k)

    logger.info(f"[query] Retrieved {len(hits)} chunks via {strategy}.")

    # ── Expand to parents (parent-child strategy) ─────────────────
    if config.chunking_strategy == ChunkingStrategy.PARENT_CHILD:
        hits = _expand_to_parents(hits)

    # ── Build source objects ──────────────────────────────────────
    sources: List[SourceChunk] = []
    for hit in hits:
        meta = {}
        try:
            meta = json.loads(hit.get("metadata", "{}"))
        except json.JSONDecodeError:
            pass

        sources.append(SourceChunk(
            document_id=hit["document_id"],
            filename=meta.get("filename", ""),
            content=hit["content"],
            score=hit["score"],
            page_number=hit.get("page_number") or None,
            chunk_type=hit.get("chunk_type", "text"),
        ))

    # ── Generate answer ───────────────────────────────────────────
    answer = _generate_answer(question, sources, max_tokens)

    return QueryResponse(
        answer=answer,
        sources=sources,
        model_used=settings.GEMINI_CHAT_MODEL,
        retrieval_strategy=strategy.value,
        chunks_retrieved=len(sources),
    )


def _expand_to_parents(hits: List[dict]) -> List[dict]:
    """
    Replace each retrieved child with its parent chunk content.
    Deduplicates parents so the same parent isn't sent twice.
    """
    parent_ids = list({h["parent_id"] for h in hits if h.get("parent_id")})
    if not parent_ids:
        return hits

    parent_map = vs.fetch_parent_chunks(parent_ids)

    seen:    set       = set()
    expanded: List[dict] = []

    for hit in hits:
        pid = hit.get("parent_id", "")
        if pid and pid in parent_map:
            if pid not in seen:
                expanded.append({**hit, "content": parent_map[pid], "chunk_type": "parent"})
                seen.add(pid)
        else:
            expanded.append(hit)

    return expanded


# ─── LLM generation ───────────────────────────────────────────────────────────

_SYSTEM_PROMPT = """You are a precise document assistant.
Answer the user's question using ONLY the provided context.
If the context does not contain enough information, say so clearly.
Cite sources using [Source N] notation when referencing specific facts.
If a context block contains a table, refer to it accurately."""


def _build_llm(max_tokens: int) -> ChatGoogleGenerativeAI:
    """
    Construct a LangChain ChatGoogleGenerativeAI instance.

    The model is intentionally NOT cached as a module-level singleton because
    ``max_tokens`` varies per call.  ChatGoogleGenerativeAI is lightweight to
    instantiate — no model weights are loaded locally.
    """
    return ChatGoogleGenerativeAI(
        model=settings.GEMINI_CHAT_MODEL,
        google_api_key=settings.GOOGLE_API_KEY,
        temperature=settings.GEMINI_TEMPERATURE,
        max_output_tokens=max_tokens,
        # Convert system messages automatically for Gemini's API format
        convert_system_message_to_human=False,
    )


def _generate_answer(
    question:   str,
    sources:    List[SourceChunk],
    max_tokens: int = 2048,
) -> str:
    """
    Build a structured prompt from retrieved chunks and call Gemini via LangChain.

    Message layout
    ──────────────
    SystemMessage  — role instructions and citation rules
    HumanMessage   — context blocks + question (single turn; no chat history)

    Using a two-message structure rather than a single concatenated string lets
    LangChain correctly map to Gemini's ``system_instruction`` + ``user`` turn,
    which gives the model a cleaner signal about which part is instruction vs data.
    """
    if not sources:
        return (
            "I couldn't find relevant information in the provided documents "
            "to answer your question."
        )

    # ── Build numbered context blocks ─────────────────────────────
    context_blocks = []
    for i, src in enumerate(sources, start=1):
        page_label = f"Page {src.page_number}" if src.page_number else "Page N/A"
        context_blocks.append(
            f"[Source {i} | {src.filename} | {page_label} | {src.chunk_type}]\n"
            f"{src.content}"
        )
    context = "\n\n---\n\n".join(context_blocks)

    # ── Compose messages ──────────────────────────────────────────
    messages = [
        SystemMessage(content=_SYSTEM_PROMPT),
        HumanMessage(content=(
            f"CONTEXT:\n{context}\n\n"
            f"QUESTION:\n{question}\n\n"
            f"ANSWER:"
        )),
    ]

    llm      = _build_llm(max_tokens)
    response = llm.invoke(messages)
    return response.content.strip()