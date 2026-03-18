"""
Chunking strategies for the RAG pipeline.

┌─────────────────────────────────────────────────────────────────────────────┐
│  RecursiveChunker   — fast, reliable, language-aware splitting              │
│  SemanticChunker    — embedding-guided semantic boundary detection          │
│  ParentChildChunker — small retrieval units + large context windows         │
└─────────────────────────────────────────────────────────────────────────────┘

All chunkers return Chunk dataclasses.  ParentChildChunker additionally returns
a parallel list of parent Chunks (which carry the full context for each group
of children).
"""

import re
import uuid
import logging
import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

logger = logging.getLogger(__name__)


# ─── Chunk Dataclass ──────────────────────────────────────────────────────────

@dataclass
class Chunk:
    content:    str
    chunk_id:   str                    = field(default_factory=lambda: str(uuid.uuid4()))
    parent_id:  Optional[str]          = None      # set for parent-child children
    chunk_type: str                    = "text"    # text | table | parent
    metadata:   dict                   = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.content)


# ─── 1. RecursiveChunker ──────────────────────────────────────────────────────

class RecursiveChunker:
    """
    Splits text recursively on a hierarchy of separators:
        paragraph → newline → sentence → comma → word → character

    Mirrors the behaviour of LangChain's RecursiveCharacterTextSplitter
    but with no external dependency.

    Args:
        chunk_size:    Max characters per chunk (default 1000).
        chunk_overlap: Characters of overlap between consecutive chunks (default 150).
    """

    _SEPARATORS = ["\n\n", "\n", ". ", "! ", "? ", "; ", ", ", " ", ""]

    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 150):
        self.chunk_size    = chunk_size
        self.chunk_overlap = chunk_overlap

    def split(self, text: str, metadata: Optional[dict] = None) -> List[Chunk]:
        raw = self._split_text(text, self._SEPARATORS)
        chunks = []
        for i, content in enumerate(raw):
            content = content.strip()
            if not content:
                continue
            chunks.append(Chunk(
                content=content,
                metadata={**(metadata or {}), "chunk_index": i},
            ))
        return chunks

    # ── internals ──────────────────────────────────────────────────

    def _split_text(self, text: str, separators: List[str]) -> List[str]:
        """Recursively split `text` using the first separator that matches."""
        final_chunks: List[str] = []

        # Pick the first separator found in the text
        sep = ""
        new_seps: List[str] = []
        for i, s in enumerate(separators):
            if s == "" or re.search(re.escape(s), text):
                sep      = s
                new_seps = separators[i + 1:]
                break

        splits = re.split(re.escape(sep), text) if sep else [text]

        good: List[str] = []
        for s in splits:
            if len(s) <= self.chunk_size:
                good.append(s)
            else:
                if good:
                    final_chunks.extend(self._merge(good, sep))
                    good = []
                # Recurse with finer-grained separators
                if new_seps:
                    final_chunks.extend(self._split_text(s, new_seps))
                else:
                    final_chunks.append(s)

        if good:
            final_chunks.extend(self._merge(good, sep))

        return final_chunks

    def _merge(self, splits: List[str], sep: str) -> List[str]:
        """Greedily merge small splits into chunks ≤ chunk_size with overlap."""
        chunks:  List[str]  = []
        current: List[str]  = []
        current_len: int    = 0

        for piece in splits:
            piece_len = len(piece)
            join_len  = len(sep) if current else 0

            if current_len + join_len + piece_len > self.chunk_size and current:
                chunks.append(sep.join(current))
                # Retain overlap: drop from front until overlap budget satisfied
                while current:
                    drop_len = len(current[0]) + len(sep)
                    if current_len - drop_len >= self.chunk_overlap:
                        current_len -= drop_len
                        current.pop(0)
                    else:
                        break

            current.append(piece)
            current_len += piece_len + (len(sep) if len(current) > 1 else 0)

        if current:
            chunks.append(sep.join(current))
        return chunks


# ─── 2. SemanticChunker ───────────────────────────────────────────────────────

class SemanticChunker:
    """
    Groups sentences into semantically coherent chunks by measuring the cosine
    distance between consecutive sentence embeddings.  A high distance signals a
    topic shift → new chunk begins.

    Args:
        embedder:                  Any embedder with an ``encode_dense(texts)``
                                   method that returns a (N, D) np.ndarray.
        breakpoint_threshold:      Cosine distance above which a new chunk starts
                                   (0 = no splits, 1 = split everywhere; default 0.35).
        max_chunk_size:            Hard cap on characters per output chunk (default 1500).
        min_chunk_size:            Minimum characters; smaller groups are merged forward
                                   (default 100).
    """

    def __init__(
        self,
        embedder,
        breakpoint_threshold: float = 0.35,
        max_chunk_size:        int   = 1500,
        min_chunk_size:        int   = 100,
    ):
        self.embedder              = embedder
        self.breakpoint_threshold  = breakpoint_threshold
        self.max_chunk_size        = max_chunk_size
        self.min_chunk_size        = min_chunk_size
        self._fallback             = RecursiveChunker(
            chunk_size=max_chunk_size, chunk_overlap=50
        )

    def split(self, text: str, metadata: Optional[dict] = None) -> List[Chunk]:
        sentences = self._sentence_split(text)
        if len(sentences) <= 2:
            # Too short to detect semantics; fall back to recursive
            return self._fallback.split(text, metadata)

        logger.debug(f"SemanticChunker: embedding {len(sentences)} sentences …")
        embeddings = self.embedder.encode_dense(sentences)   # (N, D)

        breakpoints = self._find_breakpoints(embeddings)
        groups      = self._group(sentences, breakpoints)
        chunks      = self._build_chunks(groups, metadata or {})
        return chunks

    # ── internals ──────────────────────────────────────────────────

    @staticmethod
    def _sentence_split(text: str) -> List[str]:
        """Sentence tokeniser: split on terminal punctuation + whitespace."""
        # Preserve abbreviations crudely (Dr., Mr., etc.) by requiring ≥ 2 tokens
        raw = re.split(r'(?<=[.!?])\s+(?=[A-Z])', text)
        return [s.strip() for s in raw if s.strip()]

    @staticmethod
    def _cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
        denom = (np.linalg.norm(a) * np.linalg.norm(b)) + 1e-10
        return float(1.0 - np.dot(a, b) / denom)

    def _find_breakpoints(self, embeddings: np.ndarray) -> List[int]:
        """Return sentence indices (0-based) where a new chunk should start."""
        bp = []
        for i in range(len(embeddings) - 1):
            dist = self._cosine_distance(embeddings[i], embeddings[i + 1])
            if dist > self.breakpoint_threshold:
                bp.append(i + 1)
        return bp

    def _group(
        self,
        sentences:   List[str],
        breakpoints: List[int],
    ) -> List[List[str]]:
        bp_set = set(breakpoints)
        groups: List[List[str]] = []
        current: List[str] = []

        for i, s in enumerate(sentences):
            if i in bp_set and current:
                groups.append(current)
                current = []
            current.append(s)

        if current:
            groups.append(current)
        return groups

    def _build_chunks(
        self,
        groups:   List[List[str]],
        metadata: dict,
    ) -> List[Chunk]:
        chunks: List[Chunk] = []
        chunk_index = 0
        pending: List[str] = []   # accumulates small groups

        def flush(sents: List[str]):
            nonlocal chunk_index
            if not sents:
                return
            content = " ".join(sents).strip()
            if not content:
                return
            if len(content) > self.max_chunk_size:
                # Sub-split oversized semantic groups
                for sub in self._fallback.split(content, {**metadata, "chunk_index": chunk_index}):
                    chunks.append(sub)
                    chunk_index += 1
            else:
                chunks.append(Chunk(
                    content=content,
                    metadata={**metadata, "chunk_index": chunk_index},
                ))
                chunk_index += 1

        for group in groups:
            group_text = " ".join(group)
            if len(group_text) < self.min_chunk_size:
                # Too small — accumulate with next group
                pending.extend(group)
            else:
                if pending:
                    flush(pending)
                    pending = []
                flush(group)

        if pending:
            flush(pending)

        return chunks


# ─── 3. ParentChildChunker ────────────────────────────────────────────────────

class ParentChildChunker:
    """
    Two-level chunking strategy:

    • **Parent chunks** (large) — stored with `is_parent=True`, never embedded
      for retrieval.  They provide rich context that is sent to the LLM.
    • **Child chunks** (small) — embedded and indexed in Milvus.  Each child
      carries a `parent_id` so its parent can be fetched at query time.

    The retriever finds relevant child chunks, then replaces them with the
    full parent text before calling the LLM → better precision AND recall.

    Args:
        parent_chunk_size: Characters per parent chunk (default 2000).
        parent_overlap:    Overlap between parent chunks (default 100).
        child_chunk_size:  Characters per child chunk (default 400).
        child_overlap:     Overlap between sibling children (default 50).
    """

    def __init__(
        self,
        parent_chunk_size: int = 2000,
        parent_overlap:    int = 100,
        child_chunk_size:  int = 400,
        child_overlap:     int = 50,
    ):
        self._parent_splitter = RecursiveChunker(
            chunk_size=parent_chunk_size,
            chunk_overlap=parent_overlap,
        )
        self._child_splitter = RecursiveChunker(
            chunk_size=child_chunk_size,
            chunk_overlap=child_overlap,
        )

    def split(
        self,
        text:     str,
        metadata: Optional[dict] = None,
    ) -> Tuple[List[Chunk], List[Chunk]]:
        """
        Returns (parent_chunks, child_chunks).

        Each parent chunk has ``chunk_type="parent"`` and ``chunk_id`` set.
        Each child chunk has ``chunk_type="text"`` and ``parent_id`` pointing
        to its parent's ``chunk_id``.
        """
        metadata = metadata or {}
        parent_raw = self._parent_splitter.split(text, metadata)

        all_parents: List[Chunk] = []
        all_children: List[Chunk] = []

        for p_idx, parent in enumerate(parent_raw):
            # Assign stable parent ID
            parent.chunk_id   = str(uuid.uuid4())
            parent.chunk_type = "parent"
            parent.metadata   = {
                **parent.metadata,
                "parent_index": p_idx,
                "is_parent":    True,
            }
            all_parents.append(parent)

            # Split children from parent content
            children = self._child_splitter.split(
                parent.content,
                {
                    **metadata,
                    "parent_index": p_idx,
                    "is_parent":    False,
                },
            )
            for c_idx, child in enumerate(children):
                child.chunk_id  = str(uuid.uuid4())
                child.parent_id = parent.chunk_id
                child.metadata.update({"child_index": c_idx})
                all_children.append(child)

        logger.debug(
            f"ParentChildChunker: {len(all_parents)} parents, "
            f"{len(all_children)} children"
        )
        return all_parents, all_children