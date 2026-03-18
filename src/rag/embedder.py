"""
Embedding module.

Two production-grade open-source models from HuggingFace:

┌──────────────────────┬──────────────────┬──────────────────────────────────────┐
│ Model                │ Dim  │ Retrieval │ Notes                                 │
├──────────────────────┼──────┼───────────┼───────────────────────────────────────┤
│ BAAI/bge-m3          │ 1024 │ Dense +   │ Multilingual. Enables hybrid search.   │
│                      │      │ Sparse    │ Best overall for RAG.                 │
├──────────────────────┼──────┼───────────┼───────────────────────────────────────┤
│ BAAI/bge-large-en-v1.5│ 1024│ Dense     │ English-only. Fastest inference.      │
│                      │      │ only      │ Great dense-only baseline.            │
└──────────────────────┴──────┴───────────┴───────────────────────────────────────┘

Both models are singletons — loaded once and reused across requests.
"""

import logging
import numpy as np
from abc import ABC, abstractmethod
from typing import List, Dict, Tuple, Optional

logger = logging.getLogger(__name__)

# Type alias: sparse vector as token-id → weight dict
SparseVector = Dict[int, float]


# ─── Abstract Base ─────────────────────────────────────────────────────────────

class BaseEmbedder(ABC):
    """
    Common interface for all embedders used in the RAG pipeline.

    Concrete embedders must implement:
      - ``model_name``   (property)
      - ``dense_dim``    (property)
      - ``encode_dense`` (method)

    Sparse encoding is optional — override ``encode_sparse`` and set
    ``supports_sparse = True`` in subclasses that provide it.
    """

    @property
    @abstractmethod
    def model_name(self) -> str: ...

    @property
    @abstractmethod
    def dense_dim(self) -> int: ...

    @property
    def supports_sparse(self) -> bool:
        return False

    @abstractmethod
    def encode_dense(self, texts: List[str]) -> np.ndarray:
        """
        Encode texts into L2-normalised dense vectors.

        Returns:
            np.ndarray of shape (len(texts), dense_dim), dtype float32.
        """
        ...

    def encode_sparse(self, texts: List[str]) -> List[SparseVector]:
        """
        Encode texts into sparse lexical-weight vectors.
        Only available when ``supports_sparse`` is True.

        Returns:
            List of {token_id: weight} dicts (one per input text).
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} does not support sparse encoding."
        )

    def encode_both(
        self, texts: List[str]
    ) -> Tuple[np.ndarray, List[SparseVector]]:
        """
        Efficient joint encode for hybrid search.
        Override in models that can compute both in a single forward pass.
        Default falls back to two separate calls.
        """
        return self.encode_dense(texts), self.encode_sparse(texts)


# ─── BAAI/bge-m3 ──────────────────────────────────────────────────────────────

class BGEM3Embedder(BaseEmbedder):
    """
    BAAI/bge-m3  via  FlagEmbedding.

    Strengths
    ---------
    • Multilingual (100+ languages).
    • Produces dense vectors, sparse (SPLADE-style) lexical weights, and
      ColBERT multi-vector representations in a single model.
    • Dense + sparse enables Milvus hybrid search with RRF reranking.

    Usage
    -----
    Always obtain via ``BGEM3Embedder.get_instance()`` to avoid reloading.
    """

    _instance: Optional["BGEM3Embedder"] = None

    def __init__(self, device: str = "cpu", batch_size: int = 32):
        try:
            from FlagEmbedding import BGEM3FlagModel
        except ImportError:
            raise ImportError(
                "FlagEmbedding is required for bge-m3.  "
                "Install it with: pip install FlagEmbedding"
            )

        logger.info(f"Loading BAAI/bge-m3 on {device} …")
        self._model      = BGEM3FlagModel(
            "BAAI/bge-m3",
            use_fp16=(device != "cpu"),
            device=device,
        )
        self._batch_size = batch_size
        logger.info("BAAI/bge-m3 ready.")

    @classmethod
    def get_instance(cls, device: str = "cpu") -> "BGEM3Embedder":
        if cls._instance is None:
            cls._instance = cls(device=device)
        return cls._instance

    @property
    def model_name(self) -> str:
        return "BAAI/bge-m3"

    @property
    def dense_dim(self) -> int:
        return 1024

    @property
    def supports_sparse(self) -> bool:
        return True

    def encode_dense(self, texts: List[str]) -> np.ndarray:
        out = self._model.encode(
            texts,
            batch_size=self._batch_size,
            return_dense=True,
            return_sparse=False,
            return_colbert_vecs=False,
        )
        return np.array(out["dense_vecs"], dtype=np.float32)

    def encode_sparse(self, texts: List[str]) -> List[SparseVector]:
        out = self._model.encode(
            texts,
            batch_size=self._batch_size,
            return_dense=False,
            return_sparse=True,
            return_colbert_vecs=False,
        )
        return out["lexical_weights"]   # List[Dict[int, float]]

    def encode_both(
        self, texts: List[str]
    ) -> Tuple[np.ndarray, List[SparseVector]]:
        """Single forward pass — more efficient than calling each separately."""
        out = self._model.encode(
            texts,
            batch_size=self._batch_size,
            return_dense=True,
            return_sparse=True,
            return_colbert_vecs=False,
        )
        dense  = np.array(out["dense_vecs"], dtype=np.float32)
        sparse = out["lexical_weights"]
        return dense, sparse


# ─── BAAI/bge-large-en-v1.5 ──────────────────────────────────────────────────

class BGELargeEmbedder(BaseEmbedder):
    """
    BAAI/bge-large-en-v1.5  via  sentence-transformers.

    Strengths
    ---------
    • Best English-only dense retrieval quality (MTEB leaderboard top-tier).
    • Faster inference than bge-m3 (no sparse computation overhead).
    • Lighter memory footprint — good choice for CPU-only deployments.

    Limitations
    -----------
    • English only.
    • No native sparse vectors — pair with BM25 for hybrid search.
      (This implementation falls back to dense for sparse/hybrid queries.)

    Usage
    -----
    Always obtain via ``BGELargeEmbedder.get_instance()`` to avoid reloading.
    """

    _instance: Optional["BGELargeEmbedder"] = None

    def __init__(self, device: str = "cpu", batch_size: int = 64):
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError:
            raise ImportError(
                "sentence-transformers is required for bge-large.  "
                "Install it with: pip install sentence-transformers"
            )

        logger.info(f"Loading BAAI/bge-large-en-v1.5 on {device} …")
        self._model      = SentenceTransformer(
            "BAAI/bge-large-en-v1.5",
            device=device,
        )
        self._batch_size = batch_size
        logger.info("BAAI/bge-large-en-v1.5 ready.")

    @classmethod
    def get_instance(cls, device: str = "cpu") -> "BGELargeEmbedder":
        if cls._instance is None:
            cls._instance = cls(device=device)
        return cls._instance

    @property
    def model_name(self) -> str:
        return "BAAI/bge-large-en-v1.5"

    @property
    def dense_dim(self) -> int:
        return 1024

    @property
    def supports_sparse(self) -> bool:
        return False  # No native sparse; bge-m3 is needed for hybrid

    def encode_dense(self, texts: List[str]) -> np.ndarray:
        vecs = self._model.encode(
            texts,
            batch_size=self._batch_size,
            normalize_embeddings=True,
            show_progress_bar=False,
            convert_to_numpy=True,
        )
        return np.array(vecs, dtype=np.float32)


# ─── Factory ──────────────────────────────────────────────────────────────────

def get_embedder(model_name: str, device: str = "cpu") -> BaseEmbedder:
    """
    Return the singleton embedder for the given model name.

    Args:
        model_name: One of ``"BAAI/bge-m3"`` or ``"BAAI/bge-large-en-v1.5"``.
        device:     ``"cpu"`` | ``"cuda"`` | ``"mps"``.

    Raises:
        ValueError: If model_name is not recognised.
    """
    if model_name == "BAAI/bge-m3":
        return BGEM3Embedder.get_instance(device=device)
    elif model_name == "BAAI/bge-large-en-v1.5":
        return BGELargeEmbedder.get_instance(device=device)
    else:
        raise ValueError(
            f"Unknown embedding model: '{model_name}'.  "
            f"Choose 'BAAI/bge-m3' or 'BAAI/bge-large-en-v1.5'."
        )