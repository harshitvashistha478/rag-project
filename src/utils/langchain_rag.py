"""
LangChain RAG Engine — Gemini 2.5 + BAAI/bge-m3
=================================================
LLM      : Gemini 2.5 Pro (via langchain-google-genai)
Embeddings: BAAI/bge-m3  (local, HuggingFace, free, multilingual)
Vector DB : pgvector (PostgreSQL)

Pipeline:
  1. Load  — LangChain document loaders (PDF/DOCX/TXT/CSV/XLSX)
  2. Split — RecursiveCharacterTextSplitter (character-based, bge-m3 uses its own tokenizer)
  3. Embed — HuggingFaceEmbeddings(BAAI/bge-m3) → 1024-dim vectors
  4. Store — PGVector with user_id + document_id metadata
  5. Retrieve — MMR → MultiQueryRetriever → ContextualCompressionRetriever
  6. Generate — Gemini 2.5 Pro with source-grounded system prompt
"""

import logging
import os
import tempfile
from typing import Optional

from langchain_core.documents import Document as LCDocument
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_postgres.vectorstores import PGVector as PGVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_classic.retrievers import MultiQueryRetriever
from langchain_classic.retrievers.document_compressors import LLMChainExtractor
from langchain_classic.retrievers.contextual_compression import ContextualCompressionRetriever

from src.config import settings
from src.utils.s3_functions import download_file_from_s3

logger = logging.getLogger(__name__)

VECTOR_COLLECTION = "rag_documents"


# ─── Singletons ───────────────────────────────────────────────────────────────

_embeddings: Optional[HuggingFaceEmbeddings] = None
_vector_store: Optional[PGVectorStore] = None
_llm: Optional[ChatGoogleGenerativeAI] = None


def get_embeddings() -> HuggingFaceEmbeddings:
    """
    BAAI/bge-m3 — loaded once, cached for the process lifetime.
    First call downloads the model (~570 MB) to ~/.cache/huggingface.
    Subsequent starts load from cache instantly.
    """
    global _embeddings
    if _embeddings is None:
        logger.info(f"Loading embedding model: {settings.EMBEDDING_MODEL_NAME}")
        _embeddings = HuggingFaceEmbeddings(
            model_name=settings.EMBEDDING_MODEL_NAME,
            model_kwargs={
                "device": settings.EMBEDDING_DEVICE,
            },
            encode_kwargs={
                "normalize_embeddings": True,   # cosine similarity works correctly
                "batch_size": 32,
            },
        )
        logger.info("Embedding model loaded.")
    return _embeddings


def get_vector_store() -> PGVectorStore:
    global _vector_store
    if _vector_store is None:
        _vector_store = PGVectorStore(
            embeddings=get_embeddings(),
            collection_name=VECTOR_COLLECTION,
            connection=settings.PGVECTOR_CONNECTION_STRING,
            use_jsonb=True,
        )
    return _vector_store


def get_llm() -> ChatGoogleGenerativeAI:
    """Gemini 2.5 Pro via Google Generative AI API."""
    global _llm
    if _llm is None:
        _llm = ChatGoogleGenerativeAI(
            model=settings.GEMINI_CHAT_MODEL,
            temperature=settings.GEMINI_TEMPERATURE,
            google_api_key=settings.GOOGLE_API_KEY,
            convert_system_message_to_human=True,  # Gemini requires this
        )
    return _llm


# ─── Step 1: Load from S3 ─────────────────────────────────────────────────────

def load_document_from_s3(
    s3_key: str,
    file_extension: str,
    original_filename: str,
) -> list[LCDocument]:
    """
    Downloads bytes from S3, writes to a temp file,
    loads with appropriate LangChain loader.
    """
    logger.info(f"Downloading from S3: {s3_key}")
    file_bytes = download_file_from_s3(s3_key)
    ext = file_extension.lower().strip(".")

    with tempfile.NamedTemporaryFile(delete=False, suffix=f".{ext}") as tmp:
        tmp.write(file_bytes)
        tmp_path = tmp.name

    try:
        docs = _load_by_extension(tmp_path, ext, original_filename)
    finally:
        os.unlink(tmp_path)

    logger.info(f"Loaded {len(docs)} section(s) from '{original_filename}'")
    return docs


def _load_by_extension(path: str, ext: str, filename: str) -> list[LCDocument]:
    if ext == "pdf":
        from langchain_community.document_loaders import PyPDFLoader
        loader = PyPDFLoader(path)
    elif ext in ("doc", "docx"):
        from langchain_community.document_loaders import Docx2txtLoader
        loader = Docx2txtLoader(path)
    elif ext in ("txt", "md"):
        from langchain_community.document_loaders import TextLoader
        loader = TextLoader(path, encoding="utf-8")
    elif ext == "csv":
        from langchain_community.document_loaders import CSVLoader
        loader = CSVLoader(path)
    elif ext in ("xls", "xlsx"):
        from langchain_community.document_loaders import UnstructuredExcelLoader
        loader = UnstructuredExcelLoader(path, mode="elements")
    else:
        raise ValueError(f"Unsupported extension: .{ext}")

    docs = loader.load()
    for doc in docs:
        doc.metadata["source_filename"] = filename
    return docs


# ─── Step 2: Split ────────────────────────────────────────────────────────────

def split_documents(
    docs: list[LCDocument],
    document_id: str,
    user_id: str,
    original_filename: str,
    s3_key: str,
    file_extension: str,
) -> list[LCDocument]:
    """
    Split with RecursiveCharacterTextSplitter.
    bge-m3 has its own tokenizer so we use character-based splitting
    (chunk_size=4000 chars ≈ ~1000 tokens for average English text).
    Attaches full metadata for filtered retrieval.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.RAG_CHUNK_SIZE * 4,    # chars: ~4 chars per token
        chunk_overlap=settings.RAG_CHUNK_OVERLAP * 4,
        separators=["\n\n", "\n", ". ", " ", ""],
        length_function=len,
    )

    chunks = splitter.split_documents(docs)

    for i, chunk in enumerate(chunks):
        chunk.metadata.update({
            "document_id":       document_id,
            "user_id":           user_id,
            "original_filename": original_filename,
            "s3_key":            s3_key,
            "file_extension":    file_extension,
            "chunk_index":       i,
        })

    logger.info(f"Split '{original_filename}' → {len(chunks)} chunks")
    return chunks


# ─── Step 3 & 4: Embed + store ────────────────────────────────────────────────

def embed_and_store(chunks: list[LCDocument], document_id: str) -> int:
    """Embed all chunks with bge-m3 and upsert into pgvector."""
    if not chunks:
        logger.warning(f"No chunks to embed for document {document_id}")
        return 0

    vector_store = get_vector_store()
    ids = [f"{document_id}_{i}" for i in range(len(chunks))]
    vector_store.add_documents(chunks, ids=ids)

    logger.info(f"Stored {len(chunks)} vectors for document {document_id}")
    return len(chunks)


def delete_document_vectors(document_id: str) -> None:
    """Delete all vectors belonging to a document."""
    vector_store = get_vector_store()
    vector_store.delete(filter={"document_id": document_id})
    logger.info(f"Deleted vectors for document {document_id}")


# ─── Step 5: Retriever ────────────────────────────────────────────────────────

def build_retriever(user_id: str, document_ids: Optional[list[str]] = None):
    """
    3-layer retrieval pipeline:
      Base  → pgvector MMR (diverse results, filtered by user_id)
      Wrap1 → MultiQueryRetriever (rewrites query N ways, unions results)
      Wrap2 → ContextualCompressionRetriever (strips irrelevant sentences)
    """
    vector_store = get_vector_store()
    llm = get_llm()

    # Always scope to this user's documents
    metadata_filter: dict = {"user_id": user_id}
    if document_ids:
        metadata_filter["document_id"] = {"$in": document_ids}

    # MMR: fetch 3x candidates, diversify down to k
    base_retriever = vector_store.as_retriever(
        search_type="mmr",
        search_kwargs={
            "k":           settings.RAG_RETRIEVAL_K,
            "fetch_k":     settings.RAG_RETRIEVAL_K * 3,
            "lambda_mult": 0.6,   # 0=max diversity, 1=max relevance
            "filter":      metadata_filter,
        },
    )

    # Multi-Query: Gemini rewrites the question 3 ways → union results
    multi_query_retriever = MultiQueryRetriever.from_llm(
        retriever=base_retriever,
        llm=llm,
        include_original=True,
    )

    # Contextual Compression: Gemini extracts only the relevant sentences
    compressor = LLMChainExtractor.from_llm(llm)
    compression_retriever = ContextualCompressionRetriever(
        base_compressor=compressor,
        base_retriever=multi_query_retriever,
    )

    return compression_retriever


# ─── Step 6: RAG chain ────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are an expert AI assistant for document analysis.

Answer the user's question **strictly based on the document excerpts below**.
Rules:
- Only use information present in the provided context.
- If the answer is not in the context, say: "I could not find this information in the provided documents."
- Always mention the source document filename when citing information.
- Be concise but thorough. Use bullet points for lists.
- If context contains conflicting information, highlight the conflict.

Context:
{context}"""


def _format_docs(docs: list[LCDocument]) -> str:
    parts = []
    for doc in docs:
        filename = doc.metadata.get("original_filename", "Unknown")
        page = doc.metadata.get("page", "")
        label = f"[Source: {filename}" + (f", page {page + 1}]" if page != "" else "]")
        parts.append(f"{label}\n{doc.page_content}")
    return "\n\n---\n\n".join(parts)


def build_rag_chain_with_sources(
    user_id: str,
    document_ids: Optional[list[str]] = None,
):
    """
    Full RAG chain: retrieve → compress → prompt → Gemini → parse.
    Returns a RunnableLambda that accepts:
        {"question": str, "chat_history": list[BaseMessage]}
    And outputs:
        {"answer": str, "sources": list[dict]}
    """
    retriever = build_retriever(user_id=user_id, document_ids=document_ids)
    llm = get_llm()

    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{question}"),
    ])

    def _run(inputs: dict) -> dict:
        question     = inputs["question"]
        chat_history = inputs.get("chat_history", [])

        retrieved_docs = retriever.invoke(question)
        context        = _format_docs(retrieved_docs)

        answer = (prompt | llm | StrOutputParser()).invoke({
            "question":     question,
            "context":      context,
            "chat_history": chat_history,
        })

        # Deduplicated source list
        seen, sources = set(), []
        for doc in retrieved_docs:
            filename = doc.metadata.get("original_filename", "Unknown")
            doc_id   = doc.metadata.get("document_id", "")
            key      = f"{doc_id}_{filename}"
            if key not in seen:
                seen.add(key)
                sources.append({
                    "document_id":       doc_id,
                    "original_filename": filename,
                    "page":              doc.metadata.get("page"),
                    "chunk_index":       doc.metadata.get("chunk_index"),
                })

        return {"answer": answer, "sources": sources}

    return RunnableLambda(_run)