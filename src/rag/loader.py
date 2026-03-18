"""
PDF Loader — multi-strategy extraction for text, tables, and images.

Priority stack:
  1. pymupdf4llm  → converts entire PDF to structured Markdown (preserves tables, headings)
  2. pdfplumber   → enhanced table extraction with cell-level accuracy
  3. pymupdf/fitz → image inventory + fallback plain text extraction

The result is a ParsedDocument with per-page structured data and a single
full-markdown string ready for chunking.
"""

import io
import re
import logging
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any

import fitz                # PyMuPDF — image extraction + fallback
import pymupdf4llm         # Markdown conversion (wraps PyMuPDF)
import pdfplumber          # Precise table extraction

logger = logging.getLogger(__name__)


# ─── Data Classes ─────────────────────────────────────────────────────────────

@dataclass
class ImageInfo:
    """Metadata for an image found on a PDF page (not the raw bytes)."""
    page_number: int
    image_index: int
    width:  int
    height: int
    xref:   int   # PyMuPDF internal reference — use to extract bytes if needed


@dataclass
class ParsedPage:
    page_number: int          # 1-indexed
    markdown:    str          # Full page content as markdown (text + inline tables)
    tables:      List[str]    # Additional markdown tables from pdfplumber (if different)
    images:      List[ImageInfo]
    word_count:  int = 0

    def __post_init__(self):
        self.word_count = len(self.markdown.split())


@dataclass
class ParsedDocument:
    pages:         List[ParsedPage]
    full_markdown: str          # Entire document joined — primary input to chunkers
    metadata:      Dict[str, Any]
    parse_warnings: List[str] = field(default_factory=list)

    @property
    def total_words(self) -> int:
        return sum(p.word_count for p in self.pages)

    @property
    def has_tables(self) -> bool:
        return any(p.tables for p in self.pages)

    @property
    def has_images(self) -> bool:
        return any(p.images for p in self.pages)


# ─── Table helpers ────────────────────────────────────────────────────────────

def _cell(v: Any) -> str:
    """Normalise a pdfplumber cell value to a clean string."""
    if v is None:
        return ""
    return str(v).replace("\n", " ").replace("|", "\\|").strip()


def _pdfplumber_table_to_markdown(table: List[List[Any]]) -> str:
    """
    Convert a pdfplumber table (list-of-lists) to a GitHub-flavoured
    Markdown table string.
    """
    if not table or not table[0]:
        return ""

    rows = [[_cell(c) for c in row] for row in table]
    col_count = max(len(r) for r in rows)

    # Pad all rows to same width
    rows = [r + [""] * (col_count - len(r)) for r in rows]

    header    = rows[0]
    separator = ["---"] * col_count
    body      = rows[1:]

    lines = [
        "| " + " | ".join(header)    + " |",
        "| " + " | ".join(separator) + " |",
        *["| " + " | ".join(row) + " |" for row in body],
    ]
    return "\n".join(lines)


# ─── Core extraction ──────────────────────────────────────────────────────────

def _extract_with_pymupdf4llm(
    doc: fitz.Document,
) -> Dict[int, str]:
    """
    Use pymupdf4llm to extract per-page Markdown.
    Returns {0-indexed page number: markdown string}.
    """
    try:
        # page_chunks=True → returns a list of dicts, one per page
        page_chunks = pymupdf4llm.to_markdown(doc, page_chunks=True)
        result: Dict[int, str] = {}
        for chunk in page_chunks:
            meta     = chunk.get("metadata", {})
            # pymupdf4llm uses 0-based "page" in metadata
            page_idx = meta.get("page", 0)
            result[page_idx] = chunk.get("text", "")
        return result
    except Exception as exc:
        logger.warning(f"pymupdf4llm extraction failed: {exc}")
        return {}


def _extract_tables_pdfplumber(
    pdf_bytes: bytes,
) -> Dict[int, List[str]]:
    """
    Extract tables from every page using pdfplumber.
    Returns {0-indexed page number: [markdown table, ...]}.
    """
    result: Dict[int, List[str]] = {}
    try:
        with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
            for idx, page in enumerate(pdf.pages):
                tables = page.extract_tables(
                    table_settings={
                        "vertical_strategy":   "lines_strict",
                        "horizontal_strategy": "lines_strict",
                        "snap_tolerance":       5,
                        "join_tolerance":       3,
                        "edge_min_length":     10,
                        "min_words_vertical":   3,
                        "min_words_horizontal": 1,
                        "intersection_tolerance": 3,
                    }
                )
                md_tables = [
                    _pdfplumber_table_to_markdown(t)
                    for t in (tables or [])
                    if t
                ]
                if md_tables:
                    result[idx] = [t for t in md_tables if t]
    except Exception as exc:
        logger.warning(f"pdfplumber table extraction warning: {exc}")
    return result


def _extract_images_pymupdf(
    doc: fitz.Document,
) -> Dict[int, List[ImageInfo]]:
    """
    Collect image metadata (not raw bytes) using PyMuPDF.
    Returns {0-indexed page number: [ImageInfo, ...]}.
    """
    result: Dict[int, List[ImageInfo]] = {}
    try:
        for page_idx in range(len(doc)):
            page       = doc[page_idx]
            image_list = page.get_images(full=True)
            infos      = []
            for img_idx, img in enumerate(image_list):
                xref = img[0]
                # img tuple: (xref, smask, width, height, bpc, colorspace, ...)
                w = img[2] if len(img) > 2 else 0
                h = img[3] if len(img) > 3 else 0
                # Skip tiny images (likely decorative lines / bullets)
                if w < 50 or h < 50:
                    continue
                infos.append(ImageInfo(
                    page_number=page_idx + 1,
                    image_index=img_idx,
                    width=w,
                    height=h,
                    xref=xref,
                ))
            if infos:
                result[page_idx] = infos
    except Exception as exc:
        logger.warning(f"PyMuPDF image extraction warning: {exc}")
    return result


def _fallback_text_extraction(doc: fitz.Document) -> Dict[int, str]:
    """Plain text extraction when pymupdf4llm is unavailable."""
    result = {}
    for idx in range(len(doc)):
        try:
            result[idx] = doc[idx].get_text("text")
        except Exception:
            result[idx] = ""
    return result


# ─── Public API ───────────────────────────────────────────────────────────────

def load_pdf_from_bytes(
    content: bytes,
    filename: str = "document.pdf",
) -> ParsedDocument:
    """
    Parse a PDF from raw bytes.

    Extraction layers (applied in order, results merged):
    ┌──────────────────────────────────────────────────────────────┐
    │  pymupdf4llm  →  structured Markdown per page               │
    │  pdfplumber   →  high-accuracy table Markdown per page      │
    │  PyMuPDF/fitz →  image metadata + doc meta + fallback text  │
    └──────────────────────────────────────────────────────────────┘

    Args:
        content:  Raw PDF bytes (e.g., downloaded from S3).
        filename: Original filename — stored in metadata only.

    Returns:
        ParsedDocument with per-page data and a full-document markdown string.
    """
    warnings: List[str] = []

    # ── Open once with PyMuPDF for meta + images ──────────────────
    doc: fitz.Document = fitz.open(stream=content, filetype="pdf")
    page_count = len(doc)

    doc_metadata: Dict[str, Any] = {
        "filename":    filename,
        "page_count":  page_count,
        "title":       (doc.metadata or {}).get("title",  "").strip(),
        "author":      (doc.metadata or {}).get("author", "").strip(),
        "subject":     (doc.metadata or {}).get("subject","").strip(),
        "creator":     (doc.metadata or {}).get("creator","").strip(),
    }

    # ── Layer 1: pymupdf4llm Markdown ────────────────────────────
    md_by_page = _extract_with_pymupdf4llm(doc)
    if not md_by_page:
        warnings.append("pymupdf4llm failed; using plain-text fallback.")
        md_by_page = _fallback_text_extraction(doc)

    # ── Layer 2: pdfplumber tables ────────────────────────────────
    tables_by_page = _extract_tables_pdfplumber(content)

    # ── Layer 3: image metadata ────────────────────────────────────
    images_by_page = _extract_images_pymupdf(doc)

    doc.close()

    # ── Assemble ParsedPages ──────────────────────────────────────
    pages:     List[ParsedPage] = []
    md_parts:  List[str]        = []

    for page_idx in range(page_count):
        page_markdown = md_by_page.get(page_idx, "").strip()
        plumber_tables = tables_by_page.get(page_idx, [])

        # Inject pdfplumber tables if pymupdf4llm missed them.
        # Heuristic: if a plumber table's content does NOT appear verbatim
        # in the page markdown, append it.
        extra_tables: List[str] = []
        for tbl in plumber_tables:
            # Use first cell of table as a rough check
            first_cell_match = re.search(r'\|\s*([^|]{3,}?)\s*\|', tbl)
            if first_cell_match:
                probe = first_cell_match.group(1).strip()
                if probe and probe not in page_markdown:
                    extra_tables.append(tbl)

        if extra_tables:
            page_markdown += "\n\n" + "\n\n".join(extra_tables)

        images = images_by_page.get(page_idx, [])

        parsed = ParsedPage(
            page_number=page_idx + 1,
            markdown=page_markdown,
            tables=plumber_tables,
            images=images,
        )
        pages.append(parsed)

        # Build full-document markdown with page markers
        if page_markdown:
            md_parts.append(f"<!-- Page {page_idx + 1} -->\n{page_markdown}")

    full_markdown = "\n\n---\n\n".join(md_parts)

    logger.info(
        f"Parsed PDF '{filename}': {page_count} pages | "
        f"{sum(len(p.tables) for p in pages)} tables | "
        f"{sum(len(p.images) for p in pages)} images"
    )

    return ParsedDocument(
        pages=pages,
        full_markdown=full_markdown,
        metadata=doc_metadata,
        parse_warnings=warnings,
    )


def extract_image_bytes(content: bytes, xref: int) -> Optional[bytes]:
    """
    Extract raw image bytes from a PDF by xref.
    Useful for multimodal pipelines (e.g., sending images to a vision model).

    Args:
        content: Raw PDF bytes.
        xref:    PyMuPDF xref from ImageInfo.

    Returns:
        Raw image bytes or None on failure.
    """
    try:
        doc = fitz.open(stream=content, filetype="pdf")
        base_image = doc.extract_image(xref)
        doc.close()
        return base_image["image"] if base_image else None
    except Exception as exc:
        logger.warning(f"Image extraction failed for xref={xref}: {exc}")
        return None