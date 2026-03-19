"""
PDF Loader — multi-strategy extraction for text, tables, and images.

Extraction priority per page
─────────────────────────────
  1. pymupdf4llm  → structured Markdown (preserves tables, headings, layout)
  2. pdfplumber   → high-accuracy table extraction (cell-level)
  3. PyMuPDF/fitz → image metadata + doc metadata
  4. Tesseract OCR → automatic fallback for image-based / scanned pages
                     (only triggered when layers 1-3 yield no text for a page)
"""

import io
import os
import re
import logging
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any

import fitz
import pymupdf4llm
import pdfplumber
import pytesseract

logger = logging.getLogger(__name__)


os.environ["TESSDATA_PREFIX"] = r"C:\Program Files\Tesseract-OCR\tessdata"
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"


# ─── OCR availability check ───────────────────────────────────────────────────

def _check_ocr_available() -> bool:
    try:
        import pytesseract
        from PIL import Image   # noqa
        pytesseract.get_tesseract_version()
        langs = pytesseract.get_languages()
        if "eng" not in langs:
            logger.warning(
                "Tesseract found but 'eng' language data is missing. "
                "OCR disabled.  Fix:\n"
                "  Windows: https://github.com/UB-Mannheim/tesseract/wiki — "
                "tick 'English' during installation.\n"
                "  Linux:   sudo apt-get install tesseract-ocr-eng"
            )
            return False
        logger.info("Tesseract OCR available — will auto-fallback for image-only pages.")
        return True
    except Exception as exc:
        logger.debug(f"Tesseract not available ({exc}); OCR fallback disabled.")
        return False


_OCR_AVAILABLE: bool = _check_ocr_available()


def _ocr_page(page: fitz.Page, dpi: int = 300) -> str:
    """
    Render a fitz page to a PIL image and run Tesseract OCR on it.
    Returns extracted text, or '' on failure.
    """
    try:
        import pytesseract
        from PIL import Image

        zoom   = dpi / 72
        matrix = fitz.Matrix(zoom, zoom)
        pix    = page.get_pixmap(matrix=matrix, alpha=False)
        img    = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        text   = pytesseract.image_to_string(img, lang="eng", config="--psm 3")
        return text.strip()
    except Exception as exc:
        logger.warning(f"OCR failed on page: {exc}")
        return ""


# ─── Data Classes ─────────────────────────────────────────────────────────────

@dataclass
class ImageInfo:
    page_number: int
    image_index: int
    width:  int
    height: int
    xref:   int


@dataclass
class ParsedPage:
    page_number: int
    markdown:    str
    tables:      List[str]
    images:      List[ImageInfo]
    ocr_used:    bool = False
    word_count:  int  = 0

    def __post_init__(self):
        self.word_count = len(self.markdown.split())


@dataclass
class ParsedDocument:
    pages:          List[ParsedPage]
    full_markdown:  str
    metadata:       Dict[str, Any]
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

    @property
    def ocr_pages(self) -> int:
        return sum(1 for p in self.pages if p.ocr_used)


# ─── Table helpers ────────────────────────────────────────────────────────────

def _cell(v: Any) -> str:
    if v is None:
        return ""
    return str(v).replace("\n", " ").replace("|", "\\|").strip()


def _pdfplumber_table_to_markdown(table: List[List[Any]]) -> str:
    if not table or not table[0]:
        return ""
    rows      = [[_cell(c) for c in row] for row in table]
    col_count = max(len(r) for r in rows)
    rows      = [r + [""] * (col_count - len(r)) for r in rows]
    header    = rows[0]
    separator = ["---"] * col_count
    body      = rows[1:]
    lines = [
        "| " + " | ".join(header)    + " |",
        "| " + " | ".join(separator) + " |",
        *["| " + " | ".join(row) + " |" for row in body],
    ]
    return "\n".join(lines)


# ─── Extraction helpers ───────────────────────────────────────────────────────

def _extract_with_pymupdf4llm(doc: fitz.Document) -> Dict[int, str]:
    try:
        chunks = pymupdf4llm.to_markdown(doc, page_chunks=True)
        result: Dict[int, str] = {}
        for chunk in chunks:
            meta     = chunk.get("metadata", {})
            page_idx = meta.get("page", 0)
            result[page_idx] = chunk.get("text", "")
        return result
    except Exception as exc:
        logger.warning(f"pymupdf4llm extraction failed: {exc}")
        return {}


def _extract_tables_pdfplumber(pdf_bytes: bytes) -> Dict[int, List[str]]:
    result: Dict[int, List[str]] = {}
    try:
        with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
            for idx, page in enumerate(pdf.pages):
                tables = page.extract_tables(
                    table_settings={
                        "vertical_strategy":     "lines_strict",
                        "horizontal_strategy":   "lines_strict",
                        "snap_tolerance":         5,
                        "join_tolerance":         3,
                        "edge_min_length":       10,
                        "min_words_vertical":     3,
                        "min_words_horizontal":   1,
                        "intersection_tolerance": 3,
                    }
                )
                md_tables = [_pdfplumber_table_to_markdown(t) for t in (tables or []) if t]
                if md_tables:
                    result[idx] = [t for t in md_tables if t]
    except Exception as exc:
        logger.warning(f"pdfplumber table extraction warning: {exc}")
    return result


def _extract_images_pymupdf(doc: fitz.Document) -> Dict[int, List[ImageInfo]]:
    result: Dict[int, List[ImageInfo]] = {}
    try:
        for page_idx in range(len(doc)):
            page       = doc[page_idx]
            image_list = page.get_images(full=True)
            infos      = []
            for img_idx, img in enumerate(image_list):
                xref = img[0]
                w    = img[2] if len(img) > 2 else 0
                h    = img[3] if len(img) > 3 else 0
                if w < 50 or h < 50:
                    continue
                infos.append(ImageInfo(page_number=page_idx + 1, image_index=img_idx, width=w, height=h, xref=xref))
            if infos:
                result[page_idx] = infos
    except Exception as exc:
        logger.warning(f"PyMuPDF image extraction warning: {exc}")
    return result


def _fallback_text_extraction(doc: fitz.Document) -> Dict[int, str]:
    result = {}
    for idx in range(len(doc)):
        try:
            result[idx] = doc[idx].get_text("text")
        except Exception:
            result[idx] = ""
    return result


# ─── Public API ───────────────────────────────────────────────────────────────

def load_pdf_from_bytes(
    content:  bytes,
    filename: str = "document.pdf",
    ocr_dpi:  int = 300,
) -> ParsedDocument:
    """
    Parse a PDF from raw bytes with automatic OCR fallback.

    Layer stack per page:
      1. pymupdf4llm  — structured Markdown
      2. pdfplumber   — table extraction
      3. fitz         — image metadata
      4. Tesseract    — OCR for pages where layers 1-3 yield no text
    """
    warnings: List[str] = []

    doc: fitz.Document = fitz.open(stream=content, filetype="pdf")
    page_count = len(doc)

    doc_metadata: Dict[str, Any] = {
        "filename":   filename,
        "page_count": page_count,
        "title":      (doc.metadata or {}).get("title",  "").strip(),
        "author":     (doc.metadata or {}).get("author", "").strip(),
        "subject":    (doc.metadata or {}).get("subject","").strip(),
        "creator":    (doc.metadata or {}).get("creator","").strip(),
    }

    # Layer 1
    md_by_page = _extract_with_pymupdf4llm(doc)
    if not md_by_page:
        warnings.append("pymupdf4llm failed — using plain-text fallback.")
        md_by_page = _fallback_text_extraction(doc)

    # Layer 2
    tables_by_page = _extract_tables_pdfplumber(content)

    # Layer 3
    images_by_page = _extract_images_pymupdf(doc)

    # Layer 4 — OCR for pages with no text
    empty_pages = [i for i in range(page_count) if not md_by_page.get(i, "").strip()]

    if empty_pages and not _OCR_AVAILABLE:
        warnings.append(
            f"{len(empty_pages)} page(s) have no extractable text and Tesseract OCR is not available. "
            "Install Tesseract with English language data to enable OCR fallback:\n"
            "  Windows: https://github.com/UB-Mannheim/tesseract/wiki (tick 'English')\n"
            "  Linux:   sudo apt-get install tesseract-ocr tesseract-ocr-eng\n"
            "  Then:    pip install pytesseract Pillow"
        )

    ocr_by_page: Dict[int, str] = {}
    if empty_pages and _OCR_AVAILABLE:
        logger.info(f"[OCR] Running Tesseract on {len(empty_pages)}/{page_count} image-only pages at {ocr_dpi} DPI …")
        for page_idx in empty_pages:
            ocr_text = _ocr_page(doc[page_idx], dpi=ocr_dpi)
            if ocr_text:
                ocr_by_page[page_idx] = ocr_text
                logger.debug(f"[OCR] Page {page_idx + 1}: {len(ocr_text)} chars extracted")
            else:
                logger.debug(f"[OCR] Page {page_idx + 1}: no text recovered (diagram/blank?)")

        if ocr_by_page:
            warnings.append(f"OCR applied to pages: {', '.join(str(i+1) for i in sorted(ocr_by_page))}")

    doc.close()

    # Assemble pages
    pages:    List[ParsedPage] = []
    md_parts: List[str]        = []

    for page_idx in range(page_count):
        page_markdown = md_by_page.get(page_idx, "").strip()
        ocr_used      = False

        if not page_markdown and page_idx in ocr_by_page:
            page_markdown = ocr_by_page[page_idx]
            ocr_used      = True

        plumber_tables = tables_by_page.get(page_idx, [])
        extra_tables: List[str] = []
        for tbl in plumber_tables:
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
            ocr_used=ocr_used,
        )
        pages.append(parsed)

        if page_markdown:
            md_parts.append(f"<!-- Page {page_idx + 1} -->\n{page_markdown}")

    full_markdown = "\n\n---\n\n".join(md_parts)
    ocr_count     = sum(1 for p in pages if p.ocr_used)

    logger.info(
        f"Parsed '{filename}': {page_count} pages | "
        f"{sum(len(p.tables) for p in pages)} tables | "
        f"{sum(len(p.images) for p in pages)} images | "
        f"{ocr_count} OCR pages | "
        f"{len(full_markdown):,} total chars"
    )

    return ParsedDocument(
        pages=pages,
        full_markdown=full_markdown,
        metadata=doc_metadata,
        parse_warnings=warnings,
    )


def extract_image_bytes(content: bytes, xref: int) -> Optional[bytes]:
    """Extract raw image bytes by PyMuPDF xref — for multimodal pipelines."""
    try:
        doc        = fitz.open(stream=content, filetype="pdf")
        base_image = doc.extract_image(xref)
        doc.close()
        return base_image["image"] if base_image else None
    except Exception as exc:
        logger.warning(f"Image extraction failed for xref={xref}: {exc}")
        return None