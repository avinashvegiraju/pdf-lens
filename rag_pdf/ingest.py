"""PDF Ingestion, validation, coordinate-aware extraction, and page-bound chunking."""

import hashlib
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any

try:
    import pymupdf as fitz
except ImportError:
    try:
        import fitz
    except ImportError:
        fitz = None

from .config import config
from .models import PDFMetadata, ChunkMetadata


def compute_sha256(data: bytes) -> str:
    """Compute SHA-256 hash of bytes."""
    return hashlib.sha256(data).hexdigest()


def sanitize_filename(filename: str) -> str:
    """Sanitize user-provided filename to prevent path traversal or unsafe characters."""
    name = Path(filename).name
    # Keep alphanumeric, dot, underscore, dash
    clean = re.sub(r"[^a-zA-Z0-9_\-\.]", "_", name)
    return clean or "document.pdf"


def is_heading_block(text: str) -> bool:
    """Heuristic to detect if a block is likely a section heading."""
    stripped = text.strip()
    if not stripped:
        return False
    lines = stripped.split("\n")
    if len(lines) > 2:
        return False
    first_line = lines[0].strip()
    # Less than 65 chars, title case or all caps or starts with numbers like '1.2'
    if len(first_line) <= 65:
        if first_line.isupper() or first_line.istitle() or re.match(r"^(\d+(\.\d+)*\s+)?([A-Z])", first_line):
            return True
    return False


def split_text_into_sentences(text: str) -> List[str]:
    """Split text into sentences while preserving sentence endings."""
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return []
    # Split on sentence boundaries (.!? followed by space or end)
    raw_sentences = re.split(r"(?<=[.!?])\s+", text)
    sentences = [s.strip() for s in raw_sentences if s.strip()]
    return sentences if sentences else [text]


def extract_and_chunk_pdf(
    pdf_bytes: bytes,
    filename: str,
    target_words: int = 150,
    overlap_words: int = 25
) -> Tuple[PDFMetadata, List[ChunkMetadata]]:
    """
    Validate, extract text blocks and bounding boxes per page,
    and produce sentence-aware chunks strictly within page boundaries.
    """
    if fitz is None:
        raise ImportError("PyMuPDF (fitz) is required for PDF extraction.")

    # 1. Size check
    file_size_bytes = len(pdf_bytes)
    max_bytes = config.max_pdf_size_mb * 1024 * 1024
    if file_size_bytes > max_bytes:
        raise ValueError(
            f"PDF file size ({file_size_bytes / (1024*1024):.1f} MB) exceeds maximum allowed limit of {config.max_pdf_size_mb} MB."
        )

    doc_id = compute_sha256(pdf_bytes)
    clean_filename = sanitize_filename(filename)

    # 2. Open and validate PDF with fitz
    try:
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    except Exception as e:
        raise ValueError(f"Failed to parse PDF: {str(e)}")

    page_count = len(doc)
    if page_count == 0:
        doc.close()
        raise ValueError("PDF has 0 pages.")

    if page_count > config.max_pdf_pages:
        doc.close()
        raise ValueError(
            f"PDF page count ({page_count}) exceeds limit of {config.max_pdf_pages} pages."
        )

    # 3. Process pages
    total_extracted_text_len = 0
    chunks: List[ChunkMetadata] = []
    
    current_section = clean_filename.replace(".pdf", "").replace("_", " ").title()

    for page_idx in range(page_count):
        page = doc[page_idx]
        page_num = page_idx + 1  # 1-based page number
        
        # Get blocks: list of (x0, y0, x1, y1, text, block_no, block_type)
        blocks = page.get_text("blocks")
        
        # Filter for text blocks (type == 0)
        text_blocks: List[Dict[str, Any]] = []
        for b in blocks:
            if len(b) >= 7 and b[6] == 0:  # block_type 0 is text
                x0, y0, x1, y1, block_text, block_no = b[0], b[1], b[2], b[3], b[4], b[5]
                clean_block_text = block_text.strip()
                if clean_block_text:
                    total_extracted_text_len += len(clean_block_text)
                    is_head = is_heading_block(clean_block_text)
                    text_blocks.append({
                        "bbox": [round(float(x0), 2), round(float(y0), 2), round(float(x1), 2), round(float(y1), 2)],
                        "text": clean_block_text,
                        "is_heading": is_head,
                        "block_no": block_no
                    })

        if not text_blocks:
            continue

        # Page-level chunking
        page_chunks = _chunk_page_blocks(
            text_blocks=text_blocks,
            doc_id=doc_id,
            filename=clean_filename,
            page_number=page_num,
            page_index=page_idx,
            default_section=current_section,
            target_words=target_words,
            overlap_words=overlap_words
        )
        
        # Update current_section if page had a prominent heading
        for b in text_blocks:
            if b["is_heading"]:
                current_section = b["text"].split("\n")[0].strip()

        chunks.extend(page_chunks)

    doc.close()

    # 4. Check for scanned / image-only PDF
    is_scanned = False
    error_msg = None
    if total_extracted_text_len < 50 or len(chunks) == 0:
        is_scanned = True
        error_msg = "This PDF appears image-only; OCR is not included in this version."

    metadata = PDFMetadata(
        doc_id=doc_id,
        filename=filename,
        sanitized_filename=clean_filename,
        page_count=page_count,
        file_size_bytes=file_size_bytes,
        total_chunks=len(chunks),
        is_scanned=is_scanned,
        error_message=error_msg,
        created_at=datetime.now(timezone.utc).isoformat()
    )

    return metadata, chunks


def _chunk_page_blocks(
    text_blocks: List[Dict[str, Any]],
    doc_id: str,
    filename: str,
    page_number: int,
    page_index: int,
    default_section: str,
    target_words: int,
    overlap_words: int
) -> List[ChunkMetadata]:
    """
    Chunk text within a single page while tracking contributing block bounding boxes.
    Never crosses page boundaries!
    """
    page_chunks: List[ChunkMetadata] = []
    current_section = default_section

    # Check headings on this page
    for b in text_blocks:
        if b["is_heading"]:
            current_section = b["text"].split("\n")[0].strip()
            break

    # Build sentences with block provenance
    sentence_items: List[Dict[str, Any]] = []
    for b in text_blocks:
        if b["is_heading"]:
            current_section = b["text"].split("\n")[0].strip()
            # Include heading as an anchor
            sentence_items.append({
                "text": b["text"],
                "words": b["text"].split(),
                "bbox": b["bbox"],
                "section": current_section
            })
            continue

        sentences = split_text_into_sentences(b["text"])
        for s in sentences:
            sentence_items.append({
                "text": s,
                "words": s.split(),
                "bbox": b["bbox"],
                "section": current_section
            })

    if not sentence_items:
        return []

    # Assemble sentences into target-sized chunks
    chunk_idx_on_page = 0
    i = 0
    n = len(sentence_items)

    while i < n:
        chunk_sentences: List[str] = []
        chunk_bboxes: List[List[float]] = []
        chunk_word_count = 0
        chunk_section = sentence_items[i]["section"]
        
        j = i
        while j < n:
            item = sentence_items[j]
            item_word_count = len(item["words"])

            # If adding this item pushes us too far past target and we already have content
            if chunk_word_count > 0 and (chunk_word_count + item_word_count) > (target_words + overlap_words):
                break

            chunk_sentences.append(item["text"])
            if item["bbox"] not in chunk_bboxes:
                chunk_bboxes.append(item["bbox"])
            chunk_word_count += item_word_count
            j += 1

            if chunk_word_count >= target_words:
                break

        # If chunk is too small (e.g. single tiny sentence) and there are more items, try to include at least something
        combined_text = " ".join(chunk_sentences).strip()
        if combined_text:
            chunk_idx_on_page += 1
            chunk_id = f"{doc_id[:12]}:p{page_number:04d}:c{chunk_idx_on_page:03d}"
            
            page_chunks.append(
                ChunkMetadata(
                    chunk_id=chunk_id,
                    doc_id=doc_id,
                    filename=filename,
                    page_number=page_number,
                    page_index=page_index,
                    section_title=chunk_section,
                    text=combined_text,
                    word_count=chunk_word_count,
                    block_bboxes=chunk_bboxes,
                    chunk_index_on_page=chunk_idx_on_page,
                    approximate_highlight=(len(chunk_bboxes) > 1)
                )
            )

        # Advance index considering overlap
        if j >= n:
            break
            
        # Compute how many items to backtrack for overlap
        overlap_counted = 0
        backtrack_idx = j
        while backtrack_idx > i:
            overlap_counted += len(sentence_items[backtrack_idx - 1]["words"])
            if overlap_counted >= overlap_words:
                break
            backtrack_idx -= 1
            
        if backtrack_idx <= i or backtrack_idx == j:
            # Ensure forward progress
            i = j
        else:
            i = backtrack_idx

    return page_chunks
