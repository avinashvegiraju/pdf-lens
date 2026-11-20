"""Tests for PDF ingestion, validation, and chunking."""

from pathlib import Path
import pytest
import pymupdf as fitz

from rag_pdf.ingest import extract_and_chunk_pdf, compute_sha256, sanitize_filename


@pytest.fixture
def sample_pdf_path():
    path = Path(__file__).parent.parent / "sample_data" / "demo_vehicle_guide.pdf"
    assert path.exists(), f"Sample PDF must exist at {path}"
    return path


def test_extraction_three_pages(sample_pdf_path):
    with open(sample_pdf_path, "rb") as f:
        pdf_bytes = f.read()

    meta, chunks = extract_and_chunk_pdf(pdf_bytes, "demo_vehicle_guide.pdf")

    assert meta.page_count == 3
    assert not meta.is_scanned
    assert meta.total_chunks > 0
    assert len(chunks) == meta.total_chunks

    # Verify 1-based page numbers and that page_index matches page_number - 1
    page_numbers_seen = set()
    for chunk in chunks:
        assert 1 <= chunk.page_number <= 3
        assert chunk.page_index == chunk.page_number - 1
        assert chunk.chunk_id.startswith(meta.doc_id[:12])
        assert len(chunk.text.strip()) > 0
        assert len(chunk.block_bboxes) > 0
        page_numbers_seen.add(chunk.page_number)

    # All three pages should be present in extracted chunks
    assert page_numbers_seen == {1, 2, 3}


def test_chunks_never_cross_page_boundaries(sample_pdf_path):
    with open(sample_pdf_path, "rb") as f:
        pdf_bytes = f.read()

    _, chunks = extract_and_chunk_pdf(pdf_bytes, "demo_vehicle_guide.pdf")

    doc = fitz.open(str(sample_pdf_path))
    for chunk in chunks:
        # Check text exists on its specific page and not exclusively on another page
        page_text = doc[chunk.page_index].get_text()
        first_sentence = chunk.text.split(".")[0].strip()
        assert first_sentence in page_text or first_sentence[:20] in page_text
    doc.close()


def test_scanned_pdf_detection():
    # Create an empty 2-page PDF with no text
    doc = fitz.open()
    doc.new_page(width=595, height=842)
    doc.new_page(width=595, height=842)
    pdf_bytes = doc.tobytes()
    doc.close()

    meta, chunks = extract_and_chunk_pdf(pdf_bytes, "empty_scanned.pdf")

    assert meta.is_scanned is True
    assert "image-only" in meta.error_message
    assert len(chunks) == 0


def test_sanitize_filename():
    assert sanitize_filename("../../../etc/passwd") == "passwd"
    assert sanitize_filename("my car manual (2024)!?.pdf") == "my_car_manual__2024___.pdf"
    assert sanitize_filename("normal.pdf") == "normal.pdf"
