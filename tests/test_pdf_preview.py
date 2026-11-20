"""Tests for PDF page rendering and coordinate-aware highlight overlay."""

from pathlib import Path
import pytest
from PIL import Image

from rag_pdf.pdf_preview import render_pdf_page


@pytest.fixture
def sample_pdf_path():
    path = Path(__file__).parent.parent / "sample_data" / "demo_vehicle_guide.pdf"
    assert path.exists()
    return path


def test_render_pdf_page_plain(sample_pdf_path):
    img, applied, status_msg = render_pdf_page(
        pdf_source=sample_pdf_path,
        page_index=0,
        zoom=1.2,
        show_highlights=False
    )

    assert isinstance(img, Image.Image)
    assert img.width > 200
    assert img.height > 200
    assert applied is False
    assert "Plain" in status_msg


def test_render_pdf_page_with_highlights(sample_pdf_path):
    # Sample bounding box for page 1 content
    bboxes = [[50.0, 100.0, 500.0, 300.0]]

    img, applied, status_msg = render_pdf_page(
        pdf_source=sample_pdf_path,
        page_index=0,
        zoom=1.4,
        block_bboxes=bboxes,
        show_highlights=True
    )

    assert isinstance(img, Image.Image)
    assert applied is True
    assert "Highlighted 1 source block(s)" in status_msg


def test_render_pdf_page_missing_or_out_of_bounds_bboxes(sample_pdf_path):
    # Out-of-bounds coordinates (e.g. inverted or outside page)
    bboxes = [[9999.0, 9999.0, 9999.0, 9999.0]]

    img, applied, status_msg = render_pdf_page(
        pdf_source=sample_pdf_path,
        page_index=0,
        zoom=1.0,
        block_bboxes=bboxes,
        show_highlights=True
    )

    assert isinstance(img, Image.Image)
    assert applied is False
    assert "out of visible bounds" in status_msg


def test_render_invalid_page_index(sample_pdf_path):
    with pytest.raises(IndexError):
        render_pdf_page(
            pdf_source=sample_pdf_path,
            page_index=99,  # Guide only has 3 pages
            zoom=1.0
        )
