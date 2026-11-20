"""PDF Page rendering and source highlight overlay using PyMuPDF and Pillow."""

from pathlib import Path
from typing import List, Optional, Tuple, Union
try:
    import pymupdf as fitz
except ImportError:
    import fitz
from PIL import Image, ImageDraw


def render_pdf_page(
    pdf_source: Union[bytes, Path, str],
    page_index: int,
    zoom: float = 1.4,
    block_bboxes: Optional[List[List[float]]] = None,
    show_highlights: bool = True
) -> Tuple[Image.Image, bool, str]:
    """
    Render a specific PDF page to a PIL Image at given zoom,
    optionally overlaying translucent bounding boxes over retrieved blocks.
    
    Returns: (PIL.Image, highlight_applied, status_message)
    """
    if isinstance(pdf_source, (bytes, bytearray)):
        doc = fitz.open(stream=pdf_source, filetype="pdf")
    else:
        path = Path(pdf_source)
        if not path.exists():
            raise FileNotFoundError(f"PDF file not found at {path}")
        doc = fitz.open(str(path))

    total_pages = len(doc)
    if page_index < 0 or page_index >= total_pages:
        doc.close()
        raise IndexError(f"Page index {page_index} out of range (PDF has {total_pages} pages).")

    page = doc[page_index]
    matrix = fitz.Matrix(zoom, zoom)
    
    # Render base pixmap
    pix = page.get_pixmap(matrix=matrix, alpha=False)
    base_img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)

    highlight_applied = False
    status_msg = "Plain page view"

    if show_highlights and block_bboxes and len(block_bboxes) > 0:
        rgba_img = base_img.convert("RGBA")
        overlay = Image.new("RGBA", rgba_img.size, (255, 255, 255, 0))
        draw = ImageDraw.Draw(overlay)

        valid_box_count = 0
        for bbox in block_bboxes:
            if len(bbox) >= 4:
                # PDF points transformed by zoom matrix
                # Handle possible cropbox / rotation via fitz.Rect * matrix
                r = fitz.Rect(bbox[0], bbox[1], bbox[2], bbox[3])
                tr = r * matrix

                # Normalize and clip to image dimensions
                x0 = max(0, min(pix.width, tr.x0))
                y0 = max(0, min(pix.height, tr.y0))
                x1 = max(0, min(pix.width, tr.x1))
                y1 = max(0, min(pix.height, tr.y1))

                if x1 > x0 and y1 > y0:
                    # Draw soft translucent amber/yellow rectangle
                    draw.rectangle(
                        [(x0, y0), (x1, y1)],
                        fill=(255, 235, 59, 110),
                        outline=(245, 124, 0, 200),
                        width=2
                    )
                    valid_box_count += 1

        if valid_box_count > 0:
            composited = Image.alpha_composite(rgba_img, overlay)
            base_img = composited.convert("RGB")
            highlight_applied = True
            status_msg = f"Highlighted {valid_box_count} source block(s)"
        else:
            status_msg = "Location coordinates out of visible bounds; displaying plain page"
    elif show_highlights and (not block_bboxes or len(block_bboxes) == 0):
        status_msg = "Location coordinates unavailable for this chunk; displaying plain page"

    doc.close()
    return base_img, highlight_applied, status_msg
