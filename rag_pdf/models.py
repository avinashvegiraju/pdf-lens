"""Data models for PDF Lens."""

from dataclasses import dataclass, field, asdict
from typing import List, Optional, Dict, Any

@dataclass
class PDFMetadata:
    doc_id: str                      # SHA-256 hex digest of file content
    filename: str                    # Original filename
    sanitized_filename: str          # Safe storage filename
    page_count: int
    file_size_bytes: int
    total_chunks: int
    is_scanned: bool = False
    error_message: Optional[str] = None
    created_at: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PDFMetadata":
        return cls(**data)


@dataclass
class ChunkMetadata:
    chunk_id: str                    # E.g. <doc_id[:8]>:p0001:c001
    doc_id: str
    filename: str
    page_number: int                 # 1-based page number for display
    page_index: int                  # 0-based page index for PyMuPDF
    section_title: Optional[str]
    text: str
    word_count: int
    block_bboxes: List[List[float]]  # List of [x0, y0, x1, y1] bounding boxes in PDF points
    chunk_index_on_page: int
    approximate_highlight: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ChunkMetadata":
        return cls(**data)


@dataclass
class RetrievedSource:
    citation_id: str                 # E.g. "[S1]"
    chunk_id: str
    doc_id: str
    filename: str
    page_number: int                 # 1-based
    page_index: int                  # 0-based
    section_title: Optional[str]
    text: str
    score: float                     # Cosine similarity (uncalibrated similarity score)
    block_bboxes: List[List[float]]
    approximate_highlight: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "RetrievedSource":
        return cls(**data)


@dataclass
class GenerationMetrics:
    retrieval_ms: float = 0.0
    first_token_ms: float = 0.0
    total_generation_ms: float = 0.0
    output_tokens: int = 0
    prompt_tokens: int = 0


@dataclass
class ChatMessage:
    role: str                        # "user" or "assistant"
    content: str
    sources: List[RetrievedSource] = field(default_factory=list)
    metrics: Optional[GenerationMetrics] = None
    unverified_citations: List[str] = field(default_factory=list)
