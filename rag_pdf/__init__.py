"""PDF Lens: Local PDF RAG with Right-Side Evidence Viewer."""

from .config import config, AppConfig
from .models import PDFMetadata, ChunkMetadata, RetrievedSource, GenerationMetrics, ChatMessage
from .ingest import extract_and_chunk_pdf, compute_sha256
from .embeddings import get_embedding_model, encode_texts, encode_query
from .retrieval import LocalVectorIndex
from .generation import (
    LLMClient,
    check_llama_health,
    build_grounded_prompt,
    extract_and_validate_citations,
    get_system_memory_metrics
)
from .pdf_preview import render_pdf_page

__all__ = [
    "config",
    "AppConfig",
    "PDFMetadata",
    "ChunkMetadata",
    "RetrievedSource",
    "GenerationMetrics",
    "ChatMessage",
    "extract_and_chunk_pdf",
    "compute_sha256",
    "get_embedding_model",
    "encode_texts",
    "encode_query",
    "LocalVectorIndex",
    "LLMClient",
    "check_llama_health",
    "build_grounded_prompt",
    "extract_and_validate_citations",
    "render_pdf_page",
]
