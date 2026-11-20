"""Configuration and defaults for PDF Lens RAG application."""

import os
from dataclasses import dataclass
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables if .env is present
load_dotenv()

@dataclass(frozen=True)
class AppConfig:
    # LLM server settings
    llama_base_url: str = os.getenv("LLAMA_BASE_URL", "http://127.0.0.1:8081/v1")
    llama_health_url: str = os.getenv("LLAMA_HEALTH_URL", "http://127.0.0.1:8081/health")
    llama_model: str = os.getenv("LLAMA_MODEL", "lfm3-rag")
    
    # Embedding model settings
    embedding_model_name: str = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
    embedding_dim: int = 384
    
    # Retrieval settings
    default_top_k: int = int(os.getenv("DEFAULT_TOP_K", "4"))
    min_top_k: int = 2
    max_top_k: int = 8
    
    # Context and token budget
    llm_context_tokens: int = int(os.getenv("LLM_CONTEXT_TOKENS", "8192"))
    max_output_tokens: int = int(os.getenv("MAX_OUTPUT_TOKENS", "512"))
    max_context_budget_tokens: int = int(os.getenv("MAX_CONTEXT_BUDGET_TOKENS", "3000"))
    
    # Chunking settings (sentence/paragraph-aware, 130-180 words, 20-35 words overlap)
    chunk_target_words: int = int(os.getenv("CHUNK_TARGET_WORDS", "150"))
    chunk_overlap_words: int = int(os.getenv("CHUNK_OVERLAP_WORDS", "25"))
    
    # Document limits
    max_pdf_size_mb: int = int(os.getenv("MAX_PDF_SIZE_MB", "100"))
    max_pdf_pages: int = int(os.getenv("MAX_PDF_PAGES", "1000"))
    
    # Storage directories
    base_dir: Path = Path(os.getenv("DATA_DIR", "./data")).resolve()
    
    @property
    def uploads_dir(self) -> Path:
        return self.base_dir / "uploads"
        
    @property
    def index_dir(self) -> Path:
        return self.base_dir / "index"
        
    @property
    def cache_dir(self) -> Path:
        return self.base_dir / "cache"

    def ensure_directories(self) -> None:
        """Ensure all required data directories exist."""
        self.uploads_dir.mkdir(parents=True, exist_ok=True)
        self.index_dir.mkdir(parents=True, exist_ok=True)
        self.cache_dir.mkdir(parents=True, exist_ok=True)


config = AppConfig()
