"""Local sentence-transformers embedding generation using all-MiniLM-L6-v2."""

from typing import List, Optional, Any
import numpy as np

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    SentenceTransformer = None

from .config import config

# Singleton or cached reference
_embedding_model: Optional[Any] = None


def get_embedding_model(model_name: Optional[str] = None) -> Any:
    """Load and cache the SentenceTransformer model on CPU."""
    global _embedding_model
    if _embedding_model is not None:
        return _embedding_model

    if SentenceTransformer is None:
        raise ImportError(
            "sentence-transformers is not installed. Please install it with `pip install sentence-transformers`."
        )

    name = model_name or config.embedding_model_name
    # Keep CPU as default: lightweight 384-dim model coexists comfortably with llama-server Metal
    try:
        _embedding_model = SentenceTransformer(name, device="cpu", local_files_only=True)
    except Exception:
        _embedding_model = SentenceTransformer(name, device="cpu")
    return _embedding_model


def encode_texts(
    texts: List[str],
    model: Optional[Any] = None,
    batch_size: int = 32,
    show_progress_bar: bool = False
) -> np.ndarray:
    """
    Batch-encode texts into normalized float32 embeddings of shape (len(texts), 384).
    """
    if not texts:
        return np.empty((0, config.embedding_dim), dtype=np.float32)

    active_model = model or get_embedding_model()
    embeddings = active_model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=show_progress_bar,
        normalize_embeddings=True,
        convert_to_numpy=True
    )

    return embeddings.astype(np.float32)


def encode_query(
    query: str,
    model: Optional[Any] = None
) -> np.ndarray:
    """
    Encode a single question string into a normalized 1D float32 vector of shape (384,).
    """
    active_model = model or get_embedding_model()
    emb = active_model.encode(
        [query],
        normalize_embeddings=True,
        convert_to_numpy=True
    )
    return emb[0].astype(np.float32)
