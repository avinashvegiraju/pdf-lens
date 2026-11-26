"""Local vector index and cosine similarity retrieval using NumPy."""

import json
import logging
import re
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Any
import numpy as np

try:
    from rank_bm25 import BM25Okapi
except ImportError:
    BM25Okapi = None

from .config import config
from .models import PDFMetadata, ChunkMetadata, RetrievedSource
from .embeddings import encode_texts, encode_query

logger = logging.getLogger(__name__)


def tokenize_for_bm25(text: str) -> List[str]:
    """Simple alphanumeric word tokenizer for BM25 sparse retrieval."""
    return re.findall(r"\w+", text.lower())


class LocalVectorIndex:
    """Manages document chunk indices and cosine-similarity retrieval on disk."""

    def __init__(self, index_dir: Optional[Path] = None):
        self.index_dir = index_dir or config.index_dir
        self.index_dir.mkdir(parents=True, exist_ok=True)

    def _meta_path(self, doc_id: str) -> Path:
        return self.index_dir / f"{doc_id}.json"

    def _vectors_path(self, doc_id: str) -> Path:
        return self.index_dir / f"{doc_id}.npy"

    def is_indexed(self, doc_id: str) -> bool:
        """Check if index files exist and are valid for a given doc_id."""
        meta_p = self._meta_path(doc_id)
        vec_p = self._vectors_path(doc_id)
        return meta_p.exists() and vec_p.exists()

    def save_index(
        self,
        metadata: PDFMetadata,
        chunks: List[ChunkMetadata],
        embeddings: np.ndarray
    ) -> None:
        """Save metadata JSON and embeddings .npy to disk."""
        meta_p = self._meta_path(metadata.doc_id)
        vec_p = self._vectors_path(metadata.doc_id)

        meta_data = {
            "pdf_metadata": metadata.to_dict(),
            "chunks": [c.to_dict() for c in chunks]
        }

        # Write metadata JSON
        with open(meta_p, "w", encoding="utf-8") as f:
            json.dump(meta_data, f, indent=2)

        # Write NumPy array
        np.save(vec_p, embeddings.astype(np.float32))

    def load_index(self, doc_id: str) -> Tuple[PDFMetadata, List[ChunkMetadata], np.ndarray]:
        """Load index from disk for a given doc_id."""
        meta_p = self._meta_path(doc_id)
        vec_p = self._vectors_path(doc_id)

        if not meta_p.exists() or not vec_p.exists():
            raise FileNotFoundError(f"Index for document {doc_id} not found.")

        with open(meta_p, "r", encoding="utf-8") as f:
            meta_data = json.load(f)

        pdf_meta = PDFMetadata.from_dict(meta_data["pdf_metadata"])
        chunks = [ChunkMetadata.from_dict(c) for c in meta_data["chunks"]]
        embeddings = np.load(vec_p).astype(np.float32)

        if len(chunks) != len(embeddings):
            raise ValueError(
                f"Index mismatch for {doc_id}: {len(chunks)} chunks vs {len(embeddings)} vectors."
            )

        return pdf_meta, chunks, embeddings

    def delete_index(self, doc_id: str) -> bool:
        """Delete index files for a document."""
        deleted = False
        meta_p = self._meta_path(doc_id)
        vec_p = self._vectors_path(doc_id)
        if meta_p.exists():
            meta_p.unlink()
            deleted = True
        if vec_p.exists():
            vec_p.unlink()
            deleted = True
        return deleted

    def list_indexed_documents(self) -> List[PDFMetadata]:
        """List all indexed documents available on disk."""
        documents: List[PDFMetadata] = []
        for meta_p in self.index_dir.glob("*.json"):
            try:
                with open(meta_p, "r", encoding="utf-8") as f:
                    meta_data = json.load(f)
                pdf_meta = PDFMetadata.from_dict(meta_data["pdf_metadata"])
                vec_p = self._vectors_path(pdf_meta.doc_id)
                if vec_p.exists():
                    documents.append(pdf_meta)
            except Exception as e:
                logger.warning(f"Error loading metadata from {meta_p}: {e}")
        return documents

    def index_document(
        self,
        metadata: PDFMetadata,
        chunks: List[ChunkMetadata],
        force_reindex: bool = False,
        embedding_model: Optional[Any] = None
    ) -> Tuple[PDFMetadata, List[ChunkMetadata], np.ndarray]:
        """
        Check cache; if already indexed and not force_reindex, load from disk.
        Otherwise generate embeddings and persist.
        """
        if not force_reindex and self.is_indexed(metadata.doc_id):
            try:
                return self.load_index(metadata.doc_id)
            except Exception as e:
                logger.warning(f"Failed to load cached index for {metadata.doc_id}, re-indexing: {e}")

        # If document has no chunks (e.g. image-only PDF)
        if not chunks:
            empty_vecs = np.empty((0, config.embedding_dim), dtype=np.float32)
            self.save_index(metadata, chunks, empty_vecs)
            return metadata, chunks, empty_vecs

        # Batch encode
        texts = [c.text for c in chunks]
        embeddings = encode_texts(texts, model=embedding_model)

        self.save_index(metadata, chunks, embeddings)
        return metadata, chunks, embeddings

    def search(
        self,
        query: str,
        doc_ids: Optional[List[str]] = None,
        top_k: int = 4,
        embedding_model: Optional[Any] = None,
        mode: str = "hybrid",
        dense_weight: float = 0.65
    ) -> List[RetrievedSource]:
        """
        Perform retrieval over specified documents.
        Supports:
          - 'hybrid': Fuses dense cosine similarity + BM25 sparse keyword ranking.
          - 'dense': Pure NumPy cosine dot-product (v1 behavior).
          - 'sparse': Pure BM25 keyword matching (optimal for error codes, part numbers).
        Returns top_k RetrievedSource objects with assigned citations [S1]..[SK].
        """
        if not query.strip():
            return []

        # 1. Determine targets
        if doc_ids is None or len(doc_ids) == 0:
            target_docs = [doc.doc_id for doc in self.list_indexed_documents()]
        else:
            target_docs = doc_ids

        all_chunks: List[ChunkMetadata] = []
        all_embeddings: List[np.ndarray] = []

        for did in target_docs:
            if self.is_indexed(did):
                try:
                    _, chunks, vecs = self.load_index(did)
                    if len(chunks) > 0 and len(vecs) > 0:
                        all_chunks.extend(chunks)
                        all_embeddings.append(vecs)
                except Exception as e:
                    logger.error(f"Error loading index for search {did}: {e}")

        if not all_chunks or not all_embeddings:
            return []

        num_chunks = len(all_chunks)

        # 2. Compute Dense Scores
        if mode in ("dense", "hybrid") or BM25Okapi is None:
            corpus_matrix = np.vstack(all_embeddings)  # Shape: (Total_Chunks, 384)
            q_vec = encode_query(query, model=embedding_model)  # Shape: (384,)
            dense_scores = np.dot(corpus_matrix, q_vec)
        else:
            dense_scores = np.zeros(num_chunks, dtype=np.float32)

        # 3. Compute Sparse BM25 Scores if available and requested
        if mode in ("sparse", "hybrid") and BM25Okapi is not None:
            tokenized_corpus = [tokenize_for_bm25(c.text) for c in all_chunks]
            q_tokens = tokenize_for_bm25(query)
            if q_tokens:
                bm25 = BM25Okapi(tokenized_corpus)
                raw_bm25 = np.array(bm25.get_scores(q_tokens), dtype=np.float32)
                # Min-Max normalize BM25 scores to [0, 1] range
                bm25_max = np.max(raw_bm25)
                bm25_min = np.min(raw_bm25)
                if bm25_max > bm25_min:
                    norm_bm25 = (raw_bm25 - bm25_min) / (bm25_max - bm25_min)
                elif bm25_max > 0:
                    norm_bm25 = raw_bm25 / bm25_max
                else:
                    norm_bm25 = raw_bm25
            else:
                norm_bm25 = np.zeros(num_chunks, dtype=np.float32)
        else:
            norm_bm25 = np.zeros(num_chunks, dtype=np.float32)

        # 4. Fusion of Scores
        if mode == "hybrid" and BM25Okapi is not None:
            # If dense scores are zero (or pure BM25), adjust weight
            scores = (dense_weight * dense_scores) + ((1.0 - dense_weight) * norm_bm25)
        elif mode == "sparse" and BM25Okapi is not None:
            scores = norm_bm25
        else:
            scores = dense_scores

        # 5. Sort descending
        ranked_indices = np.argsort(scores)[::-1]

        # 6. Deduplicate / diversity filter
        selected_sources: List[RetrievedSource] = []
        seen_texts: List[str] = []

        for idx in ranked_indices:
            if len(selected_sources) >= top_k:
                break

            chunk = all_chunks[idx]
            score = float(scores[idx])

            # Deduplication: skip if chunk text is identical or substring
            clean_chunk_text = " ".join(chunk.text.split())
            is_dup = False
            for seen in seen_texts:
                if clean_chunk_text in seen or seen in clean_chunk_text:
                    is_dup = True
                    break
            if is_dup and len(selected_sources) > 0:
                continue

            citation_id = f"[S{len(selected_sources) + 1}]"
            retrieved_src = RetrievedSource(
                citation_id=citation_id,
                chunk_id=chunk.chunk_id,
                doc_id=chunk.doc_id,
                filename=chunk.filename,
                page_number=chunk.page_number,
                page_index=chunk.page_index,
                section_title=chunk.section_title,
                text=chunk.text,
                score=round(score, 4),
                block_bboxes=chunk.block_bboxes,
                approximate_highlight=chunk.approximate_highlight
            )
            selected_sources.append(retrieved_src)
            seen_texts.append(clean_chunk_text)

        return selected_sources
