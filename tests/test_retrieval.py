"""Tests for embedding generation, vector caching, and cosine retrieval."""

from pathlib import Path
import numpy as np
import pytest

from rag_pdf.ingest import extract_and_chunk_pdf
from rag_pdf.models import ChunkMetadata, PDFMetadata
from rag_pdf.retrieval import LocalVectorIndex
from rag_pdf.embeddings import encode_texts, encode_query, get_embedding_model


@pytest.fixture
def temp_index_dir(tmp_path):
    idx_dir = tmp_path / "test_index"
    idx_dir.mkdir(parents=True, exist_ok=True)
    return idx_dir


def test_deterministic_cosine_ranking(temp_index_dir):
    """Test retrieval ranking logic using mathematically controlled vectors."""
    index = LocalVectorIndex(index_dir=temp_index_dir)

    # 3 mock chunks
    chunks = [
        ChunkMetadata(
            chunk_id="test:p0001:c001",
            doc_id="test_doc",
            filename="test.pdf",
            page_number=1,
            page_index=0,
            section_title="Engine",
            text="Engine temperature overheating warning.",
            word_count=4,
            block_bboxes=[[10, 10, 100, 50]],
            chunk_index_on_page=1
        ),
        ChunkMetadata(
            chunk_id="test:p0002:c001",
            doc_id="test_doc",
            filename="test.pdf",
            page_number=2,
            page_index=1,
            section_title="Tires",
            text="Cold tire pressure check.",
            word_count=4,
            block_bboxes=[[10, 10, 100, 50]],
            chunk_index_on_page=1
        ),
        ChunkMetadata(
            chunk_id="test:p0003:c001",
            doc_id="test_doc",
            filename="test.pdf",
            page_number=3,
            page_index=2,
            section_title="Battery",
            text="Battery alternator failure.",
            word_count=4,
            block_bboxes=[[10, 10, 100, 50]],
            chunk_index_on_page=1
        )
    ]

    meta = PDFMetadata(
        doc_id="test_doc",
        filename="test.pdf",
        sanitized_filename="test.pdf",
        page_count=3,
        file_size_bytes=1024,
        total_chunks=3
    )

    # Controlled unit vectors in 384 dimensions
    vecs = np.zeros((3, 384), dtype=np.float32)
    vecs[0, 0] = 1.0  # Ch 1 aligned with dim 0
    vecs[1, 1] = 1.0  # Ch 2 aligned with dim 1
    vecs[2, 2] = 1.0  # Ch 3 aligned with dim 2

    index.save_index(meta, chunks, vecs)
    assert index.is_indexed("test_doc")

    # Mock embedder that returns a vector aligned with dim 0
    class MockEmbedder:
        def encode(self, texts, **kwargs):
            arr = np.zeros((len(texts), 384), dtype=np.float32)
            arr[:, 0] = 1.0
            return arr

    results = index.search("engine overheat", doc_ids=["test_doc"], top_k=2, embedding_model=MockEmbedder())

    assert len(results) == 2
    assert results[0].citation_id == "[S1]"
    assert results[0].page_number == 1
    assert results[0].score == pytest.approx(1.0, abs=1e-4)

    assert results[1].citation_id == "[S2]"
    assert results[1].score == pytest.approx(0.0, abs=1e-4)


def test_index_persistence_and_cache_reuse(temp_index_dir):
    """Test saving to disk and reusing index without recomputation."""
    index = LocalVectorIndex(index_dir=temp_index_dir)

    meta = PDFMetadata(
        doc_id="doc_cache_test",
        filename="cached.pdf",
        sanitized_filename="cached.pdf",
        page_count=1,
        file_size_bytes=500,
        total_chunks=1
    )
    chunks = [
        ChunkMetadata(
            chunk_id="doc_cache_test:p0001:c001",
            doc_id="doc_cache_test",
            filename="cached.pdf",
            page_number=1,
            page_index=0,
            section_title="Test",
            text="Cached chunk text.",
            word_count=3,
            block_bboxes=[[0, 0, 10, 10]],
            chunk_index_on_page=1
        )
    ]
    orig_vecs = np.ones((1, 384), dtype=np.float32) / np.sqrt(384)

    # Save initial
    index.save_index(meta, chunks, orig_vecs)

    # Track if encode is called
    called = []
    class GuardEmbedder:
        def encode(self, *args, **kwargs):
            called.append(True)
            return orig_vecs

    # Should load from cache
    loaded_meta, loaded_chunks, loaded_vecs = index.index_document(
        meta, chunks, force_reindex=False, embedding_model=GuardEmbedder()
    )

    assert len(called) == 0, "Embedding model should not be called when index is already cached"
    assert loaded_meta.doc_id == "doc_cache_test"
    assert np.allclose(loaded_vecs, orig_vecs)


def test_real_semantic_retrieval_on_sample_pdf(temp_index_dir):
    """Integration test using actual MiniLM model on the synthetic vehicle guide."""
    sample_pdf_path = Path(__file__).parent.parent / "sample_data" / "demo_vehicle_guide.pdf"
    with open(sample_pdf_path, "rb") as f:
        pdf_bytes = f.read()

    meta, chunks = extract_and_chunk_pdf(pdf_bytes, "demo_vehicle_guide.pdf")

    index = LocalVectorIndex(index_dir=temp_index_dir)
    embedder = get_embedding_model()
    index.index_document(meta, chunks, embedding_model=embedder)

    # Question about engine temperature
    query = "What should I do when the engine temperature warning turns red?"
    results = index.search(query, doc_ids=[meta.doc_id], top_k=3, embedding_model=embedder)

    assert len(results) >= 1
    # Top result MUST be from Page 1 (where overheating and red warning are described)
    top_result = results[0]
    assert top_result.page_number == 1
    assert "temperature" in top_result.text.lower() or "overheating" in top_result.text.lower()
    assert top_result.score > 0.4


def test_bm25_sparse_and_hybrid_search(temp_index_dir):
    """Test BM25 keyword matching and hybrid fusion for specific terminology."""
    sample_pdf_path = Path(__file__).parent.parent / "sample_data" / "demo_vehicle_guide.pdf"
    with open(sample_pdf_path, "rb") as f:
        pdf_bytes = f.read()

    meta, chunks = extract_and_chunk_pdf(pdf_bytes, "demo_vehicle_guide.pdf")

    index = LocalVectorIndex(index_dir=temp_index_dir)
    embedder = get_embedding_model()
    index.index_document(meta, chunks, embedding_model=embedder)

    # Specific keyword test: "TPMS" and "33 psi" on Page 2
    query = "TPMS 33 psi"
    results_bm25 = index.search(query, doc_ids=[meta.doc_id], top_k=2, embedding_model=embedder, mode="sparse")
    assert len(results_bm25) >= 1
    assert results_bm25[0].page_number == 2
    assert "33 psi" in results_bm25[0].text or "tpms" in results_bm25[0].text.lower()

    # Hybrid search
    results_hybrid = index.search(query, doc_ids=[meta.doc_id], top_k=2, embedding_model=embedder, mode="hybrid")
    assert len(results_hybrid) >= 1
    assert results_hybrid[0].page_number == 2
