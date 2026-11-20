# 🔍 PDF Lens — Local PDF RAG with Right-Side Evidence Viewer

**PDF Lens** is a lightweight, fully private retrieval-augmented generation (RAG) system built for Apple Silicon Macs. It allows users to query PDF documents and immediately view the exact supporting source passages—complete with translucent highlight overlays directly on the rendered original PDF page—before and while the local **LiquidAI LFM3-1.7B** language model streams grounded answers with verified citations (`[S1]`, `[S2]`).

---

## 🏗️ Architecture & Three-Column UI Layout

```text
+-------------------+--------------------------------+-------------------------------------+
| LEFT / LIBRARY    | CENTER / CHAT                  | RIGHT / EVIDENCE                    |
| ~22%              | ~43%                           | ~35%                                |
+-------------------+--------------------------------+-------------------------------------+
| PDF upload        | PDF Lens                       | Retrieved evidence                  |
| Sample PDF        | LFM3-1.7B • Local llama.cpp     |                                     |
| Document selector |                                | [S1] Manual.pdf · Page 12           |
| PDF statistics    | User: Why is warning red?      | Similarity: 0.76 (illustrative)      |
|                  |                                | 'Stop the car safely ...'            |
| Server status     | LFM3: The manual instructs ... | [View on original page]             |
| Top-K selector    | ... [S1]                       |                                     |
| Search controls   |                                | [S2] Manual.pdf · Page 13           |
| Clear chat        | [Ask about your PDF ...]       | 'Allow engine to cool ...'           |
|                  |                                |                                     |
|                  |                                | Original PDF page preview           |
|                  |                                | [highlighted source paragraph]       |
+-------------------+--------------------------------+-------------------------------------+
```

### Key Behavioral Innovations
- **Pre-Generation Evidence Display:** The moment a user submits a question, NumPy executes cosine-similarity vector retrieval and updates the right column with top matching source cards *before* model generation begins.
- **One-Click Page & Region Highlighting:** Clicking **View Page** on any citation card renders the original physical PDF page with PyMuPDF, overlaying translucent amber bounding boxes over the exact contributing text blocks.
- **Strict Page-Boundary Chunking:** Chunks (~130–180 words) never cross page boundaries, guaranteeing 100% accurate page number citations.
- **Pure NumPy Vector Store:** No heavy vector databases, Docker containers, or background database daemons required. Embeddings are stored as local float32 `.npy` arrays with SHA-256 caching for zero-delay reuse upon refresh.

---

## 📦 Stack Specifications

| Component | Choice | Specification |
|---|---|---|
| **Text Generation LLM** | `LiquidAI/LFM3-1.7B` | Served via official GGUF `LiquidAI/LFM3-1.7B-GGUF:Q4_K_M` |
| **Inference Runtime** | `llama.cpp` (`llama-server`) | Localhost Metal offload (`http://127.0.0.1:8081/v1`) |
| **Embedding Model** | `all-MiniLM-L6-v2` | 384-dimensional dense vectors on CPU via `sentence-transformers` |
| **PDF Extraction & Preview** | `PyMuPDF` (`pymupdf`) | Block/bbox coordinate extraction and native page pixmap rendering |
| **Vector Retrieval** | `NumPy` | Unit-normalized cosine dot product ranking (`matrix @ query_vector`) |
| **User Interface** | `Streamlit` | Custom wide 3-column responsive layout with session state persistence |
| **Client** | `OpenAI` Python SDK | Streaming SSE client connecting directly to `127.0.0.1:8081/v1` |

---

## 🚀 Quickstart Commands

### Terminal A: Start LFM3 Server
```bash
brew install llama.cpp
./scripts/serve_llama.sh
```

### Terminal B: Launch Application
```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
./scripts/run_app.sh
```
Open **[http://127.0.0.1:8501](http://127.0.0.1:8501)** in your browser.

---

## 🧪 Verification & Acceptance Tests

Run the full automated test suite (16 tests across 4 modules):

```bash
PYTHONPATH=. .venv/bin/python -m pytest tests/ -v
```

### Test Coverage Summary
- `tests/test_ingest.py`: Multi-page extraction, 1-based page metadata, strict page boundary invariants, scanned PDF detection, filename sanitization.
- `tests/test_retrieval.py`: Persistence and cache reuse, deterministic ranking with orthogonal vectors, real MiniLM semantic ranking on synthetic car guide.
- `tests/test_generation.py`: Context budgeting and chunk pruning, citation extraction/validation, mock streaming generation, health check detection.
- `tests/test_pdf_preview.py`: PyMuPDF page rendering, highlight overlays, bounds safety checks, invalid page index handling.

---

## ⚠️ Known Boundaries & Realistic Limitations

1. **Scanned / Image-Only PDFs:** OCR is intentionally omitted in this lightweight MVP. Image-only PDFs trigger an explicit informative message: *"This PDF appears image-only; OCR is not included in this version."*
2. **Highlight Coordinate Granularity:** Bounding boxes represent the contributing text blocks on the page. When a chunk spans multiple blocks, all contributing blocks are highlighted and marked as composite highlights.
3. **1.7B Parameter Model Bounds:** `LFM3-1.7B` is extremely fast and efficient on 16 GB Macs, but complex multi-hop reasoning or ambiguous domain jargon may occasionally require clear, direct phrasing in user questions.
