# 🔍 PDF Lens v2 — Local Agentic PDF RAG with Evidence Viewer

**PDF Lens v2** is a lightweight, fully private **Agentic Retrieval-Augmented Generation (RAG)** system built for Apple Silicon Macs. It combines an autonomous multi-agent reasoning loop (Query Planner, Hybrid Search Tool, Python Math Sandbox, and Self-Reflective Critic) with local **LiquidAI LFM3-1.7B** Metal GPU inference and PyMuPDF geometric highlight previews.

---

## 🚀 What's New in v2 (Agentic RAG)
- **Autonomous Query Planning:** Decomposes complex and comparative multi-part prompts into targeted sub-queries.
- **Hybrid Retrieval (Dense + BM25):** Combines 384-dim dense embeddings (`all-MiniLM-L6-v2`) with sparse lexical BM25 (`rank-bm25`) for exact part numbers, fault codes, and keywords.
- **Python Calculation Sandbox:** Executes local, sandboxed Python expressions for math and unit conversions (e.g., PSI to kPa/bar).
- **Self-Reflective Critic Agent (CRAG):** Fact-checks draft answers, verifies `[S#]` citations, and detects hallucinations before presenting the final answer.
- **Multi-Page Evidence Canvas:** Enables quick-switching and previewing across all distinct physical pages cited in an answer.
- **Live Hardware Telemetry:** Displays real-time Retrieval Latency, TTFT, Tokens/sec, Total Response Time, and RAM footprint.

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
