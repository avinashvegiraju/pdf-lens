# 🚀 START HERE — PDF Lens Quickstart Guide

Welcome to **PDF Lens**! This application lets you chat with your local PDF documents using **LiquidAI's LFM3-1.7B** model and **PyMuPDF**, running 100% on your Mac with zero cloud APIs, zero subscriptions, and complete data privacy.

---

## ⚡ 2-Minute Quickstart

### Terminal A: Start the Local LLM Server

In your first terminal window, start the llama.cpp inference server:

```bash
# 1. Install llama.cpp if not already installed
brew install llama.cpp

# 2. Run the server launcher (downloads official GGUF on first run)
cd pdf-lens
./scripts/serve_llama.sh
```

> **Note:** On first launch, `llama-server` automatically downloads `LiquidAI/LFM3-1.7B-GGUF:Q4_K_M` (~1.1 GB) from Hugging Face. Once ready, you will see `HTTP server listening on 127.0.0.1:8081`.

---

### Terminal B: Launch the PDF Lens App

In a second terminal window, install dependencies and launch the Streamlit app:

```bash
cd pdf-lens

# 1. Create and activate a Python 3.12 virtual environment
python3.12 -m venv .venv
source .venv/bin/activate

# 2. Install required packages
pip install -r requirements.txt

# 3. Start PDF Lens
./scripts/run_app.sh
```

Now open your web browser to:
👉 **[http://127.0.0.1:8501](http://127.0.0.1:8501)**

---

## 🔍 How to Test the App (30-Second Walkthrough)

1. In the **Left Column**, click the **🚗 Load Sample Car Manual** button.
   - It indexes the included 3-page synthetic car guide.
2. In the **Middle Column**, ask:
   > *"What should I do when the engine temperature warning turns red?"*
3. **Watch the Right Column:**
   - **Immediately** (before model generation begins), the top matching evidence card appears (`[S1]`, Page 1, Similarity score).
4. **Watch the Middle Column:**
   - The LFM3 model streams the grounded answer citing `[S1]`.
5. In the **Right Column**, click **View Page**:
   - The original PDF page appears below with translucent amber highlighting covering the exact paragraph where the answer came from!
6. Try asking an unrelated question:
   > *"What is Apple's current stock price?"*
   - Notice how the model refrains: *"I couldn't find that information in the selected PDF pages."*

---

## 🧠 Plain English AI Concepts: How It Works

| Concept | Plain English Explanation |
|---|---|
| **PDF** | Your raw source material (e.g., an owner's manual or report). |
| **Chunks** | Small paragraphs (~130–180 words) extracted from each page. Chunks never cross page boundaries so citations remain exact. |
| **Embeddings** | Mathematical fingerprints (lists of 384 numbers) describing what a passage *means*, not just the exact keywords it contains. |
| **Retrieval** | Comparing the question's numbers against every chunk's numbers to find the closest matches using simple vector geometry (Cosine Similarity). |
| **LFM3-1.7B** | Liquid AI's ultra-compact, high-efficiency language model that reads *only* the retrieved chunks and writes a clear answer. |
| **Right Evidence Panel** | The transparent proof. You can directly inspect the citations `[S1]` and original highlighted page so you never have to trust the AI blindly. |

---

## 📐 Deep Dive: Why 384 Dimensions & What is Cosine Similarity?

### Why 384 Dimensions?
The embedding model (`sentence-transformers/all-MiniLM-L6-v2`) represents human language in a 384-dimensional geometric space. 
Imagine a 2D map with X and Y (e.g., Latitude and Longitude). In 2D, you can map two concepts (like "hot vs cold" and "big vs small"). 
In 384 dimensions, the model has 384 different semantic directions to capture subtle nuances: tone, automotive parts, safety warnings, temperatures, electrical systems, and instructions.

### What is Cosine Similarity?
When chunks are embedded, each vector is normalized so its length is 1 (a unit vector on a 384-dimensional sphere). 
To find how similar two sentences are, we compute the angle $\theta$ between their vectors.
The **cosine similarity** is simply:

$$\text{similarity} = \cos(\theta) = \mathbf{A} \cdot \mathbf{B} = \sum_{i=1}^{384} A_i B_i$$

- **1.0**: The meanings point in the exact same direction.
- **0.0**: The concepts are completely unrelated (orthogonal).
- **-1.0**: The concepts are polar opposites.

Because the vectors are unit-normalized, cosine similarity is calculated instantly in NumPy with a single matrix multiplication (`matrix @ query_vector`) in under 5 milliseconds!

---

## 🛠️ Troubleshooting Guide

### 1. `llama-server offline` indicator in red
- **Cause:** The background inference server in Terminal A is not running or is on a different port.
- **Fix:** In Terminal A, execute `./scripts/serve_llama.sh`. Verify it says listening on `127.0.0.1:8081`. You can test it with `./scripts/check_llama.sh`.

### 2. Port 8081 already occupied
- **Cause:** Another process is using port 8081.
- **Fix:** Check what is running with `lsof -i :8081`. Kill the old process with `kill -9 <PID>` or change the port in `.env` and `scripts/serve_llama.sh`.

### 3. "This PDF appears image-only; OCR is not included in this version"
- **Cause:** The uploaded PDF is a scanned picture of paper rather than a digital PDF with embedded text.
- **Fix:** Open the PDF in Preview or Acrobat and check if you can highlight text with your cursor. PDF Lens MVP processes digital text blocks directly without OCR.

### 4. Slow first launch / Downloading model weights
- **Cause:** On first run, Hugging Face downloads the embedding weights (`all-MiniLM-L6-v2`, ~90 MB) and llama.cpp downloads `LFM3-1.7B-GGUF` (~1.1 GB).
- **Fix:** Wait for the initial download to complete. Subsequent runs load instantly from local cache without needing an active internet connection.

### 5. Memory usage on Apple M2 Pro (16 GB)
- `llama-server` running LFM3 Q4_K_M uses approximately **1.2 GB of unified memory**.
- The `all-MiniLM-L6-v2` embedding model uses approximately **180 MB of RAM**.
- The Streamlit app uses approximately **150 MB of RAM**.
- Total memory footprint is under **1.6 GB**, leaving over **14 GB of RAM free** for your system and other applications.
