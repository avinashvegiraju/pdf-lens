"""PDF Lens: Local PDF RAG Chat with Right-Side Evidence Viewer."""

import os
import sys
import time
from pathlib import Path
from typing import List, Optional, Dict, Any
import streamlit as st

# Configure page layout
st.set_page_config(
    page_title="PDF Lens — Local PDF RAG",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom styling for high readability and 3-column desktop layout
st.markdown("""
<style>
    /* Compact layout adjustments */
    .block-container {
        padding-top: 1.2rem;
        padding-bottom: 2rem;
        padding-left: 2rem;
        padding-right: 2rem;
    }
    .badge-ok {
        background-color: #e6f4ea;
        color: #137333;
        font-size: 0.82rem;
        font-weight: 600;
        padding: 3px 8px;
        border-radius: 4px;
        border: 1px solid #ceead6;
        display: inline-block;
    }
    .badge-err {
        background-color: #fce8e6;
        color: #c5221f;
        font-size: 0.82rem;
        font-weight: 600;
        padding: 3px 8px;
        border-radius: 4px;
        border: 1px solid #fad2cf;
        display: inline-block;
    }
    .source-card {
        border: 1px solid #e0e0e0;
        border-radius: 6px;
        padding: 10px;
        margin-bottom: 10px;
        background-color: #fafafa;
    }
    .source-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 4px;
    }
    .source-citation {
        font-weight: 700;
        color: #1a73e8;
    }
    .source-meta {
        font-size: 0.8rem;
        color: #5f6368;
    }
    .source-score {
        font-size: 0.78rem;
        background: #e8f0fe;
        color: #1967d2;
        padding: 2px 6px;
        border-radius: 4px;
    }
    .source-excerpt {
        font-size: 0.84rem;
        color: #3c4043;
        line-height: 1.35;
        border-left: 3px solid #1a73e8;
        padding-left: 8px;
        margin-top: 6px;
    }
    .metric-pill {
        font-size: 0.75rem;
        color: #5f6368;
        background: #f1f3f4;
        padding: 2px 6px;
        border-radius: 4px;
        margin-right: 6px;
    }
</style>
""", unsafe_allow_html=True)

from rag_pdf.config import config
from rag_pdf.models import PDFMetadata, ChunkMetadata, RetrievedSource, ChatMessage, GenerationMetrics
from rag_pdf.ingest import extract_and_chunk_pdf, compute_sha256
from rag_pdf.embeddings import get_embedding_model
from rag_pdf.retrieval import LocalVectorIndex
from rag_pdf.generation import (
    LLMClient,
    check_llama_health,
    extract_and_validate_citations
)
from rag_pdf.pdf_preview import render_pdf_page

# Ensure directories exist
config.ensure_directories()

# Initialize cached singletons
@st.cache_resource(show_spinner=False)
def load_embedder():
    return get_embedding_model()

@st.cache_resource(show_spinner=False)
def get_vector_index():
    return LocalVectorIndex(config.index_dir)

embedder = load_embedder()
vector_index = get_vector_index()

# ----------------- SESSION STATE INITIALIZATION -----------------
if "chat_messages" not in st.session_state:
    st.session_state.chat_messages = []

if "last_retrieved_sources" not in st.session_state:
    st.session_state.last_retrieved_sources = []

if "selected_source_for_preview" not in st.session_state:
    st.session_state.selected_source_for_preview = None

if "preview_show_highlights" not in st.session_state:
    st.session_state.preview_show_highlights = True

if "preview_zoom" not in st.session_state:
    st.session_state.preview_zoom = 1.4

if "active_doc_id" not in st.session_state:
    st.session_state.active_doc_id = "ALL"

# Load existing indexed documents from disk on first run
if "indexed_docs" not in st.session_state:
    docs = vector_index.list_indexed_documents()
    st.session_state.indexed_docs = {d.doc_id: d for d in docs}

# Helper to save uploaded PDF file
def store_pdf_bytes(pdf_bytes: bytes, doc_id: str) -> Path:
    target_path = config.uploads_dir / f"{doc_id}.pdf"
    if not target_path.exists():
        with open(target_path, "wb") as f:
            f.write(pdf_bytes)
    return target_path

def get_pdf_path(doc_id: str) -> Optional[Path]:
    p = config.uploads_dir / f"{doc_id}.pdf"
    if p.exists():
        return p
    # Check sample data directory
    sample_p = Path(__file__).parent / "sample_data" / "demo_vehicle_guide.pdf"
    if sample_p.exists():
        try:
            with open(sample_p, "rb") as f:
                if compute_sha256(f.read()) == doc_id:
                    return sample_p
        except Exception:
            pass
    return None

# ----------------- THREE-COLUMN LAYOUT -----------------
col_left, col_center, col_right = st.columns([1.15, 2.25, 1.8], gap="medium")

# =======================================================
# COLUMN 1: LEFT / LIBRARY & SETTINGS (~22%)
# =======================================================
with col_left:
    st.subheader("📚 PDF Library")
    
    # 1. Health indicator
    is_connected, health_msg = check_llama_health()
    if is_connected:
        st.markdown(f'<div class="badge-ok">🟢 {health_msg}</div>', unsafe_allow_html=True)
    else:
        st.markdown(f'<div class="badge-err">🔴 {health_msg}</div>', unsafe_allow_html=True)
        st.caption("Start local LFM3 server:")
        st.code("./scripts/serve_llama.sh", language="bash")

    st.markdown("---")

    # 2. Upload PDF
    uploaded_files = st.file_uploader(
        "Upload PDF(s)",
        type=["pdf"],
        accept_multiple_files=True,
        help=f"Max {config.max_pdf_size_mb} MB & {config.max_pdf_pages} pages per PDF. Text-based PDFs only."
    )

    if uploaded_files:
        for uf in uploaded_files:
            file_bytes = uf.read()
            doc_id = compute_sha256(file_bytes)
            if doc_id not in st.session_state.indexed_docs:
                with st.spinner(f"Indexing {uf.name}..."):
                    try:
                        meta, chunks = extract_and_chunk_pdf(file_bytes, uf.name)
                        if meta.is_scanned:
                            st.error(f"⚠️ {uf.name}: {meta.error_message}")
                        else:
                            store_pdf_bytes(file_bytes, doc_id)
                            vector_index.index_document(meta, chunks, embedding_model=embedder)
                            st.session_state.indexed_docs[doc_id] = meta
                            st.success(f"Indexed: {uf.name} ({len(chunks)} chunks)")
                    except Exception as e:
                        st.error(f"Error processing {uf.name}: {e}")

    # 3. Load Sample PDF button
    sample_pdf_path = Path(__file__).parent / "sample_data" / "demo_vehicle_guide.pdf"
    if sample_pdf_path.exists():
        if st.button("🚗 Load Sample Car Manual", use_container_width=True):
            with open(sample_pdf_path, "rb") as f:
                sb = f.read()
            s_doc_id = compute_sha256(sb)
            if s_doc_id not in st.session_state.indexed_docs:
                with st.spinner("Indexing sample manual..."):
                    meta, chunks = extract_and_chunk_pdf(sb, "demo_vehicle_guide.pdf")
                    store_pdf_bytes(sb, s_doc_id)
                    vector_index.index_document(meta, chunks, embedding_model=embedder)
                    st.session_state.indexed_docs[s_doc_id] = meta
                    st.session_state.active_doc_id = s_doc_id
                    st.rerun()
            else:
                st.session_state.active_doc_id = s_doc_id
                st.rerun()

    # 4. Document selector & stats
    doc_options = {"ALL": "All Indexed Documents"}
    for did, dmeta in st.session_state.indexed_docs.items():
        doc_options[did] = f"{dmeta.filename} ({dmeta.total_chunks} chunks)"

    selected_doc_key = st.selectbox(
        "Active Retrieval Scope",
        options=list(doc_options.keys()),
        format_func=lambda k: doc_options[k],
        index=0 if st.session_state.active_doc_id not in doc_options else list(doc_options.keys()).index(st.session_state.active_doc_id)
    )
    st.session_state.active_doc_id = selected_doc_key

    # Display document statistics
    if selected_doc_key != "ALL" and selected_doc_key in st.session_state.indexed_docs:
        active_meta = st.session_state.indexed_docs[selected_doc_key]
        st.markdown(f"""
        **File:** `{active_meta.filename}`  
        **Pages:** {active_meta.page_count}  
        **Chunks:** {active_meta.total_chunks}  
        **Size:** {active_meta.file_size_bytes / 1024:.1f} KB  
        """)

        # Document actions
        col_act1, col_act2 = st.columns(2)
        with col_act1:
            if st.button("🔄 Reindex", use_container_width=True):
                pdf_p = get_pdf_path(selected_doc_key)
                if pdf_p and pdf_p.exists():
                    with open(pdf_p, "rb") as f:
                        b = f.read()
                    meta, chunks = extract_and_chunk_pdf(b, active_meta.filename)
                    vector_index.index_document(meta, chunks, force_reindex=True, embedding_model=embedder)
                    st.session_state.indexed_docs[selected_doc_key] = meta
                    st.success("Reindexed!")
                    st.rerun()
        with col_act2:
            if st.button("🗑️ Delete", use_container_width=True):
                vector_index.delete_index(selected_doc_key)
                p = config.uploads_dir / f"{selected_doc_key}.pdf"
                if p.exists():
                    p.unlink()
                del st.session_state.indexed_docs[selected_doc_key]
                st.session_state.active_doc_id = "ALL"
                st.rerun()

    st.markdown("---")

    # 5. Search Controls
    top_k = st.slider("Top-K Passages", min_value=2, max_value=8, value=4, step=1)
    
    if st.button("🧹 Clear Chat", use_container_width=True):
        st.session_state.chat_messages = []
        st.session_state.last_retrieved_sources = []
        st.session_state.selected_source_for_preview = None
        st.rerun()

    # 6. Learning Expander
    with st.expander("ℹ️ How PDF Lens Works"):
        st.markdown("""
        **RAG Pipeline Lifecycle:**
        1. **Upload:** Reads native PDF blocks and geometry via PyMuPDF.
        2. **Chunk:** Cuts text into 130–180 word chunks strictly inside physical pages.
        3. **Embed:** Transforms passages into 384-dimensional unit vectors (`all-MiniLM-L6-v2`).
        4. **Retrieve:** Computes cosine dot-products `matrix @ query` in NumPy.
        5. **Generate:** Streams grounded answer from local **LiquidAI LFM3-1.7B** with citations `[S1]`, `[S2]`.
        """)


# =======================================================
# COLUMN 2: CENTER / CHAT (~43%)
# =======================================================
with col_center:
    st.markdown("### 💬 PDF Lens Chat")
    st.caption("Powered by **LiquidAI LFM3-1.7B** via local llama.cpp Metal inference")

    # Render chat history
    chat_container = st.container()
    with chat_container:
        for msg in st.session_state.chat_messages:
            with st.chat_message(msg.role):
                st.markdown(msg.content)
                if msg.metrics:
                    m = msg.metrics
                    st.markdown(
                        f'<div style="margin-top:4px;">'
                        f'<span class="metric-pill">⚡ Retrieval: {m.retrieval_ms:.0f}ms</span>'
                        f'<span class="metric-pill">⏱️ 1st token: {m.first_token_ms:.0f}ms</span>'
                        f'<span class="metric-pill">⏳ Gen: {m.total_generation_ms:.0f}ms</span>'
                        f'<span class="metric-pill">📝 {m.output_tokens} tokens</span>'
                        f'</div>',
                        unsafe_allow_html=True
                    )
                if msg.unverified_citations:
                    st.warning(f"Note: Unverified citation(s) mentioned: {', '.join(msg.unverified_citations)}")

    # Chat input
    user_query = st.chat_input("Ask a question about your indexed PDF(s)...")


# =======================================================
# COLUMN 3: RIGHT / EVIDENCE & PREVIEW (~35%)
# =======================================================
with col_right:
    st.markdown("### 📑 Retrieved Evidence")
    
    # Placeholders for dynamic instant-update before generation
    evidence_container = st.container()


# =======================================================
# CHAT EXECUTION & INSTANT EVIDENCE HOOK
# =======================================================
if user_query:
    # Append user question
    st.session_state.chat_messages.append(ChatMessage(role="user", content=user_query))
    with col_center:
        with st.chat_message("user"):
            st.markdown(user_query)

    # 1. Determine active doc_ids
    if st.session_state.active_doc_id == "ALL":
        target_doc_ids = list(st.session_state.indexed_docs.keys())
    else:
        target_doc_ids = [st.session_state.active_doc_id]

    if not target_doc_ids:
        with col_center:
            with st.chat_message("assistant"):
                st.warning("Please upload a PDF or load the sample car manual first.")
    else:
        # 2. IMMEDIATE RETRIEVAL BEFORE GENERATION STARTS!
        t_ret_start = time.perf_counter()
        sources = vector_index.search(
            query=user_query,
            doc_ids=target_doc_ids,
            top_k=top_k,
            embedding_model=embedder
        )
        t_ret_end = time.perf_counter()
        retrieval_duration_ms = (t_ret_end - t_ret_start) * 1000.0

        st.session_state.last_retrieved_sources = sources
        if sources:
            st.session_state.selected_source_for_preview = sources[0]
        else:
            st.session_state.selected_source_for_preview = None

        # 3. STREAM GENERATION IN CENTER COLUMN
        is_conn, _ = check_llama_health()
        if not is_conn:
            with col_center:
                with st.chat_message("assistant"):
                    st.error(
                        "⚠️ llama-server is currently offline! Top matching passages were retrieved "
                        "and are displayed in the right column, but generation requires the server."
                    )
                    st.info("Start the server in Terminal A:\n`./scripts/serve_llama.sh`")
        else:
            with col_center:
                with st.chat_message("assistant"):
                    response_placeholder = st.empty()
                    llm_client = LLMClient()
                    
                    full_response = ""
                    metrics = None

                    # Format brief chat history for follow-ups
                    history_payload = []
                    for m in st.session_state.chat_messages[:-1]:
                        history_payload.append({"role": m.role, "content": m.content})

                    try:
                        stream_gen = llm_client.stream_answer(
                            question=user_query,
                            sources=sources,
                            chat_history=history_payload
                        )
                        for token, final_metrics in stream_gen:
                            if token:
                                full_response += token
                                response_placeholder.markdown(full_response + "▌")
                            if final_metrics:
                                metrics = final_metrics

                        response_placeholder.markdown(full_response)
                        
                        if metrics:
                            metrics.retrieval_ms = round(retrieval_duration_ms, 1)

                        # Validate citations
                        verified_cits, unverified_cits = extract_and_validate_citations(full_response, sources)

                        # Store in history
                        st.session_state.chat_messages.append(
                            ChatMessage(
                                role="assistant",
                                content=full_response,
                                sources=sources,
                                metrics=metrics,
                                unverified_citations=unverified_cits
                            )
                        )
                    except Exception as e:
                        response_placeholder.error(f"Error during streaming generation: {e}")

        # Force rerun so evidence cards and interactive preview buttons sync seamlessly
        st.rerun()


# =======================================================
# RENDER EVIDENCE CARDS & PREVIEW IN RIGHT COLUMN
# =======================================================
with evidence_container:
    sources = st.session_state.last_retrieved_sources
    if not sources:
        st.info("Ask a question to see retrieved passages and original PDF page citations here.")
    else:
        st.markdown(f"**Top {len(sources)} Retrieved Passages:**")
        
        # Render cards with "View page" buttons
        for idx, src in enumerate(sources):
            card_col1, card_col2 = st.columns([3.8, 1.2])
            with card_col1:
                section_str = f" · {src.section_title}" if src.section_title else ""
                st.markdown(
                    f'<div class="source-header">'
                    f'  <span class="source-citation">{src.citation_id} {src.filename} (p. {src.page_number}{section_str})</span>'
                    f'  <span class="source-score">Similarity: {src.score:.3f}</span>'
                    f'</div>'
                    f'<div class="source-excerpt">{src.text[:220]}...</div>',
                    unsafe_allow_html=True
                )
            with card_col2:
                btn_key = f"view_src_{idx}_{src.chunk_id}"
                if st.button("View Page", key=btn_key, use_container_width=True):
                    st.session_state.selected_source_for_preview = src
                    st.rerun()

        st.markdown("---")
        
        # Render PDF Page Preview
        selected_src = st.session_state.selected_source_for_preview
        if selected_src:
            st.markdown(
                f"**Page Preview:** `{selected_src.filename}` — **Page {selected_src.page_number}**"
            )
            
            # Preview controls
            ctrl_col1, ctrl_col2 = st.columns([2, 2])
            with ctrl_col1:
                highlight_toggle = st.checkbox(
                    "Highlight Source Region",
                    value=st.session_state.preview_show_highlights,
                    key="toggle_highlights"
                )
                st.session_state.preview_show_highlights = highlight_toggle
            with ctrl_col2:
                zoom_options = [1.0, 1.2, 1.4, 1.6, 1.8]
                current_zoom = (
                    st.session_state.preview_zoom
                    if st.session_state.preview_zoom in zoom_options
                    else 1.4
                )
                zoom_val = st.select_slider(
                    "Zoom",
                    options=zoom_options,
                    value=current_zoom,
                    key="slider_zoom"
                )
                st.session_state.preview_zoom = zoom_val

            pdf_file_path = get_pdf_path(selected_src.doc_id)
            if pdf_file_path and pdf_file_path.exists():
                try:
                    preview_img, applied, status_msg = render_pdf_page(
                        pdf_source=pdf_file_path,
                        page_index=selected_src.page_index,
                        zoom=st.session_state.preview_zoom,
                        block_bboxes=selected_src.block_bboxes,
                        show_highlights=st.session_state.preview_show_highlights
                    )
                    st.caption(f"Status: {status_msg}")
                    st.image(preview_img, use_container_width=True)
                except Exception as e:
                    st.error(f"Error rendering PDF page: {e}")
            else:
                st.warning(f"Source PDF file `{selected_src.filename}` not found on disk.")
