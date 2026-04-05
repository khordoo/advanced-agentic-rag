#!/usr/bin/env python3
"""
RAG Chunking Visualizer — Interactive two-page Streamlit app.

Page 1 (Chunking Lab): Paste or load markdown → hit Process → see 3-stage chunking.
Page 2 (Search & Chat): Query the indexed chunks → see retrieval pipeline → LLM answer.

Run with: uv run streamlit run examples/streamlit_app.py
"""

import sys
from pathlib import Path

import streamlit as st
from langchain_text_splitters import (
    MarkdownHeaderTextSplitter,
    RecursiveCharacterTextSplitter,
)

# Ensure rag_core is importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# ── Page config ─────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="RAG Chunking Visualizer",
    page_icon="🔬",
    layout="wide",
)

# ── CSS ─────────────────────────────────────────────────────────────────────

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

    .main-title {
        background: linear-gradient(135deg, #6366f1, #8b5cf6, #a855f7);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 2.2rem;
        font-weight: 700;
        margin-bottom: 0;
    }
    .stage-badge {
        display: inline-block;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 0.75rem;
        font-weight: 600;
        letter-spacing: 0.5px;
        margin-bottom: 8px;
    }
    .stage-1 { background: #dbeafe; color: #1e40af; }
    .stage-2 { background: #fce7f3; color: #9d174d; }
    .stage-3 { background: #d1fae5; color: #065f46; }

    .chunk-card {
        border: 1px solid rgba(128,128,128,0.3);
        border-radius: 12px;
        padding: 16px;
        margin-bottom: 12px;
        background: rgba(128,128,128,0.05);
        transition: box-shadow 0.2s ease;
    }
    .chunk-card:hover { box-shadow: 0 4px 12px rgba(0,0,0,0.15); }
    .chunk-card pre {
        color: inherit !important;
        background: transparent !important;
        white-space: pre-wrap;
        font-size: 0.85rem;
        margin-top: 8px;
    }

    .meta-tag {
        display: inline-block;
        background: rgba(128,128,128,0.1);
        border: 1px solid rgba(128,128,128,0.25);
        padding: 2px 8px;
        border-radius: 6px;
        font-size: 0.75rem;
        color: inherit;
        margin-right: 4px;
        margin-bottom: 4px;
    }

    .stat-box {
        background: linear-gradient(135deg, rgba(248,250,252,0.8), rgba(241,245,249,0.8));
        border: 1px solid rgba(128,128,128,0.2);
        border-radius: 12px;
        padding: 20px;
        text-align: center;
    }
    .stat-number { font-size: 2rem; font-weight: 700; color: #6366f1; }
    .stat-label { font-size: 0.85rem; color: #64748b; margin-top: 4px; }

    div[data-testid="stExpander"] {
        border: 1px solid rgba(128,128,128,0.2);
        border-radius: 12px;
        margin-bottom: 8px;
    }

    .search-result-header {
        font-size: 0.85rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 1px;
        color: #6366f1;
        margin-bottom: 12px;
    }

    .answer-card {
        border: 2px solid #6366f1;
        border-radius: 16px;
        padding: 24px;
        background: rgba(99,102,241,0.05);
        margin-top: 16px;
    }

    .highlight-match {
        border-left: 3px solid #6366f1;
        padding-left: 12px;
        margin-bottom: 8px;
    }
    .highlight-reranked {
        border-left: 3px solid #f59e0b;
        padding-left: 12px;
        margin-bottom: 8px;
    }
    .highlight-parent {
        border-left: 3px solid #22c55e;
        padding-left: 12px;
        margin-bottom: 8px;
    }
</style>
""", unsafe_allow_html=True)

# ── Sample document ─────────────────────────────────────────────────────────

SAMPLE_MARKDOWN = """# Annual Report 2024

Welcome to the annual report. This document covers financial performance, risk factors, and strategic outlook for fiscal year 2024.

## Financial Highlights

Revenue grew 15% year-over-year to $4.2 billion, driven by strong demand across all business segments. Operating income improved by 22%, reflecting disciplined cost management and operational efficiency gains. The gross margin expanded by 180 basis points to 42.3%, while free cash flow generation remained robust at $890 million, supporting continued investment in R&D and strategic acquisitions.

Capital expenditures totalled $450 million, primarily directed toward expanding manufacturing capacity in key growth markets including Southeast Asia, Latin America, and Eastern Europe, and upgrading digital infrastructure across all operating regions. The company returned $1.2 billion to shareholders through dividends and share buybacks during the fiscal year, demonstrating our commitment to delivering long-term shareholder value.

Net earnings per share increased to $6.72, up from $5.90 in the prior year, reflecting the combined impact of revenue growth, margin expansion, and disciplined share repurchase activity. The Board of Directors approved a 9% increase in the quarterly dividend to $1.01 per share.

By segment, Health Care delivered the strongest performance with sales growth of 8% to $1.1 billion, followed by Fabric & Home Care at 6% growth to $1.4 billion. The Grooming segment achieved mid-single-digit growth, while Beauty and Baby, Feminine & Family Care each grew in the low single digits. E-commerce sales now represent 18% of total revenue, up from 14% in the prior year.

The company completed two bolt-on acquisitions during the year for a combined consideration of $340 million, expanding our presence in the premium skincare and professional grooming categories. Integration of both businesses is on track and expected to be accretive to earnings within 12 months.

## Risk Factors

The company faces a broad range of risks that could materially affect our financial condition, results of operations, and cash flows. These include macroeconomic uncertainty, geopolitical tensions, evolving regulatory requirements across more than 70 countries where we operate, and increasing competition from both established multinational players and nimble direct-to-consumer startups.

Foreign exchange fluctuations represent a significant risk given that approximately 55% of our revenue is generated outside the United States. In fiscal 2024, unfavourable currency movements reduced reported net sales by approximately $420 million, or roughly 2 percentage points of growth. We manage this exposure through a combination of natural hedging and financial instruments, but complete mitigation is not possible.

Commodity cost inflation continues to pressure input costs. Key raw materials including resin, pulp, and palm oil experienced price increases of 8-15% during the fiscal year. While we have implemented pricing actions to offset these increases, there can be no assurance that further price increases will be accepted by consumers or retailers. Private-label competition intensifies during periods of sustained inflation.

### Cybersecurity Risks

Cybersecurity threats continue to evolve in sophistication and frequency. The company invested $120 million in cybersecurity infrastructure during the fiscal year, representing a 35% increase over the prior period. Our security operations centre now monitors more than 2.3 billion events per day across our global network. Despite these investments, no assurance can be given that the company's systems will not be breached, and any such breach could result in significant financial and reputational harm.

### Supply Chain Risks

Global supply chain disruptions have impacted procurement costs and lead times. The company has diversified its supplier base across 12 countries and maintained safety stock levels at 45 days of inventory to mitigate these risks. We have also invested $85 million in supply chain digitisation, including AI-powered demand forecasting and real-time logistics optimisation.

## Strategic Outlook

The company's strategy centres on three pillars: organic growth through product innovation, selective acquisitions in adjacent markets, and margin expansion via operational excellence. R&D investment increased to $680 million, with a focus on AI-driven analytics and cloud-native platforms. The company plans to launch 15 new products across all segments in fiscal 2025.
"""

# ── Session state init ──────────────────────────────────────────────────────

defaults = {
    "indexed": False,
    "processed": False,
    "vector_store": None,
    "doc_store": {},
    "all_parents": [],
    "header_chunks": [],
    "total_children": 0,
    "chat_history": [],
    # Cached chunking results for display
    "display_header_chunks": [],
    "display_all_parents": [],
    "display_parent_section_map": [],
    "display_total_children": 0,
    "display_child_data": [],
    "display_text_len": 0,
}
for key, val in defaults.items():
    if key not in st.session_state:
        st.session_state[key] = val

# ── Sidebar ─────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("### 🔬 RAG Visualizer")
    page = st.radio(
        "Navigate",
        ["📄 Chunking Lab", "🔍 Search & Chat"],
        label_visibility="collapsed",
    )

    st.markdown("---")
    st.markdown("### ⚙️ Chunking Parameters")

    st.markdown("**Stage 1: Header Split**")
    use_h1 = st.checkbox("Split on # (h1)", value=True)
    use_h2 = st.checkbox("Split on ## (h2)", value=True)
    use_h3 = st.checkbox("Split on ### (h3)", value=True)

    st.markdown("---")
    st.markdown("**Stage 2: Parent Chunks**")
    parent_size = st.slider("Parent chunk size", 200, 3000, 1500, step=100)
    parent_overlap = st.slider("Parent overlap", 0, 500, 150, step=25)

    st.markdown("---")
    st.markdown("**Stage 3: Child Chunks**")
    child_size = st.slider("Child chunk size", 50, 1000, 300, step=50)
    child_overlap = st.slider("Child overlap", 0, 200, 50, step=10)

    if st.session_state.indexed:
        st.markdown("---")
        st.success(
            f"✅ Indexed: {len(st.session_state.all_parents)} parents, "
            f"{st.session_state.total_children} children"
        )

# ── Helper: build headers list ──────────────────────────────────────────────

headers = []
if use_h1:
    headers.append(("#", "h1"))
if use_h2:
    headers.append(("##", "h2"))
if use_h3:
    headers.append(("###", "h3"))


# ── Indexing helper ─────────────────────────────────────────────────────────

def _index_chunks(all_parents, header_chunks, child_splitter, total_children):
    """Build an in-memory vector index from parent/child chunks."""
    import uuid
    from langchain_core.documents import Document
    from langchain_core.vectorstores import InMemoryVectorStore
    from langchain_huggingface import HuggingFaceEmbeddings

    emb = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vector_store = InMemoryVectorStore(emb)
    doc_store = {}

    all_children_docs = []
    for p_idx, parent in enumerate(all_parents):
        parent_id = uuid.uuid4().hex

        breadcrumb = " > ".join(
            str(parent.metadata.get(k, ""))
            for k in ["h1", "h2", "h3"]
            if k in parent.metadata
        )
        header = f"[Section: {breadcrumb}]\n\n"
        enriched = header + parent.page_content

        doc_store[parent_id] = {"content": enriched, "metadata": str(parent.metadata)}

        child_texts = child_splitter.split_text(parent.page_content)
        for child_text in child_texts:
            all_children_docs.append(
                Document(
                    page_content=header + child_text,
                    metadata={"parent": parent_id},
                )
            )

    vector_store.add_documents(all_children_docs)

    st.session_state.vector_store = vector_store
    st.session_state.doc_store = doc_store
    st.session_state.all_parents = all_parents
    st.session_state.header_chunks = header_chunks
    st.session_state.total_children = total_children
    st.session_state.indexed = True


# ═══════════════════════════════════════════════════════════════════════════
# PAGE 1: CHUNKING LAB
# ═══════════════════════════════════════════════════════════════════════════

if page == "📄 Chunking Lab":
    st.markdown('<p class="main-title">📄 Chunking Lab</p>', unsafe_allow_html=True)
    st.markdown(
        "Paste markdown (or load a sample), then hit **Process Document** to see the "
        "3-stage chunking strategy in action."
    )

    # ── Input section ───────────────────────────────────────────────────────

    col_input, col_btn = st.columns([5, 1])

    with col_btn:
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("📥 Load Sample", use_container_width=True):
            st.session_state.text_input = SAMPLE_MARKDOWN
            st.session_state.processed = False
            st.session_state.indexed = False
            st.rerun()

    with col_input:
        input_text = st.text_area(
            "Paste your markdown document here",
            height=200,
            placeholder="# My Document\n\n## Section 1\n\nContent here...",
            key="text_input",
        )

    # ── Process button (always visible below the text area) ───────────────────

    if st.button(
        "⚡  Process Document",
        type="primary",
        use_container_width=True,
    ):
        if not input_text.strip():
            st.error("Paste a markdown document above first.")
            st.stop()
        if not headers:
            st.warning("Select at least one header level in the sidebar.")
            st.stop()
        with st.spinner("Processing document — splitting headers, parents, and children..."):
            md_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers)
            hdr_chunks = md_splitter.split_text(input_text)

            p_splitter = RecursiveCharacterTextSplitter(
                separators=["\n\n", "\n", " ", ""],
                chunk_size=parent_size,
                chunk_overlap=parent_overlap,
            )
            c_splitter = RecursiveCharacterTextSplitter(
                separators=["\n\n", "\n", " ", ""],
                chunk_size=child_size,
                chunk_overlap=child_overlap,
            )

            parents = []
            psec_map = []
            for i, section in enumerate(hdr_chunks):
                for doc in p_splitter.split_documents([section]):
                    parents.append(doc)
                    psec_map.append(i)

            total_ch = 0
            ch_data = []
            for p_idx, parent in enumerate(parents):
                texts = c_splitter.split_text(parent.page_content)
                total_ch += len(texts)
                ch_data.append((p_idx, texts))

            st.session_state.display_header_chunks = hdr_chunks
            st.session_state.display_all_parents = parents
            st.session_state.display_parent_section_map = psec_map
            st.session_state.display_total_children = total_ch
            st.session_state.display_child_data = ch_data
            st.session_state.display_text_len = len(input_text)
            st.session_state.processed = True
            st.session_state.indexed = False

        st.rerun()

    # ── Show results (only after processing) ────────────────────────────────

    if not st.session_state.processed:
        if not input_text.strip():
            st.info("👆 Paste a markdown document above or click **Load Sample**, then hit **Process Document**.")
        st.stop()

    header_chunks = st.session_state.display_header_chunks
    all_parents = st.session_state.display_all_parents
    parent_section_map = st.session_state.display_parent_section_map
    total_children = st.session_state.display_total_children
    child_data = st.session_state.display_child_data
    text_len = st.session_state.display_text_len

    # --- Stage 1: Header split ---
    st.markdown("---")
    st.markdown('<span class="stage-badge stage-1">STAGE 1</span>', unsafe_allow_html=True)
    st.markdown("### 📑 Markdown Header Split")
    st.markdown(
        "Split along heading boundaries to preserve document structure. "
        "Each chunk knows *which section* it belongs to."
    )

    col1, col2 = st.columns([3, 1])
    with col1:
        for i, chunk in enumerate(header_chunks):
            with st.expander(
                f"Section {i + 1}  —  {' > '.join(chunk.metadata.values())}  "
                f"({len(chunk.page_content)} chars)",
                expanded=i == 0,
            ):
                meta_html = " ".join(
                    f'<span class="meta-tag">{k}: {v}</span>'
                    for k, v in chunk.metadata.items()
                )
                st.markdown(meta_html, unsafe_allow_html=True)
                st.text(chunk.page_content)

    with col2:
        st.markdown(
            f'<div class="stat-box"><div class="stat-number">{len(header_chunks)}</div>'
            f'<div class="stat-label">Header Sections</div></div>',
            unsafe_allow_html=True,
        )
        st.markdown("")
        avg_len = sum(len(c.page_content) for c in header_chunks) // max(len(header_chunks), 1)
        st.markdown(
            f'<div class="stat-box"><div class="stat-number">{avg_len}</div>'
            f'<div class="stat-label">Avg chars / section</div></div>',
            unsafe_allow_html=True,
        )

    # --- Stage 2: Parent split ---
    st.markdown("---")
    st.markdown('<span class="stage-badge stage-2">STAGE 2</span>', unsafe_allow_html=True)
    st.markdown("### 📦 Recursive Parent Split")

    sections_that_split = sum(
        1 for sec in header_chunks if len(sec.page_content) > parent_size
    )
    if sections_that_split == 0:
        st.info(
            f"All sections are under **{parent_size}** chars — no parent splitting needed. "
            f"Try lowering the parent chunk size in the sidebar to see splitting."
        )
    else:
        st.markdown(
            f"**{sections_that_split}** section(s) exceeded **{parent_size}** chars and "
            f"were recursively split into smaller parents."
        )

    col1, col2 = st.columns([3, 1])
    with col1:
        for i, (parent, sec_idx) in enumerate(zip(all_parents, parent_section_map)):
            section_label = " > ".join(header_chunks[sec_idx].metadata.values())
            needs_split = len(header_chunks[sec_idx].page_content) > parent_size
            split_indicator = " ✂️" if needs_split else ""

            with st.expander(
                f"Parent {i + 1}  —  {section_label}{split_indicator}  "
                f"({len(parent.page_content)} chars)",
                expanded=False,
            ):
                meta_html = " ".join(
                    f'<span class="meta-tag">{k}: {v}</span>'
                    for k, v in parent.metadata.items()
                )
                st.markdown(meta_html, unsafe_allow_html=True)
                st.text(parent.page_content)

    with col2:
        st.markdown(
            f'<div class="stat-box"><div class="stat-number">{len(all_parents)}</div>'
            f'<div class="stat-label">Parent Chunks</div></div>',
            unsafe_allow_html=True,
        )
        st.markdown("")
        avg_p = sum(len(p.page_content) for p in all_parents) // max(len(all_parents), 1)
        st.markdown(
            f'<div class="stat-box"><div class="stat-number">{avg_p}</div>'
            f'<div class="stat-label">Avg chars / parent</div></div>',
            unsafe_allow_html=True,
        )

    # --- Stage 3: Child split ---
    st.markdown("---")
    st.markdown('<span class="stage-badge stage-3">STAGE 3</span>', unsafe_allow_html=True)
    st.markdown("### 🧩 Child Chunk Split")
    st.markdown(
        "Each parent is split into **small child chunks** for embedding. "
        f"Child size: ~**{child_size}** chars, overlap: **{child_overlap}** chars. "
        "These are what vector search matches against — smaller = more precise."
    )

    col1, col2 = st.columns([3, 1])
    with col1:
        for p_idx, child_texts in child_data:
            section_label = " > ".join(
                all_parents[p_idx].metadata.get(k, "")
                for k in ["h1", "h2", "h3"]
                if k in all_parents[p_idx].metadata
            )

            with st.expander(
                f"Parent {p_idx + 1} → {len(child_texts)} children  —  {section_label}",
                expanded=p_idx == 1,
            ):
                for c_idx, child_text in enumerate(child_texts):
                    st.markdown(
                        f'<div class="chunk-card">'
                        f'<strong>Child {c_idx + 1}</strong> '
                        f'<span class="meta-tag">{len(child_text)} chars</span>'
                        f'<span class="meta-tag">parent: {p_idx + 1}</span>'
                        f'<pre>{child_text}</pre>'
                        f'</div>',
                        unsafe_allow_html=True,
                    )

    with col2:
        st.markdown(
            f'<div class="stat-box"><div class="stat-number">{total_children}</div>'
            f'<div class="stat-label">Child Chunks</div></div>',
            unsafe_allow_html=True,
        )
        st.markdown("")
        all_child_lens = [len(t) for _, texts in child_data for t in texts]
        avg_c = sum(all_child_lens) // max(len(all_child_lens), 1)
        st.markdown(
            f'<div class="stat-box"><div class="stat-number">{avg_c}</div>'
            f'<div class="stat-label">Avg chars / child</div></div>',
            unsafe_allow_html=True,
        )

    # --- Summary ---
    st.markdown("---")
    st.markdown("### 📊 Pipeline Summary")
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("Input Length", f"{text_len:,} chars")
    with c2:
        st.metric("Header Sections", len(header_chunks))
    with c3:
        st.metric("Parent Chunks", len(all_parents))
    with c4:
        st.metric("Child Chunks", total_children)

    st.markdown(
        f"**Flow:** {text_len:,} chars → "
        f"{len(header_chunks)} sections → "
        f"{len(all_parents)} parents → "
        f"{total_children} children"
    )

    # --- Index for search ---
    st.markdown("---")
    if st.session_state.indexed:
        st.success("✅ Chunks are indexed! Switch to **🔍 Search & Chat** in the sidebar.")
    else:
        if st.button(
            "✅  Index these chunks for search  →",
            type="primary",
            use_container_width=True,
        ):
            c_splitter = RecursiveCharacterTextSplitter(
                separators=["\n\n", "\n", " ", ""],
                chunk_size=child_size,
                chunk_overlap=child_overlap,
            )
            with st.spinner("Indexing — embedding children into vector store..."):
                _index_chunks(all_parents, header_chunks, c_splitter, total_children)
            st.rerun()


# ═══════════════════════════════════════════════════════════════════════════
# PAGE 2: SEARCH & CHAT
# ═══════════════════════════════════════════════════════════════════════════

elif page == "🔍 Search & Chat":
    st.markdown('<p class="main-title">🔍 Search & Chat</p>', unsafe_allow_html=True)

    if not st.session_state.indexed:
        st.warning(
            "⚠️ No document indexed yet. Go to **📄 Chunking Lab**, "
            "process a document, and click **Index these chunks for search**."
        )
        st.stop()

    st.markdown(
        f"**{len(st.session_state.all_parents)} parents** and "
        f"**{st.session_state.total_children} children** indexed. "
        "Enter a query to see the retrieval pipeline in action."
    )

    # ── Query input ─────────────────────────────────────────────────────────

    query = st.text_input(
        "🔎 Ask a question about the document",
        placeholder="e.g. How much did the company invest in cybersecurity?",
        key="query_input",
    )

    if not query.strip():
        st.info("👆 Type a question above to search the indexed document.")

        if st.session_state.chat_history:
            st.markdown("---")
            st.markdown("### 💬 Previous Queries")
            for entry in reversed(st.session_state.chat_history):
                with st.expander(f"Q: {entry['query']}", expanded=False):
                    st.markdown(f"**Answer:** {entry.get('answer', 'N/A')}")
                    st.caption(f"{entry['n_parents']} parent chunks retrieved")
        st.stop()

    # ── Run retrieval ───────────────────────────────────────────────────────

    vector_store = st.session_state.vector_store
    doc_store = st.session_state.doc_store

    with st.spinner("Searching children via vector similarity..."):
        child_results = vector_store.similarity_search(query, k=10)

    # ── Reranking ───────────────────────────────────────────────────────────

    with st.spinner("Reranking with cross-encoder..."):
        from sentence_transformers import CrossEncoder

        reranker = CrossEncoder("BAAI/bge-reranker-v2-m3")
        pairs = [(query, doc.page_content) for doc in child_results]
        scores = reranker.predict(pairs)
        scored = sorted(zip(scores, child_results), key=lambda x: x[0], reverse=True)
        top_n = min(3, len(scored))
        reranked = [(s, doc) for s, doc in scored[:top_n]]

    # ── Parent lookup ───────────────────────────────────────────────────────

    seen_parents = set()
    parent_results = []
    for _, doc in reranked:
        pid = doc.metadata.get("parent")
        if pid and pid not in seen_parents:
            seen_parents.add(pid)
            parent = doc_store.get(pid)
            if parent:
                parent_results.append(parent)

    # ── Visualize the 3-column retrieval pipeline ───────────────────────────

    st.markdown("---")
    st.markdown("### 🔬 Retrieval Pipeline")

    col_child, col_rerank, col_parent = st.columns(3)

    with col_child:
        st.markdown(
            '<div class="search-result-header">① Child Matches (Vector Search)</div>',
            unsafe_allow_html=True,
        )
        for i, doc in enumerate(child_results[:6]):
            preview = doc.page_content[:120].replace("\n", " ")
            st.markdown(
                f'<div class="highlight-match">'
                f'<span class="meta-tag">#{i+1}</span> '
                f'<span style="font-size:0.85rem">{preview}…</span>'
                f'</div>',
                unsafe_allow_html=True,
            )
        if len(child_results) > 6:
            st.caption(f"+ {len(child_results) - 6} more matches")

    with col_rerank:
        st.markdown(
            '<div class="search-result-header">② After Reranking (Cross-Encoder)</div>',
            unsafe_allow_html=True,
        )
        for i, (score, doc) in enumerate(reranked):
            preview = doc.page_content[:120].replace("\n", " ")
            score_color = "#22c55e" if score > 0.9 else "#f59e0b" if score > 0.5 else "#ef4444"
            st.markdown(
                f'<div class="highlight-reranked">'
                f'<span class="meta-tag" style="background:{score_color}22;'
                f'border-color:{score_color};color:{score_color}">'
                f'score: {score:.3f}</span> '
                f'<span style="font-size:0.85rem">{preview}…</span>'
                f'</div>',
                unsafe_allow_html=True,
            )

    with col_parent:
        st.markdown(
            '<div class="search-result-header">③ Parent Context (What LLM Reads)</div>',
            unsafe_allow_html=True,
        )
        for i, parent in enumerate(parent_results):
            content = parent.get("content", "")
            preview = content[:200].replace("\n", " ")
            st.markdown(
                f'<div class="highlight-parent">'
                f'<span class="meta-tag">Parent {i+1}</span> '
                f'<span style="font-size:0.85rem">{preview}…</span>'
                f'</div>',
                unsafe_allow_html=True,
            )

    # ── Expandable full context ─────────────────────────────────────────────

    with st.expander("📄 View full parent context sent to LLM", expanded=False):
        for i, parent in enumerate(parent_results):
            st.markdown(f"**Parent {i + 1}:**")
            st.text(parent.get("content", ""))
            st.markdown("---")

    # ── LLM Answer ──────────────────────────────────────────────────────────

    st.markdown("---")
    st.markdown("### 💬 LLM Answer")

    llm_answer = None

    try:
        from rag_core.config import load_config
        from rag_core.agents import AgentFactory, PromptLoader

        cfg = load_config()
        context_text = "\n\n----\n\n".join(
            p.get("content", "") for p in parent_results
        )
        prompt = PromptLoader.load("rag_qa", context=context_text, question=query)

        provider = None
        if cfg.get("OLLAMA_BASE_URL"):
            provider = "ollama"
        if cfg.get("OPEN_AI_COMPATIBLE_BASE_URL"):
            provider = "lmstudio"
        try:
            import boto3
            boto3.Session().get_credentials()
            provider = "aws"
        except Exception:
            pass

        if provider:
            with st.spinner(f"Generating answer via {provider}..."):
                llm = AgentFactory.create(provider, config=cfg)
                response = llm.invoke(prompt.template)
                llm_answer = response.content
        else:
            st.info(
                "💡 **No LLM provider configured.** The retrieval pipeline above works "
                "without an LLM. To see generated answers, configure a provider in `.env` "
                "(AWS Bedrock, Ollama, or LM Studio)."
            )

    except Exception as e:
        st.info(
            f"💡 **LLM not available** ({type(e).__name__}). "
            "The retrieval visualization above shows the full pipeline. "
            "Configure an LLM provider in `.env` to see generated answers."
        )

    if llm_answer:
        st.markdown(
            f'<div class="answer-card">{llm_answer}</div>',
            unsafe_allow_html=True,
        )
        st.session_state.chat_history.append({
            "query": query,
            "answer": llm_answer,
            "n_parents": len(parent_results),
        })
    elif parent_results:
        st.session_state.chat_history.append({
            "query": query,
            "answer": "(LLM not configured — retrieval results shown above)",
            "n_parents": len(parent_results),
        })

    # ── Chat history ────────────────────────────────────────────────────────

    if len(st.session_state.chat_history) > 1:
        st.markdown("---")
        st.markdown("### 📝 Query History")
        for entry in reversed(st.session_state.chat_history[:-1]):
            with st.expander(f"Q: {entry['query']}", expanded=False):
                st.markdown(f"**Answer:** {entry.get('answer', 'N/A')}")
                st.caption(f"{entry['n_parents']} parent chunks retrieved")

# ── Footer ──────────────────────────────────────────────────────────────────

st.markdown("---")
st.caption("Built with Streamlit • Part of the Advanced RAG Pipeline")
