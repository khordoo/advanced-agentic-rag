#!/usr/bin/env python3
"""
Streamlit Chunking Visualizer
==============================

Interactive app to visualize the 3-stage chunking strategy:
  1. Markdown header splitting
  2. Recursive split into parent chunks
  3. Sub-split into child chunks

Run with: uv run streamlit run examples/streamlit_app.py
"""

import streamlit as st
from langchain_text_splitters import (
    MarkdownHeaderTextSplitter,
    RecursiveCharacterTextSplitter,
)

# ── Page config ─────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="RAG Chunking Visualizer",
    page_icon="🔬",
    layout="wide",
)

# ── Custom CSS ──────────────────────────────────────────────────────────────

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

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
        border: 1px solid rgba(128, 128, 128, 0.3);
        border-radius: 12px;
        padding: 16px;
        margin-bottom: 12px;
        background: rgba(128, 128, 128, 0.05);
        transition: box-shadow 0.2s ease;
    }
    .chunk-card:hover {
        box-shadow: 0 4px 12px rgba(0,0,0,0.15);
    }
    .chunk-card pre {
        color: inherit !important;
        background: transparent !important;
    }

    .meta-tag {
        display: inline-block;
        background: rgba(128, 128, 128, 0.1);
        border: 1px solid rgba(128, 128, 128, 0.25);
        padding: 2px 8px;
        border-radius: 6px;
        font-size: 0.75rem;
        color: inherit;
        margin-right: 4px;
        margin-bottom: 4px;
    }

    .stat-box {
        background: linear-gradient(135deg, #f8fafc, #f1f5f9);
        border: 1px solid #e2e8f0;
        border-radius: 12px;
        padding: 20px;
        text-align: center;
    }
    .stat-number {
        font-size: 2rem;
        font-weight: 700;
        color: #6366f1;
    }
    .stat-label {
        font-size: 0.85rem;
        color: #64748b;
        margin-top: 4px;
    }

    div[data-testid="stExpander"] {
        border: 1px solid #e5e7eb;
        border-radius: 12px;
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

# ── Header ──────────────────────────────────────────────────────────────────

st.markdown('<p class="main-title">🔬 RAG Chunking Visualizer</p>', unsafe_allow_html=True)
st.markdown(
    "See how the **3-stage chunking strategy** breaks a document into "
    "searchable pieces. Adjust the parameters and watch the chunks change in real time."
)

# ── Sidebar: parameters ────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("### ⚙️ Chunking Parameters")

    st.markdown("---")
    st.markdown("**Stage 1: Header Split**")
    use_h1 = st.checkbox("Split on # (h1)", value=True)
    use_h2 = st.checkbox("Split on ## (h2)", value=True)
    use_h3 = st.checkbox("Split on ### (h3)", value=True)

    st.markdown("---")
    st.markdown("**Stage 2: Parent Chunks**")
    parent_size = st.slider("Parent chunk size (chars)", 200, 3000, 1500, step=100)
    parent_overlap = st.slider("Parent overlap (chars)", 0, 500, 150, step=25)

    st.markdown("---")
    st.markdown("**Stage 3: Child Chunks**")
    child_size = st.slider("Child chunk size (chars)", 50, 1000, 300, step=50)
    child_overlap = st.slider("Child overlap (chars)", 0, 200, 50, step=10)

    st.markdown("---")
    st.markdown(
        "💡 **Tip:** Try setting parent size to 500 and child size to 100 "
        "to see more granular chunking."
    )

# ── Input ───────────────────────────────────────────────────────────────────

tab_sample, tab_custom = st.tabs(["📄 Sample Document", "✏️ Custom Input"])

with tab_sample:
    input_text = SAMPLE_MARKDOWN
    st.code(input_text, language="markdown")

with tab_custom:
    input_text_custom = st.text_area(
        "Paste your markdown here",
        height=300,
        placeholder="# My Document\n\n## Section 1\n\nContent here...",
    )
    if input_text_custom.strip():
        input_text = input_text_custom

# ── Stage 1: Header splitting ──────────────────────────────────────────────

st.markdown("---")
st.markdown('<span class="stage-badge stage-1">STAGE 1</span>', unsafe_allow_html=True)
st.markdown("### 📑 Markdown Header Split")
st.markdown(
    "Split along heading boundaries to preserve document structure. "
    "Each chunk knows *which section* it belongs to via metadata."
)

headers = []
if use_h1:
    headers.append(("#", "h1"))
if use_h2:
    headers.append(("##", "h2"))
if use_h3:
    headers.append(("###", "h3"))

if not headers:
    st.warning("Select at least one header level to split on.")
    st.stop()

md_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers)
header_chunks = md_splitter.split_text(input_text)

col1, col2 = st.columns([3, 1])
with col1:
    for i, chunk in enumerate(header_chunks):
        with st.expander(
            f"Section {i + 1}  —  {' > '.join(chunk.metadata.values())}  "
            f"({len(chunk.page_content)} chars)",
            expanded=i == 0,
        ):
            # metadata tags
            meta_html = " ".join(
                f'<span class="meta-tag">{k}: {v}</span>'
                for k, v in chunk.metadata.items()
            )
            st.markdown(meta_html, unsafe_allow_html=True)
            st.text(chunk.page_content)

with col2:
    st.markdown(
        f'<div class="stat-box">'
        f'<div class="stat-number">{len(header_chunks)}</div>'
        f'<div class="stat-label">Header Sections</div>'
        f'</div>',
        unsafe_allow_html=True,
    )
    st.markdown("")
    avg_len = sum(len(c.page_content) for c in header_chunks) // max(len(header_chunks), 1)
    st.markdown(
        f'<div class="stat-box">'
        f'<div class="stat-number">{avg_len}</div>'
        f'<div class="stat-label">Avg chars / section</div>'
        f'</div>',
        unsafe_allow_html=True,
    )

# ── Stage 2: Parent splitting ──────────────────────────────────────────────

st.markdown("---")
st.markdown('<span class="stage-badge stage-2">STAGE 2</span>', unsafe_allow_html=True)
st.markdown("### 📦 Recursive Parent Split")
st.markdown(
    "Header sections can be **very long** — too long for precise retrieval. "
    f"We recursively split each section into parent chunks of ~**{parent_size}** chars "
    f"with **{parent_overlap}** chars overlap. The LLM reads these."
)

parent_splitter = RecursiveCharacterTextSplitter(
    separators=["\n\n", "\n", " ", ""],
    chunk_size=parent_size,
    chunk_overlap=parent_overlap,
)

all_parents = []
parent_section_map = []

for i, section in enumerate(header_chunks):
    parent_docs = parent_splitter.split_documents([section])
    for doc in parent_docs:
        all_parents.append(doc)
        parent_section_map.append(i)

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
        f'<div class="stat-box">'
        f'<div class="stat-number">{len(all_parents)}</div>'
        f'<div class="stat-label">Parent Chunks</div>'
        f'</div>',
        unsafe_allow_html=True,
    )
    st.markdown("")
    avg_p = sum(len(p.page_content) for p in all_parents) // max(len(all_parents), 1)
    st.markdown(
        f'<div class="stat-box">'
        f'<div class="stat-number">{avg_p}</div>'
        f'<div class="stat-label">Avg chars / parent</div>'
        f'</div>',
        unsafe_allow_html=True,
    )

# ── Stage 3: Child splitting ───────────────────────────────────────────────

st.markdown("---")
st.markdown('<span class="stage-badge stage-3">STAGE 3</span>', unsafe_allow_html=True)
st.markdown("### 🧩 Child Chunk Split")
st.markdown(
    "Each parent is further split into **small child chunks** for embedding. "
    f"Child size: ~**{child_size}** chars, overlap: **{child_overlap}** chars. "
    "These are what vector search matches against — smaller = more precise."
)

child_splitter = RecursiveCharacterTextSplitter(
    separators=["\n\n", "\n", " ", ""],
    chunk_size=child_size,
    chunk_overlap=child_overlap,
)

total_children = 0
col1, col2 = st.columns([3, 1])

with col1:
    for p_idx, parent in enumerate(all_parents):
        child_texts = child_splitter.split_text(parent.page_content)
        total_children += len(child_texts)

        section_label = " > ".join(parent.metadata.get(k, "") for k in ["h1", "h2", "h3"] if k in parent.metadata)

        with st.expander(
            f"Parent {p_idx + 1} → {len(child_texts)} children  —  {section_label}",
            expanded=False,
        ):
            for c_idx, child_text in enumerate(child_texts):
                st.markdown(
                    f'<div class="chunk-card">'
                    f'<strong>Child {c_idx + 1}</strong> '
                    f'<span class="meta-tag">{len(child_text)} chars</span>'
                    f'<span class="meta-tag">parent: {p_idx + 1}</span>'
                    f'<pre style="white-space: pre-wrap; font-size: 0.85rem; '
                    f'margin-top: 8px;">{child_text}</pre>'
                    f'</div>',
                    unsafe_allow_html=True,
                )

with col2:
    st.markdown(
        f'<div class="stat-box">'
        f'<div class="stat-number">{total_children}</div>'
        f'<div class="stat-label">Child Chunks</div>'
        f'</div>',
        unsafe_allow_html=True,
    )
    st.markdown("")
    avg_c = 0
    if total_children > 0:
        all_child_lens = []
        for parent in all_parents:
            for t in child_splitter.split_text(parent.page_content):
                all_child_lens.append(len(t))
        avg_c = sum(all_child_lens) // len(all_child_lens)

    st.markdown(
        f'<div class="stat-box">'
        f'<div class="stat-number">{avg_c}</div>'
        f'<div class="stat-label">Avg chars / child</div>'
        f'</div>',
        unsafe_allow_html=True,
    )

# ── Summary ─────────────────────────────────────────────────────────────────

st.markdown("---")
st.markdown("### 📊 Pipeline Summary")

c1, c2, c3, c4 = st.columns(4)
with c1:
    st.metric("Input Length", f"{len(input_text):,} chars")
with c2:
    st.metric("Header Sections", len(header_chunks))
with c3:
    st.metric("Parent Chunks", len(all_parents))
with c4:
    st.metric("Child Chunks", total_children)

st.markdown(
    f"**Flow:** {len(input_text):,} chars → "
    f"{len(header_chunks)} sections → "
    f"{len(all_parents)} parents → "
    f"{total_children} children (embedded in vector DB)"
)

st.markdown("---")
st.caption(
    "Built with Streamlit • Part of the Advanced RAG Pipeline • "
    "[View source](https://github.com/your-repo/advanced-rag)"
)
