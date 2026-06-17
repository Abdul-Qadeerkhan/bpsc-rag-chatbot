import streamlit as st
import os
from rag_engine import RAGEngine

# ─── Page Config ───────────────────────────────────────────────
st.set_page_config(
    page_title="BPSC Exam Prep Assistant",
    page_icon="🎓",
    layout="wide"
)

# ─── Custom CSS ────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Noto+Nastaliq+Urdu&family=Inter:wght@400;600;700&display=swap');

body { background-color: #0f172a; color: #e2e8f0; }

.main-header {
    background: linear-gradient(135deg, #1e3a5f 0%, #0f2744 100%);
    border: 1px solid #2d5a8e;
    border-radius: 16px;
    padding: 2rem;
    margin-bottom: 1.5rem;
    text-align: center;
}
.main-header h1 {
    font-family: 'Inter', sans-serif;
    font-size: 2rem;
    font-weight: 700;
    color: #60a5fa;
    margin: 0;
}
.main-header p {
    color: #94a3b8;
    margin-top: 0.5rem;
    font-size: 0.95rem;
}
.urdu-text {
    font-family: 'Noto Nastaliq Urdu', serif;
    font-size: 1.3rem;
    color: #fbbf24;
    direction: rtl;
}
.stat-box {
    background: #1e293b;
    border: 1px solid #334155;
    border-radius: 12px;
    padding: 1rem;
    text-align: center;
}
.stat-box .number { font-size: 1.8rem; font-weight: 700; color: #60a5fa; }
.stat-box .label  { font-size: 0.8rem; color: #64748b; text-transform: uppercase; letter-spacing: 0.05em; }
.source-card {
    background: #1e293b;
    border-left: 3px solid #3b82f6;
    border-radius: 8px;
    padding: 0.75rem 1rem;
    margin-top: 0.5rem;
    font-size: 0.85rem;
    color: #94a3b8;
}
.chat-user { background: #1e3a5f; border-radius: 12px 12px 2px 12px; padding: 0.75rem 1rem; margin: 0.5rem 0; }
.chat-bot  { background: #1e293b; border-radius: 12px 12px 12px 2px; padding: 0.75rem 1rem; margin: 0.5rem 0; border-left: 3px solid #3b82f6; }
.upload-zone {
    background: #1e293b;
    border: 2px dashed #334155;
    border-radius: 12px;
    padding: 1.5rem;
    text-align: center;
}
stFileUploader { background: transparent; }
</style>
""", unsafe_allow_html=True)

# ─── Header ────────────────────────────────────────────────────
st.markdown("""
<div class="main-header">
    <h1>🎓 BPSC Exam Prep Assistant</h1>
    <p>AI-powered chatbot for Balochistan Public Service Commission exam preparation</p>
    <div class="urdu-text">بلوچستان پبلک سروس کمیشن — امتحانی تیاری</div>
</div>
""", unsafe_allow_html=True)

# ─── Stats Row ─────────────────────────────────────────────────
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.markdown('<div class="stat-box"><div class="number">RAG</div><div class="label">AI Engine</div></div>', unsafe_allow_html=True)
with col2:
    doc_count = st.session_state.get("doc_count", 0)
    st.markdown(f'<div class="stat-box"><div class="number">{doc_count}</div><div class="label">Docs Loaded</div></div>', unsafe_allow_html=True)
with col3:
    chunk_count = st.session_state.get("chunk_count", 0)
    st.markdown(f'<div class="stat-box"><div class="number">{chunk_count}</div><div class="label">Text Chunks</div></div>', unsafe_allow_html=True)
with col4:
    msg_count = len(st.session_state.get("messages", []))
    st.markdown(f'<div class="stat-box"><div class="number">{msg_count}</div><div class="label">Q&A Pairs</div></div>', unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ─── Sidebar ───────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ⚙️ Configuration")
    # Try to load API key from Streamlit secrets (for deployed version)
    default_key = st.secrets.get("GROQ_API_KEY", "") if hasattr(st, "secrets") else ""
    api_key = st.text_input(
        "Groq API Key",
        value=default_key,
        type="password",
        help="Get free key at console.groq.com"
    )

    st.markdown("---")
    st.markdown("### 📁 Upload Study Material")
    st.markdown('<div class="upload-zone">Upload BPSC past papers, syllabus, or notes (PDF/TXT)</div>', unsafe_allow_html=True)
    uploaded_files = st.file_uploader(
        "Choose files", 
        type=["pdf", "txt"], 
        accept_multiple_files=True,
        label_visibility="collapsed"
    )

    if uploaded_files and api_key:
        if st.button("🚀 Process Documents", use_container_width=True):
            with st.spinner("Processing documents..."):
                engine = RAGEngine(api_key)
                result = engine.ingest_files(uploaded_files)
                st.session_state["rag_engine"] = engine
                st.session_state["doc_count"] = result["doc_count"]
                st.session_state["chunk_count"] = result["chunk_count"]
                st.success(f"✅ Processed {result['doc_count']} docs → {result['chunk_count']} chunks")
                st.rerun()

    st.markdown("---")
    st.markdown("### 💡 Sample Questions")
    sample_questions = [
        "What is the structure of BPSC CSS exam?",
        "Explain the geography of Balochistan",
        "What are the major rivers of Pakistan?",
        "History of Quetta city",
        "بلوچستان کے اضلاع کے نام بتائیں",
    ]
    for q in sample_questions:
        if st.button(q, use_container_width=True, key=f"sample_{q[:20]}"):
            st.session_state["prefill_question"] = q

    st.markdown("---")
    if st.button("🗑️ Clear Chat", use_container_width=True):
        st.session_state["messages"] = []
        st.rerun()

# ─── Session Init ──────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state["messages"] = []

# ─── Chat Area ─────────────────────────────────────────────────
chat_col, _ = st.columns([3, 1])
with chat_col:
    # Display chat history
    for msg in st.session_state["messages"]:
        if msg["role"] == "user":
            st.markdown(f'<div class="chat-user">🙋 <strong>You:</strong> {msg["content"]}</div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="chat-bot">🤖 <strong>Assistant:</strong><br>{msg["content"]}</div>', unsafe_allow_html=True)
            if msg.get("sources"):
                with st.expander("📚 Sources used"):
                    for src in msg["sources"]:
                        st.markdown(f'<div class="source-card">📄 {src}</div>', unsafe_allow_html=True)

    # Input
    prefill = st.session_state.pop("prefill_question", "")
    question = st.text_input(
        "Ask your question:",
        value=prefill,
        placeholder="e.g. What topics are important for BPSC General Knowledge paper?",
        key="question_input"
    )

    ask_btn = st.button("🔍 Ask AI", use_container_width=False, type="primary")

    if ask_btn and question:
        if not api_key:
            st.error("⚠️ Please enter your Groq API key in the sidebar.")
        elif "rag_engine" not in st.session_state:
            # Fallback: answer without documents using just LLM
            with st.spinner("Thinking..."):
                engine = RAGEngine(api_key)
                answer, sources = engine.answer_without_docs(question)
                st.session_state["messages"].append({"role": "user", "content": question})
                st.session_state["messages"].append({"role": "assistant", "content": answer, "sources": sources})
                st.rerun()
        else:
            with st.spinner("Searching knowledge base..."):
                engine = st.session_state["rag_engine"]
                answer, sources = engine.query(question)
                st.session_state["messages"].append({"role": "user", "content": question})
                st.session_state["messages"].append({"role": "assistant", "content": answer, "sources": sources})
                st.rerun()

    if not st.session_state["messages"]:
        st.info("💬 Upload study material in the sidebar, then ask questions — or just ask directly without uploading!")
