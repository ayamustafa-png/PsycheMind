import os
import base64

import streamlit as st
from PIL import Image

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from sentence_transformers import CrossEncoder

from vectorstore_utils import build_or_load_vectorstore, get_embedding_model
from utils.document_processing import process_uploaded_document, DocumentProcessingError


# ==========================================================
# BRAND
# ==========================================================

BRAND = {
    "primary": "#7C5CFC",
    "navy": "#111827",
    "secondary": "#A78BFA",
    "background": "#F8FAFC",
    "text": "#1E293B",
    "card": "#1E293B",
}

ASSETS_DIR = os.path.join(os.path.dirname(__file__), "assets")
LOGO_PNG = os.path.join(ASSETS_DIR, "psychemind_logo.png")
LOGO_B64_FILE = os.path.join(ASSETS_DIR, "logo_b64.txt")

with open(LOGO_B64_FILE, "r") as f:
    LOGO_B64 = f.read().strip()

LOGO_IMG_TAG = f'<img src="data:image/png;base64,{LOGO_B64}" class="brand-logo" />'


# ==========================================================
# PAGE CONFIG
# ==========================================================

st.set_page_config(
    page_title="PsycheMind — AI-Powered Psychology Assistant",
    page_icon=Image.open(LOGO_PNG),
    layout="wide",
    initial_sidebar_state="expanded",
)


# ==========================================================
# CUSTOM CSS
# ==========================================================

st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Manrope:wght@400;500;600;700;800&display=swap');

html, body, [class*="css"] {{
    font-family: 'Manrope', sans-serif;
}}

.stApp {{
    background: {BRAND["background"]};
}}

/* Sidebar */
section[data-testid="stSidebar"] {{
    background: {BRAND["navy"]};
}}
section[data-testid="stSidebar"] * {{
    color: #E2E8F0 !important;
}}
section[data-testid="stSidebar"] hr {{
    border-color: #ffffff22;
}}

/* Header */
.brand-header {{
    background: linear-gradient(135deg, {BRAND["navy"]} 0%, #1B1F3B 60%, {BRAND["primary"]}33 100%);
    border-radius: 16px;
    padding: 2rem 2.2rem;
    margin-bottom: 1.4rem;
}}
.brand-logo {{
    width: 52px;
    height: 52px;
    vertical-align: middle;
    margin-right: 0.7rem;
}}
.brand-name {{
    font-size: 2rem;
    font-weight: 800;
    color: #FFFFFF;
    vertical-align: middle;
}}
.brand-tagline {{
    color: {BRAND["secondary"]};
    font-weight: 600;
    font-size: 0.95rem;
    margin: 0.3rem 0 0.6rem 0;
}}
.brand-description {{
    color: #CBD5E1;
    font-size: 0.98rem;
    max-width: 640px;
    line-height: 1.5;
}}
.sidebar-logo-row {{
    margin-bottom: 0.2rem;
}}

/* Badges */
.badge-row {{ margin-top: 1rem; }}
.badge {{
    display: inline-block;
    background: #FFFFFF14;
    border: 1px solid #FFFFFF33;
    color: #E2E8F0;
    border-radius: 999px;
    padding: 0.25rem 0.85rem;
    font-size: 0.78rem;
    font-weight: 600;
    margin-right: 0.5rem;
}}

/* Source indicator */
.source-indicator {{
    display: inline-block;
    background: {BRAND["primary"]}14;
    border: 1px solid {BRAND["primary"]}44;
    color: {BRAND["primary"]};
    border-radius: 8px;
    padding: 0.35rem 0.8rem;
    font-size: 0.88rem;
    font-weight: 600;
    margin-bottom: 0.8rem;
}}

/* Cards */
.card {{
    background: #FFFFFF;
    border: 1px solid #E2E8F0;
    border-radius: 14px;
    padding: 1.2rem 1.4rem;
    box-shadow: 0 1px 3px rgba(17,24,39,0.05);
    margin-bottom: 1rem;
}}
.answer-card {{
    background: #FFFFFF;
    border-left: 4px solid {BRAND["primary"]};
    border-radius: 12px;
    padding: 1.1rem 1.4rem;
    box-shadow: 0 1px 4px rgba(17,24,39,0.06);
    margin-bottom: 0.9rem;
    line-height: 1.6;
    color: {BRAND["text"]};
}}
.user-msg {{
    color: {BRAND["text"]};
    font-weight: 600;
    margin: 1.1rem 0 0.5rem 0;
}}

/* Inputs & buttons */
div[data-testid="stTextInput"] input {{
    border: 1.5px solid #CBD5E1 !important;
    border-radius: 10px !important;
    padding: 0.6rem 0.9rem !important;
}}
div[data-testid="stTextInput"] input:focus {{
    border-color: {BRAND["primary"]} !important;
    box-shadow: 0 0 0 1px {BRAND["primary"]}55 !important;
}}
.stButton > button {{
    border-radius: 10px;
    font-weight: 600;
    border: 1px solid transparent;
}}
.stButton > button[kind="primary"] {{
    background: {BRAND["primary"]};
    color: white;
}}
.stButton > button[kind="primary"]:hover {{
    background: #6c4ce0;
    color: white;
}}
div[data-testid="stExpander"] {{
    border-radius: 10px !important;
    border: 1px solid #E2E8F0 !important;
    background: #FFFFFF;
}}

.footer {{
    text-align: center;
    color: #94A3B8;
    font-size: 0.82rem;
    margin-top: 2.5rem;
    padding-top: 1rem;
    border-top: 1px solid #E2E8F0;
    line-height: 1.6;
}}
</style>
""", unsafe_allow_html=True)


# ==========================================================
# API KEY
# ==========================================================

api_key = st.secrets.get("GOOGLE_API_KEY", os.environ.get("GOOGLE_API_KEY"))

if not api_key:
    st.error(
        "GOOGLE_API_KEY is not set. Add it to Streamlit secrets "
        "(Settings → Secrets) or as an environment variable."
    )
    st.stop()

os.environ["GOOGLE_API_KEY"] = api_key


# ==========================================================
# CACHED MODEL / STORE LOADERS
# (Same models and settings as the original pipeline — only the
#  wiring to support two knowledge sources is new.)
# ==========================================================

@st.cache_resource(show_spinner=False)
def load_embedding_model():
    return get_embedding_model()


@st.cache_resource(show_spinner=False)
def load_book_vectorstore():
    return build_or_load_vectorstore()


@st.cache_resource(show_spinner=False)
def load_reranker():
    return CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")


@st.cache_resource(show_spinner=False)
def load_llm():
    return ChatGoogleGenerativeAI(
        model="gemini-3.5-flash",
        temperature=0.2,
        max_retries=4
    )


embedding_model = load_embedding_model()
book_vectorstore = load_book_vectorstore()
reranker = load_reranker()
llm = load_llm()


# ==========================================================
# PROMPT (unchanged from the original pipeline)
# ==========================================================

prompt = ChatPromptTemplate.from_template("""
You are an academic research assistant.

Answer the user's question using ONLY the information
provided in the context.

IMPORTANT CITATION RULES:

1. Use only the provided context.
2. Write the answer as clear paragraphs.
3. Do NOT add a citation after every sentence. Add citation(s) only
   ONCE, at the very end of each paragraph, covering everything stated
   in that paragraph.
4. Citations MUST use this exact format:
   [Source X | Page Y]
5. If a paragraph draws on more than one source, list them together
   at the end of that paragraph, e.g. [Source 1 | Page 12] [Source 3 | Page 45].
6. X and Y must come directly from the provided context.
7. Never invent a source number or page number.
8. If the context does not contain enough information, say:
   "I don't have enough information in the provided sources."
9. Do not use outside knowledge.

Context:
{context}

Question:
{question}

Answer:
""")


# ==========================================================
# RAG HELPERS (unchanged logic from the original pipeline)
# ==========================================================

def prepare_query(question):
    question = question.strip()
    if not question:
        raise ValueError("Question cannot be empty.")
    return question


def rerank_documents(question, documents, top_n=4, score_threshold=0.0):
    if not documents:
        return []

    pairs = [[question, doc.page_content] for doc in documents]
    scores = reranker.predict(pairs)
    scored_documents = list(zip(documents, scores))
    scored_documents.sort(key=lambda x: x[1], reverse=True)

    filtered_documents = [
        (doc, score) for doc, score in scored_documents
        if score >= score_threshold
    ]

    return filtered_documents[:top_n]


def remove_duplicate_documents(scored_documents):
    unique_documents = []
    seen_content = set()

    for doc, score in scored_documents:
        content = doc.page_content.strip()
        if content not in seen_content:
            unique_documents.append((doc, score))
            seen_content.add(content)

    return unique_documents


def format_docs(scored_documents):
    formatted_docs = []

    for i, (doc, score) in enumerate(scored_documents, start=1):
        page = doc.metadata.get("page", "Unknown")
        source = doc.metadata.get("source", "Unknown")

        formatted_docs.append(
            f"\n[Source {i} | Page {page}]\n\nSource file: {source}\n\n{doc.page_content}\n"
        )

    return "\n\n".join(formatted_docs)


def answer_question(question, retriever):
    question = prepare_query(question)

    retrieved_docs = retriever.invoke(question)

    reranked_docs = rerank_documents(question, retrieved_docs, top_n=4, score_threshold=0.0)
    final_docs = remove_duplicate_documents(reranked_docs)

    context = format_docs(final_docs)

    formatted_prompt = prompt.invoke({"context": context, "question": question})
    response = llm.invoke(formatted_prompt)
    answer = StrOutputParser().invoke(response)

    return answer, final_docs, len(retrieved_docs)


# ==========================================================
# SESSION STATE
# ==========================================================

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "doc_vectorstore" not in st.session_state:
    st.session_state.doc_vectorstore = None

if "doc_name" not in st.session_state:
    st.session_state.doc_name = None

if "doc_chunk_count" not in st.session_state:
    st.session_state.doc_chunk_count = None

if "knowledge_source" not in st.session_state:
    st.session_state.knowledge_source = "📖 Psychology Book"

if "question_box" not in st.session_state:
    st.session_state.question_box = ""


# ==========================================================
# SIDEBAR
# ==========================================================

with st.sidebar:

    st.markdown(
        f'<div class="sidebar-logo-row">{LOGO_IMG_TAG}'
        f'<span style="font-size:1.3rem;font-weight:800;color:white;vertical-align:middle;">PsycheMind</span></div>',
        unsafe_allow_html=True
    )
    st.caption("AI-Powered Psychology Assistant")
    st.markdown("---")

    st.markdown("**📚 Knowledge Base**")
    knowledge_source = st.radio(
        "Knowledge Source",
        ["📖 Psychology Book", "📂 My Uploaded Document"],
        key="knowledge_source",
        label_visibility="collapsed",
    )

    if knowledge_source == "📂 My Uploaded Document":
        st.markdown("**📂 Upload Your Own File**")
        uploaded_file = st.file_uploader(
            "Upload a document",
            type=["pdf", "txt", "docx"],
            label_visibility="collapsed",
        )

        if uploaded_file is not None:
            st.success(f"✓ File uploaded: {uploaded_file.name}")

            if st.button("🚀 Analyze Document", type="primary", use_container_width=True):
                with st.status("🔄 Processing document...", expanded=True) as status:
                    try:
                        st.write("✓ Extracting text")
                        st.write("✓ Cleaning text")
                        st.write("✓ Creating chunks")
                        st.write("✓ Generating embeddings")
                        vectorstore, chunk_count = process_uploaded_document(
                            uploaded_file, embedding_model
                        )
                        st.write("✓ Building vector database")
                        st.session_state.doc_vectorstore = vectorstore
                        st.session_state.doc_name = uploaded_file.name
                        st.session_state.doc_chunk_count = chunk_count
                        st.write("✓ Document ready")
                        status.update(label="✅ Document ready", state="complete")
                    except DocumentProcessingError:
                        status.update(label="❌ Processing failed", state="error")
                        st.error(
                            "We couldn't process this document. "
                            "Please check the file format and try again."
                        )

        if st.session_state.doc_vectorstore is not None:
            st.markdown(
                f'<div class="card" style="padding:0.8rem 1rem;">'
                f'<b>✅ Document Ready</b><br>{st.session_state.doc_name}<br>'
                f'{st.session_state.doc_chunk_count} chunks indexed</div>',
                unsafe_allow_html=True
            )

    st.markdown("---")
    st.markdown("**⚙️ RAG Pipeline**")
    st.markdown(
        "✓ Documents Loaded  \n"
        "✓ Vector Database Ready  \n"
        "✓ Reranker Loaded  \n"
        "✓ LLM Connected"
    )
    st.caption("Initial Retrieval: Top-10 · Final Documents: Top-4")
    st.caption("Embedding: MiniLM-L6-v2")
    st.caption("Reranker: MS MARCO MiniLM")

    st.markdown("---")
    st.markdown("**ℹ️ About PsycheMind**")
    st.caption("Version 1.0")
    st.caption("Built with Python, LangChain, ChromaDB, Cross-Encoder, Google Gemini")


# ==========================================================
# HEADER
# ==========================================================

st.markdown(f"""
<div class="brand-header">
    {LOGO_IMG_TAG}<span class="brand-name">PsycheMind</span>
    <div class="brand-tagline">AI-Powered Psychology Assistant</div>
    <div class="brand-description">
        Explore psychology concepts and academic documents using AI-powered
        retrieval, reranking, and source-grounded answers.
    </div>
    <div class="badge-row">
        <span class="badge">RAG Enabled</span>
        <span class="badge">Cross-Encoder Reranking</span>
        <span class="badge">Source Citations</span>
    </div>
</div>
""", unsafe_allow_html=True)


# ==========================================================
# ACTIVE SOURCE INDICATOR
# ==========================================================

if knowledge_source == "📖 Psychology Book":
    st.markdown('<div class="source-indicator">📖 Source: Psychology 2e</div>', unsafe_allow_html=True)
else:
    label = st.session_state.doc_name or "no document uploaded"
    st.markdown(f'<div class="source-indicator">📂 Source: {label}</div>', unsafe_allow_html=True)


# ==========================================================
# CORE: ASK A QUESTION
# ==========================================================

def get_active_retriever():
    if knowledge_source == "📖 Psychology Book":
        return book_vectorstore.as_retriever(search_kwargs={"k": 10}), "Psychology 2e"

    if st.session_state.doc_vectorstore is None:
        return None, None

    k = min(10, st.session_state.doc_chunk_count or 10)
    return st.session_state.doc_vectorstore.as_retriever(search_kwargs={"k": k}), st.session_state.doc_name


def ask(question):
    retriever, source_label = get_active_retriever()

    if retriever is None:
        st.warning("Please upload a document before selecting My Uploaded Document.")
        return

    try:
        answer, docs, retrieved_count = answer_question(question, retriever)
        st.session_state.chat_history.append({
            "question": question,
            "answer": answer,
            "docs": docs,
            "retrieved_count": retrieved_count,
            "source_label": source_label,
        })
    except Exception:
        st.session_state.chat_history.append({
            "question": question,
            "answer": None,
            "error": "PsycheMind couldn't generate the answer right now. Please try again.",
            "source_label": source_label,
        })


# ==========================================================
# EMPTY STATE / SUGGESTED QUESTIONS
# ==========================================================

EXAMPLE_QUESTIONS = [
    "What is classical conditioning?",
    "How does memory work?",
    "What are the main types of learning?",
    "What is the difference between sensation and perception?",
]

if not st.session_state.chat_history:
    st.markdown(
        '<div style="text-align:center; padding: 2rem 1rem 1rem 1rem;">'
        f'<div style="font-size:2.4rem;">🧠</div>'
        '<div style="font-size:1.4rem; font-weight:800; color:#1E293B; margin-top:0.4rem;">Welcome to PsycheMind</div>'
        '<div style="color:#64748B; max-width:520px; margin:0.6rem auto 0 auto;">'
        'Your intelligent assistant for psychology research and academic learning. '
        'Ask a question or upload your own document to explore it with AI.'
        '</div></div>',
        unsafe_allow_html=True
    )

    st.markdown("**Suggested Questions**")
    cols = st.columns(2)
    for i, q in enumerate(EXAMPLE_QUESTIONS):
        with cols[i % 2]:
            if st.button(q, key=f"suggested_{i}", use_container_width=True):
                ask(q)
                st.rerun()


# ==========================================================
# QUESTION INPUT
# ==========================================================

input_col, button_col = st.columns([5, 1])

with input_col:
    question_text = st.text_input(
        "Ask a question",
        key="question_box",
        placeholder="Ask a question about psychology or your document...",
        label_visibility="collapsed",
    )

with button_col:
    ask_clicked = st.button("✨ Ask PsycheMind", type="primary", use_container_width=True)

if ask_clicked:
    if not question_text.strip():
        st.warning("Please enter a question.")
    else:
        ask(question_text.strip())
        st.rerun()

if st.session_state.chat_history:
    if st.button("Clear Conversation"):
        st.session_state.chat_history = []
        st.rerun()


# ==========================================================
# CHAT HISTORY
# ==========================================================

for turn in reversed(st.session_state.chat_history):

    st.markdown(f'<div class="user-msg">👤 {turn["question"]}</div>', unsafe_allow_html=True)

    if turn.get("error"):
        st.error(turn["error"])
        continue

    st.markdown("**🧠 PsycheMind**")
    st.markdown(f'<div class="answer-card">{turn["answer"]}</div>', unsafe_allow_html=True)

    with st.expander("🔄 Retrieval Process"):
        st.write("✓ Query received")
        st.write(f"✓ Retrieved {turn['retrieved_count']} candidate chunks")
        st.write("✓ Cross-Encoder reranking completed")
        st.write(f"✓ Selected top {len(turn['docs'])} relevant chunks")
        st.write("✓ Context built")
        st.write("✓ Grounded answer generated")

    st.markdown("**📚 Retrieved Sources**")
    for i, (doc, score) in enumerate(turn["docs"], start=1):
        page = doc.metadata.get("page", "Unknown")
        with st.expander(f"Source {i} — Page {page}"):
            st.write(f"**Reranker Score:** {score:.4f}")
            st.write(f"**Source:** {doc.metadata.get('source', turn['source_label'])}")
            st.write("**Retrieved Content:**")
            st.write(doc.page_content)


# ==========================================================
# FOOTER
# ==========================================================

st.markdown("""
<div class="footer">
    <b>PsycheMind</b> — AI-Powered Psychology Assistant<br>
    Built with Python • LangChain • ChromaDB • Cross-Encoder • Gemini<br>
    Version 1.0
</div>
""", unsafe_allow_html=True)
