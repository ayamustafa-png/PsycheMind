import os
import streamlit as st

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from sentence_transformers import CrossEncoder

from vectorstore_utils import build_or_load_vectorstore

# ==========================================================
# PAGE CONFIG
# ==========================================================

st.set_page_config(
    page_title="Psychology RAG",
    page_icon="🧠",
    layout="centered"
)

# Palette: five accent hues that stand in for five retrieved excerpts
# blending into one synthesized answer — echoes what the app actually
# does (pull several chunks, merge them into a single response).
PALETTE = ["#FF6B57", "#FFB238", "#2EC4B6", "#7C5CFC", "#FF4D97"]

st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,500;9..144,700&family=Work+Sans:wght@400;500;600&display=swap');

html, body, [class*="css"] {{
    font-family: 'Work Sans', sans-serif;
}}

.stApp {{
    background: #FFFBF5;
}}

h1, h2, h3 {{
    font-family: 'Fraunces', serif !important;
    color: #22223B !important;
}}

.hero-title {{
    font-family: 'Fraunces', serif;
    font-weight: 700;
    font-size: 2.4rem;
    color: #22223B;
    margin-bottom: 0.2rem;
}}

.hero-subtitle {{
    color: #55556B;
    font-size: 1.02rem;
    margin-bottom: 1rem;
}}

.chunk-stripe {{
    height: 6px;
    width: 100%;
    border-radius: 6px;
    margin: 0.6rem 0 1.6rem 0;
    background: linear-gradient(90deg, {PALETTE[0]} 0 20%, {PALETTE[1]} 20% 40%, {PALETTE[2]} 40% 60%, {PALETTE[3]} 60% 80%, {PALETTE[4]} 80% 100%);
}}

div[data-testid="stTextInput"] input {{
    border: 2px solid #22223B22 !important;
    border-radius: 10px !important;
    padding: 0.6rem 0.8rem !important;
    font-size: 1rem !important;
}}

div[data-testid="stTextInput"] input:focus {{
    border-color: {PALETTE[3]} !important;
    box-shadow: 0 0 0 1px {PALETTE[3]}55 !important;
}}

.stButton > button {{
    background: linear-gradient(90deg, {PALETTE[0]}, {PALETTE[3]});
    color: white;
    border: none;
    border-radius: 10px;
    padding: 0.45rem 1rem;
    font-weight: 500;
}}

.stButton > button:hover {{
    filter: brightness(1.07);
    color: white;
}}

.answer-card {{
    background: white;
    border-left: 6px solid {PALETTE[2]};
    border-radius: 10px;
    padding: 1.1rem 1.3rem;
    box-shadow: 0 2px 10px rgba(34,34,59,0.06);
    margin-bottom: 1.2rem;
    line-height: 1.55;
}}

div[data-testid="stExpander"] {{
    border-radius: 10px !important;
    border: 1px solid #22223B18 !important;
    margin-bottom: 0.5rem;
}}
</style>
""", unsafe_allow_html=True)


# ==========================================================
# TITLE
# ==========================================================

st.markdown('<div class="hero-title">🧠 Psychology Book Q&A</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="hero-subtitle">Ask a question — it retrieves the most '
    'relevant passages from Psychology 2e, blends them, and answers with '
    'page citations.</div>',
    unsafe_allow_html=True
)
st.markdown('<div class="chunk-stripe"></div>', unsafe_allow_html=True)


# ==========================================================
# API KEY
# ==========================================================
# Reads the key from Streamlit secrets (when deployed) or from an
# environment variable (when run locally). Nothing is hardcoded here,
# so this file is safe to push to a public GitHub repo.

api_key = st.secrets.get("GOOGLE_API_KEY", os.environ.get("GOOGLE_API_KEY"))

if not api_key:
    st.error(
        "GOOGLE_API_KEY is not set. Add it to Streamlit secrets "
        "(Settings → Secrets) or as an environment variable."
    )
    st.stop()

os.environ["GOOGLE_API_KEY"] = api_key


# ==========================================================
# LOAD MODELS
# ==========================================================

@st.cache_resource
def load_models():

    # Vector database (builds it on first run if it doesn't exist yet,
    # loads it straight away on every run after that)
    vectorstore = build_or_load_vectorstore()

    # Retrieve Top 10
    retriever = vectorstore.as_retriever(
        search_kwargs={"k": 10}
    )

    # Reranker
    reranker = CrossEncoder(
        "cross-encoder/ms-marco-MiniLM-L-6-v2"
    )

    # LLM
    llm = ChatGoogleGenerativeAI(
        model="gemini-3.5-flash",
        temperature=0.2,
        max_retries=4
    )

    return vectorstore, retriever, reranker, llm


vectorstore, retriever, reranker, llm = load_models()


# ==========================================================
# PROMPT
# ==========================================================

prompt = ChatPromptTemplate.from_template("""
You are an academic research assistant.

Answer the user's question using ONLY the information
provided in the context.

IMPORTANT CITATION RULES:

1. Use only the provided context.
2. Every important factual claim must have a citation.
3. Citations MUST use this exact format:
   [Source X | Page Y]
4. X and Y must come directly from the provided context.
5. Never invent a source number or page number.
6. If multiple sources support a claim, cite all relevant sources.
7. If the context does not contain enough information, say:
   "I don't have enough information in the provided sources."
8. Do not use outside knowledge.

Context:
{context}

Question:
{question}

Answer:
""")


# ==========================================================
# QUERY PREPARATION
# ==========================================================

def prepare_query(question):

    question = question.strip()

    if not question:
        raise ValueError("Question cannot be empty.")

    return question


# ==========================================================
# RERANKING
# ==========================================================

def rerank_documents(
    question,
    documents,
    top_n=4,
    score_threshold=0.0
):

    if not documents:
        return []

    pairs = [
        [question, doc.page_content]
        for doc in documents
    ]

    scores = reranker.predict(pairs)

    scored_documents = list(
        zip(documents, scores)
    )

    # Highest score first
    scored_documents.sort(
        key=lambda x: x[1],
        reverse=True
    )

    # Apply threshold
    filtered_documents = [
        (doc, score)
        for doc, score in scored_documents
        if score >= score_threshold
    ]

    # Keep Top N
    final_documents = filtered_documents[:top_n]

    return final_documents


# ==========================================================
# REMOVE DUPLICATES
# ==========================================================

def remove_duplicate_documents(
    scored_documents
):

    unique_documents = []
    seen_content = set()

    for doc, score in scored_documents:

        content = doc.page_content.strip()

        if content not in seen_content:

            unique_documents.append(
                (doc, score)
            )

            seen_content.add(content)

    return unique_documents


# ==========================================================
# RETRIEVE + RERANK
# ==========================================================

def retrieve_and_rerank(question):

    question = prepare_query(question)

    # Retrieve Top 10
    retrieved_docs = retriever.invoke(
        question
    )

    # Rerank
    reranked_docs = rerank_documents(
        question,
        retrieved_docs,
        top_n=4,
        score_threshold=0.0
    )

    # Remove duplicates
    final_docs = remove_duplicate_documents(
        reranked_docs
    )

    return final_docs


# ==========================================================
# FORMAT CONTEXT
# ==========================================================

def format_docs(scored_documents):

    formatted_docs = []

    for i, (doc, score) in enumerate(
        scored_documents,
        start=1
    ):

        page = doc.metadata.get(
            "page",
            "Unknown"
        )

        source = doc.metadata.get(
            "source",
            "Unknown"
        )

        formatted_docs.append(
            f"""
[Source {i} | Page {page}]

Source file: {source}

{doc.page_content}
"""
        )

    return "\n\n".join(formatted_docs)


# ==========================================================
# RAG FUNCTION
# ==========================================================

def answer_question(question):

    # Retrieve + rerank
    docs = retrieve_and_rerank(
        question
    )

    # Build context
    context = format_docs(
        docs
    )

    # Build prompt
    formatted_prompt = prompt.invoke(
        {
            "context": context,
            "question": question
        }
    )

    # Generate answer
    response = llm.invoke(
        formatted_prompt
    )

    answer = StrOutputParser().invoke(
        response
    )

    return answer, docs


# ==========================================================
# USER INTERFACE
# ==========================================================

EXAMPLE_QUESTIONS = [
    "What is classical conditioning?",
    "How does short-term memory work?",
    "What causes cognitive dissonance?",
]

if "question" not in st.session_state:
    st.session_state.question = ""

st.write("**Try asking:**")
chip_cols = st.columns(len(EXAMPLE_QUESTIONS))
for col, example in zip(chip_cols, EXAMPLE_QUESTIONS):
    with col:
        if st.button(example, key=f"chip_{example}", use_container_width=True):
            st.session_state.question = example

question = st.text_input(
    "🔎 Ask a question about the book:",
    placeholder="Example: What is classical conditioning?",
    key="question"
)


if question:

    with st.spinner(
        "🔍 Searching → Reranking → Generating answer..."
    ):

        try:

            answer, docs = answer_question(
                question
            )

            # ------------------------------------------
            # ANSWER
            # ------------------------------------------

            st.subheader("💡 Answer")

            st.markdown(
                f'<div class="answer-card">{answer}</div>',
                unsafe_allow_html=True
            )

            # ------------------------------------------
            # SOURCES
            # ------------------------------------------

            st.subheader("📚 Retrieved Sources")

            for i, (doc, score) in enumerate(
                docs,
                start=1
            ):

                page = doc.metadata.get(
                    "page",
                    "Unknown"
                )

                dot_color = PALETTE[(i - 1) % len(PALETTE)]

                with st.expander(
                    f"Source {i} — Page {page}"
                ):

                    st.markdown(
                        f'<span style="color:{dot_color};">●</span> '
                        f'**Reranker Score:** {score:.4f}',
                        unsafe_allow_html=True
                    )

                    st.write(
                        doc.page_content
                    )

        except Exception as e:

            st.error(
                f"Something went wrong: {str(e)}"
            )
