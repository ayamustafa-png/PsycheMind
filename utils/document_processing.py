"""
Handles the "My Uploaded Document" knowledge source: extracting text from
a user-uploaded PDF/TXT/DOCX, cleaning + chunking it (same settings as the
main Psychology 2e pipeline), and building a temporary, in-memory Chroma
vector store for it. This is kept separate from vectorstore_utils.py,
which owns the persistent Psychology 2e vector store — the two knowledge
sources never mix.
"""

import os
import re
import tempfile
import uuid

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma


class DocumentProcessingError(Exception):
    """Raised when an uploaded file can't be read or processed."""
    pass


def clean_text(text):
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


def _extract_pdf(file_bytes):
    from langchain_community.document_loaders import PyPDFLoader

    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
        tmp.write(file_bytes)
        tmp_path = tmp.name

    try:
        pages = PyPDFLoader(tmp_path).load()
        return "\n\n".join(clean_text(p.page_content) for p in pages)
    finally:
        os.remove(tmp_path)


def _extract_docx(file_bytes):
    import docx
    import io

    doc = docx.Document(io.BytesIO(file_bytes))
    return clean_text("\n".join(p.text for p in doc.paragraphs))


def _extract_txt(file_bytes):
    return clean_text(file_bytes.decode("utf-8", errors="ignore"))


def extract_text(uploaded_file):
    """uploaded_file: a Streamlit UploadedFile object."""
    name = uploaded_file.name.lower()
    file_bytes = uploaded_file.getvalue()

    try:
        if name.endswith(".pdf"):
            text = _extract_pdf(file_bytes)
        elif name.endswith(".docx"):
            text = _extract_docx(file_bytes)
        elif name.endswith(".txt"):
            text = _extract_txt(file_bytes)
        else:
            raise DocumentProcessingError("Unsupported file type.")
    except DocumentProcessingError:
        raise
    except Exception as e:
        raise DocumentProcessingError(str(e))

    if not text or len(text) < 20:
        raise DocumentProcessingError("No readable text found in this file.")

    return text


def chunk_text(text, source_name):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=150
    )

    documents = splitter.create_documents([text])

    for i, doc in enumerate(documents):
        doc.metadata["chunk_id"] = i
        doc.metadata["source"] = source_name
        doc.metadata["page"] = "-"  # uploaded plain text has no page numbers

    return documents


def build_temp_vectorstore(chunks, embedding_model):
    """
    Builds an ephemeral, in-memory Chroma collection (no persist_directory,
    unique collection name) so each uploaded document gets its own
    isolated vector store that never touches the Psychology 2e database.
    """
    collection_name = f"upload-{uuid.uuid4().hex[:8]}"

    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embedding_model,
        collection_name=collection_name
    )

    return vectorstore


def process_uploaded_document(uploaded_file, embedding_model):
    """
    Full pipeline for one uploaded file: extract -> clean -> chunk ->
    embed -> temporary vector store. Returns (vectorstore, chunk_count).
    Raises DocumentProcessingError on any failure, with a message safe
    to show directly to the user.
    """
    text = extract_text(uploaded_file)
    chunks = chunk_text(text, source_name=uploaded_file.name)

    if not chunks:
        raise DocumentProcessingError("Could not split this document into chunks.")

    vectorstore = build_temp_vectorstore(chunks, embedding_model)

    return vectorstore, len(chunks)
