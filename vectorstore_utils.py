"""
Builds (or loads, if already built) the Chroma vector store for the
Psychology2e RAG project. Kept in one place so both app.py and
build_vectorstore.py use the exact same logic.
"""

import os
import re
import requests

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

PDF_URL = "https://assets.openstax.org/oscms-prodcms/media/documents/Psychology2e_WEB.pdf"
PDF_PATH = "psychology2e.pdf"
PERSIST_DIR = "chroma_db"
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


def download_pdf():
    if os.path.exists(PDF_PATH):
        return

    response = requests.get(PDF_URL)

    if response.status_code == 200:
        with open(PDF_PATH, "wb") as f:
            f.write(response.content)
        print("PDF downloaded successfully.")
    else:
        raise RuntimeError(f"Failed to download PDF. Status code: {response.status_code}")


def clean_text(text):
    text = re.sub(r'\s+', ' ', text)
    text = text.strip()
    return text


def build_chunks():
    download_pdf()

    loader = PyPDFLoader(PDF_PATH)
    pages = loader.load()

    for page in pages:
        page.page_content = clean_text(page.page_content)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=150
    )
    chunks = splitter.split_documents(pages)

    for i, chunk in enumerate(chunks):
        chunk.metadata["chunk_id"] = i

    print(f"Number of pages: {len(pages)}")
    print(f"Number of chunks: {len(chunks)}")

    return chunks


def get_embedding_model():
    return HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)


def build_or_load_vectorstore():
    """
    If a persisted Chroma DB already exists on disk, load it.
    Otherwise build it from scratch from the PDF (downloads it if needed).
    This is what lets the app start up on its own on any host, with no
    manual step required.
    """
    embedding_model = get_embedding_model()

    if os.path.exists(PERSIST_DIR) and os.listdir(PERSIST_DIR):
        vectorstore = Chroma(
            persist_directory=PERSIST_DIR,
            embedding_function=embedding_model
        )
        print("Loaded existing vector store.")
        return vectorstore

    chunks = build_chunks()

    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embedding_model,
        persist_directory=PERSIST_DIR
    )
    print("Vector store created successfully")
    print(f"Number of vectors stored: {vectorstore._collection.count()}")

    return vectorstore


if __name__ == "__main__":
    build_or_load_vectorstore()
