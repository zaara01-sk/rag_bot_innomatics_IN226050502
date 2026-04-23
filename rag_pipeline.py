"""
RAG Pipeline Module
Handles PDF loading, chunking, embedding, and ChromaDB storage.
"""

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
import os


CHROMA_PERSIST_DIR = "./chroma_store"
COLLECTION_NAME = "support_docs"

# chunk_size=500 after trial and error — too large and retrieval gets noisy,
# too small and you lose sentence context at chunk edges.
# overlap=50 avoids dropping context right at boundaries.

CHUNK_SIZE = 500
CHUNK_OVERLAP = 50


def load_and_chunk(pdf_path: str) -> list:
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    loader = PyPDFLoader(pdf_path)
    pages = loader.load()
    print(f"[+] Loaded {len(pages)} pages from {pdf_path}")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ".", " "]
    )
    chunks = splitter.split_documents(pages)
    print(f"[+] Created {len(chunks)} chunks")
    return chunks


def build_vectorstore(chunks: list) -> Chroma:
    # sentence-transformers runs on CPU, no API key needed
    embeddings = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"}
    )

    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=CHROMA_PERSIST_DIR,
        collection_name=COLLECTION_NAME
    )
    # langchain-chroma auto-persists, no explicit .persist() needed
    print(f"[+] Stored {len(chunks)} chunks in ChromaDB at {CHROMA_PERSIST_DIR}")
    return vectorstore


def load_vectorstore() -> Chroma:
    embeddings = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"}
    )
    vectorstore = Chroma(
        persist_directory=CHROMA_PERSIST_DIR,
        embedding_function=embeddings,
        collection_name=COLLECTION_NAME
    )
    return vectorstore


def get_retriever(vectorstore: Chroma, k: int = 4):
    return vectorstore.as_retriever(search_kwargs={"k": k})
