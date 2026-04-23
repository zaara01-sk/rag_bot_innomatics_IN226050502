"""
Main entry point for the RAG Customer Support Assistant.
Usage:
    # First time: ingest your PDF knowledge base
    python main.py --ingest --pdf path/to/your_docs.pdf

    # Then run queries interactively
    python main.py --query

    # Or run a single query
    python main.py --ask "What is your return policy?"
"""

import argparse
import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from rag_pipeline import load_and_chunk, build_vectorstore, load_vectorstore, get_retriever
from graph_workflow import build_graph, SupportState

load_dotenv()
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
CHROMA_PERSIST_DIR = "./chroma_store"


def get_llm():
    return ChatGroq(
        model="llama-3.1-8b-instant",  
        api_key=GROQ_API_KEY,
        temperature=0.2
    )


def ingest_pdf(pdf_path: str):
    print(f"\n[Ingestion] Starting PDF ingestion: {pdf_path}")
    chunks = load_and_chunk(pdf_path)
    build_vectorstore(chunks)
    print("[Ingestion] Done! Knowledge base is ready.")


def run_query(query: str, app, verbose: bool = True):
    initial_state: SupportState = {
        "query": query,
        "retrieved_chunks": [],
        "answer": "",
        "confidence": "",
        "needs_human": False,
        "human_response": None,
        "final_output": ""
    }

    result = app.invoke(initial_state)

    if verbose:
        print("\n" + "="*60)
        print("QUERY:", query)
        print("="*60)
        print(result["final_output"])
        print("="*60 + "\n")

    return result["final_output"]


def interactive_mode(app):
    print("\n RAG Customer Support Assistant")
    print("Type 'quit' or 'exit' to stop.\n")
    while True:
        query = input("You: ").strip()
        if query.lower() in ("quit", "exit", "q"):
            print("Goodbye!")
            break
        if not query:
            continue
        run_query(query, app)


def main():
    parser = argparse.ArgumentParser(description="RAG Customer Support Assistant")
    parser.add_argument("--ingest", action="store_true", help="Ingest a PDF into the knowledge base")
    parser.add_argument("--pdf", type=str, help="Path to PDF for ingestion")
    parser.add_argument("--query", action="store_true", help="Start interactive query mode")
    parser.add_argument("--ask", type=str, help="Run a single query")
    args = parser.parse_args()

    if args.ingest:
        if not args.pdf:
            print("Error: --pdf is required with --ingest")
            return
        ingest_pdf(args.pdf)
        return

    # Load existing vectorstore
    if not os.path.exists(CHROMA_PERSIST_DIR):
        print("No knowledge base found. Please run with --ingest first.")
        return

    vectorstore = load_vectorstore()
    retriever = get_retriever(vectorstore, k=4)
    llm = get_llm()
    app = build_graph(retriever, llm)

    if args.ask:
        run_query(args.ask, app)
    elif args.query:
        interactive_mode(app)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
