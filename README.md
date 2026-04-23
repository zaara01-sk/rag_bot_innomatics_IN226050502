# RAG Customer Support Assistant
**Innomatics Research Labs — Final Internship Project**
**Intern ID: IN226050502**

A Retrieval-Augmented Generation (RAG) system that answers customer support queries from a PDF knowledge base using LangGraph workflow orchestration and Human-in-the-Loop (HITL) escalation.

---

## Project Structure

```
rag_support_bot/
├── main.py              # CLI entry point
├── rag_pipeline.py      # PDF loading, chunking, ChromaDB ingestion
├── graph_workflow.py    # LangGraph nodes + HITL logic
requirements.txt
README.md
```

## Setup

```bash
# Create and activate virtual environment
python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # macOS/Linux

# Install dependencies
pip install -r requirements.txt

# Set your Groq API key
set GROQ_API_KEY=your_key_here       # Windows
# export GROQ_API_KEY=your_key_here  # macOS/Linux
```

## Dependencies

```
langchain
langchain-community
langchain-text-splitters
langchain-chroma
langchain-groq
langgraph
chromadb
sentence-transformers
pypdf
```

> **Note:** `langchain_text_splitters` and `langchain_chroma` are the updated modular packages.
> The older `langchain.text_splitter` and `langchain_community.vectorstores.Chroma` imports
> were deprecated in LangChain 0.2.9 and resolved during development.

## Usage

**Step 1 — Ingest your PDF:**
```bash
cd rag_support_bot
python main.py --ingest --pdf /path/to/support_docs.pdf
```

**Step 2 — Run interactive mode:**
```bash
python main.py --query
```

**Or ask a single question:**
```bash
python main.py --ask "What is the return policy?"
```

## How It Works

1. PDF → chunked (500 tokens, 50 overlap) → embedded with `all-MiniLM-L6-v2` (CPU, no API key) → stored in ChromaDB
2. User query → top-4 chunks retrieved → sent to `llama-3.1-8b-instant` via Groq with retrieved context
3. LangGraph routes the answer: low confidence OR complex query → HITL escalation
4. HITL node presents AI draft to human agent for review/override

## LangGraph Flow

```
[retrieve] → [generate] → [route] → (conditional)
                                    ├── [output]  (high confidence, routine query)
                                    └── [hitl]    (low confidence / complex / no context)
```

## HITL Escalation Triggers

| Condition | Example |
|---|---|
| Complex/sensitive keyword in query | "lawsuit", "fraud", "billing dispute" |
| Low confidence answer from LLM | Answer contains "cannot", "don't know" |
| No relevant chunks retrieved | Query outside PDF knowledge base |

## Tech Stack

| Component | Choice | Why |
|---|---|---|
| LLM | `llama-3.1-8b-instant` via Groq | Llama 3.1 — better instruction following vs 3.0, fast inference, free tier |
| Embeddings | `all-MiniLM-L6-v2` | Lightweight, CPU-compatible, no API cost |
| Vector DB | ChromaDB (`langchain-chroma`) | Local persistence, no external infra |
| Workflow | LangGraph | Clean state machine, native conditional routing |
| PDF Loading | LangChain `PyPDFLoader` | Multi-page support, metadata preserved |
| Text Splitting | `langchain-text-splitters` | Modular updated package (post LangChain 0.2.9) |
