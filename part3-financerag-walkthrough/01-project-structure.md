# Part 3, Section 1: Project Structure

## What You'll Learn
- How FinanceRAG is organized
- What each file does
- How the pieces connect

---

## The Big Picture

FinanceRAG has **4 core modules** that work together:

```
User Question
     ↓
[1] API receives request
     ↓
[2] RAG Chain coordinates everything
     ↓
[3] Vector Store finds relevant chunks
     ↓
[4] LLM generates answer
     ↓
Response with sources
```

---

## File Structure

```
FinanceRAG/
├── main.py                 # Entry point (CLI commands)
├── requirements.txt        # Dependencies
├── .env                    # API keys (not in git!)
│
├── src/
│   ├── config/
│   │   └── settings.py     # [1] Configuration
│   │
│   ├── tools/
│   │   ├── document_processor.py  # [2] Load & chunk docs
│   │   ├── vector_store.py        # [3] ChromaDB operations
│   │   └── rag_chain.py           # [4] RAG pipeline
│   │
│   └── api/
│       └── main.py         # [5] FastAPI endpoints
│
├── sample_docs/            # Your documents go here
│   ├── aml_policy.md
│   ├── basel_iii_capital_requirements.md
│   └── ...
│
└── data/
    └── chroma/             # Vector database storage
```

---

## How Files Connect

```
┌─────────────────────────────────────────────────────────────┐
│                         main.py                              │
│                    (CLI entry point)                         │
│                                                              │
│   python main.py ingest    →  document_processor.py          │
│                            →  vector_store.py                │
│                                                              │
│   python main.py query     →  rag_chain.py                   │
│                            →  vector_store.py                │
│                                                              │
│   python main.py serve     →  api/main.py                    │
│                            →  rag_chain.py                   │
└─────────────────────────────────────────────────────────────┘
```

---

## The 5 Files We'll Walk Through

| # | File | Purpose | Lines |
|---|------|---------|-------|
| 1 | `settings.py` | Configuration & environment vars | ~40 |
| 2 | `document_processor.py` | Load files, split into chunks | ~110 |
| 3 | `vector_store.py` | Store & search embeddings | ~130 |
| 4 | `rag_chain.py` | Orchestrate retrieval + generation | ~150 |
| 5 | `api/main.py` | REST endpoints | ~200 |

**Total:** ~630 lines of Python for a production-grade RAG system.

---

## Key Concept: Singletons

You'll see this pattern in several files:

```python
# Private variable to hold the single instance
_instance = None

def get_instance():
    """Get or create the singleton."""
    global _instance
    if _instance is None:
        _instance = MyClass()
    return _instance
```

**Why?** We only want ONE vector store connection, ONE RAG chain, etc. Creating multiple would waste memory and cause bugs.

---

## Next Up

Section 2: `settings.py` — where all configuration lives.
