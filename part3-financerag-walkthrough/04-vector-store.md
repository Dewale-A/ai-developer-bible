# Part 3, Section 4: Vector Store

## What You'll Learn
- How ChromaDB stores embeddings
- Similarity search mechanics
- The singleton pattern for database connections

---

## What the Vector Store Does

```
Document Chunks          Embeddings              ChromaDB
      │                      │                      │
      ▼                      ▼                      ▼
["Basel III..."]  →  [0.12, -0.34, ...]  →  Stored on disk
["AML policy..."] →  [0.45, 0.23, ...]   →  Searchable!
```

---

## File: `src/tools/vector_store.py`

### Part 1: Imports & Setup

```python
from typing import List, Optional, Dict, Any
import structlog

from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma

from src.config.settings import get_settings

logger = structlog.get_logger()
```

**Key imports:**

| Import | Purpose |
|--------|---------|
| `OpenAIEmbeddings` | Converts text → vectors via OpenAI API |
| `Chroma` | LangChain wrapper for ChromaDB |

---

### Part 2: Class Initialization

```python
class VectorStoreManager:
    """Manage ChromaDB vector store operations."""
    
    def __init__(self):
        self.settings = get_settings()
        self.embeddings = OpenAIEmbeddings(
            model=self.settings.embedding_model,
            openai_api_key=self.settings.openai_api_key
        )
        self._vector_store: Optional[Chroma] = None
```

**What happens here:**
1. Load settings (API key, model name)
2. Create embeddings object (will call OpenAI when needed)
3. `_vector_store = None` — lazy initialization (created when first used)

---

### Part 3: Lazy Vector Store Property

```python
@property
def vector_store(self) -> Chroma:
    """Get or initialize vector store."""
    if self._vector_store is None:
        self._vector_store = Chroma(
            collection_name=self.settings.collection_name,
            embedding_function=self.embeddings,
            persist_directory=self.settings.chroma_persist_dir
        )
    return self._vector_store
```

**The `@property` pattern:**
```python
# When you write:
manager.vector_store.add_documents(...)

# Python actually calls:
manager.vector_store()  # Returns Chroma instance
```

**Why lazy?** Don't connect to database until actually needed. Faster startup.

**Chroma parameters:**

| Parameter | Value | Purpose |
|-----------|-------|---------|
| `collection_name` | "finance_docs" | Group related documents |
| `embedding_function` | OpenAIEmbeddings | How to create vectors |
| `persist_directory` | "./data/chroma" | Save to disk (survives restarts) |

---

### Part 4: Adding Documents

```python
def add_documents(self, documents: List[Document]) -> List[str]:
    """Add documents to the vector store."""
    if not documents:
        logger.warning("no_documents_to_add")
        return []
    
    try:
        ids = self.vector_store.add_documents(documents)
        logger.info("documents_added", count=len(ids))
        return ids
    except Exception as e:
        logger.error("add_documents_error", error=str(e))
        raise
```

**What `add_documents()` does internally:**

```
Step 1: Extract text from each Document
        ["Basel III requires...", "AML policy states..."]

Step 2: Send to OpenAI Embeddings API
        → Returns [[0.12, -0.34, ...], [0.45, 0.23, ...]]

Step 3: Store in ChromaDB
        - Vector + metadata + unique ID

Step 4: Return IDs
        ["abc123", "def456"]
```

---

### Part 5: Similarity Search

```python
def similarity_search(
    self, 
    query: str, 
    k: Optional[int] = None,
    filter: Optional[Dict[str, Any]] = None
) -> List[Document]:
    """Search for similar documents."""
    k = k or self.settings.top_k_results
    
    try:
        results = self.vector_store.similarity_search(
            query=query,
            k=k,
            filter=filter
        )
        logger.info("similarity_search", query_preview=query[:50], results=len(results))
        return results
    except Exception as e:
        logger.error("search_error", error=str(e))
        raise
```

**What happens during search:**

```
Query: "What are capital requirements?"
        │
        ▼
[1] Embed the query
    "What are capital requirements?" → [0.23, -0.12, 0.89, ...]
        │
        ▼
[2] Find nearest vectors in ChromaDB
    Compare query vector to all stored vectors
    Using cosine similarity or L2 distance
        │
        ▼
[3] Return top K documents
    Sorted by similarity score
```

**The `filter` parameter:**
```python
# Only search AML documents:
results = manager.similarity_search(
    query="KYC requirements",
    filter={"filename": "aml_policy.md"}
)
```

---

### Part 6: Search with Scores

```python
def similarity_search_with_scores(
    self, 
    query: str, 
    k: Optional[int] = None
) -> List[tuple[Document, float]]:
    """Search with relevance scores."""
    k = k or self.settings.top_k_results
    
    try:
        results = self.vector_store.similarity_search_with_score(
            query=query,
            k=k
        )
        logger.info("similarity_search_scored", 
                   query_preview=query[:50], 
                   results=len(results))
        return results
    except Exception as e:
        logger.error("search_error", error=str(e))
        raise
```

**Returns tuples:** `(Document, distance_score)`

```python
results = manager.similarity_search_with_scores("Basel III")
# [
#   (Document("Basel III requires..."), 0.15),  ← Lower = more similar
#   (Document("Capital ratios..."), 0.28),
#   (Document("AML policy..."), 0.72),          ← Less relevant
# ]
```

**Note:** ChromaDB returns DISTANCE (lower = better), not similarity (higher = better). That's why `rag_chain.py` converts it: `relevance = 1 - distance`.

---

### Part 7: Get Retriever

```python
def get_retriever(self, k: Optional[int] = None):
    """Get a retriever interface for the vector store."""
    k = k or self.settings.top_k_results
    return self.vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": k}
    )
```

**What's a Retriever?**

LangChain's standard interface for "get relevant documents":
```python
retriever = manager.get_retriever(k=5)
docs = retriever.invoke("What is Basel III?")
# Returns List[Document]
```

**Why use it?** Works with LangChain's chains and pipelines out of the box.

---

### Part 8: Collection Stats

```python
def get_collection_stats(self) -> Dict[str, Any]:
    """Get statistics about the collection."""
    try:
        collection = self.vector_store._collection
        count = collection.count()
        
        return {
            "collection_name": self.settings.collection_name,
            "document_count": count,
            "embedding_model": self.settings.embedding_model,
            "persist_directory": self.settings.chroma_persist_dir
        }
    except Exception as e:
        logger.error("stats_error", error=str(e))
        return {"error": str(e)}
```

**Accessing `_collection`:** The underscore means "private", but we need it for count. In production, you might wrap this more carefully.

---

### Part 9: Clear Collection

```python
def clear_collection(self) -> bool:
    """Clear all documents from the collection."""
    try:
        self.vector_store._client.delete_collection(self.settings.collection_name)
        self._vector_store = None  # Reset to force recreation
        logger.info("collection_cleared", collection=self.settings.collection_name)
        return True
    except Exception as e:
        logger.error("clear_error", error=str(e))
        return False
```

**Important:** After deleting, we set `_vector_store = None` so next access creates a fresh collection.

---

### Part 10: Singleton Pattern

```python
# Singleton instance
_vector_store_manager: Optional[VectorStoreManager] = None


def get_vector_store_manager() -> VectorStoreManager:
    """Get singleton vector store manager instance."""
    global _vector_store_manager
    if _vector_store_manager is None:
        _vector_store_manager = VectorStoreManager()
    return _vector_store_manager
```

**Why Singleton?**

```python
# BAD: Multiple connections
manager1 = VectorStoreManager()  # Opens ChromaDB
manager2 = VectorStoreManager()  # Opens ChromaDB again!

# GOOD: Single connection
manager1 = get_vector_store_manager()  # Opens ChromaDB
manager2 = get_vector_store_manager()  # Returns SAME instance
```

---

## Visual: How Search Works

```
                    Query: "capital requirements"
                              │
                              ▼
                    ┌─────────────────┐
                    │ OpenAI Embed    │
                    │ API Call        │
                    └────────┬────────┘
                              │
                    [0.23, -0.12, 0.89, ...]
                              │
                              ▼
        ┌─────────────────────────────────────────┐
        │              ChromaDB                    │
        │                                          │
        │  Vector 1: [0.21, -0.10, 0.85, ...] ✓   │ ← Close!
        │  Vector 2: [0.45, 0.33, -0.22, ...]     │
        │  Vector 3: [0.19, -0.15, 0.91, ...] ✓   │ ← Close!
        │  Vector 4: [-0.8, 0.12, 0.05, ...]      │
        │  ...                                     │
        └─────────────────────────────────────────┘
                              │
                              ▼
                    Top K Results (k=2):
                    [Document 1, Document 3]
```

---

## Interview Questions

**Q: Why use ChromaDB instead of Pinecone or FAISS?**

A: Trade-offs:
- **ChromaDB**: Simple, local, good for dev/small scale
- **Pinecone**: Managed service, scales infinitely, costs money
- **FAISS**: Facebook's library, extremely fast, no persistence built-in

**Q: What's the difference between L2 distance and cosine similarity?**

A: 
- **L2 (Euclidean)**: Straight-line distance between points
- **Cosine**: Angle between vectors (ignores magnitude)

For normalized embeddings (like OpenAI's), they're equivalent. Cosine is more intuitive for text.

**Q: How would you handle 10 million documents?**

A: 
1. Use a distributed vector DB (Pinecone, Weaviate, Qdrant)
2. Add indexes for filtering (by date, category)
3. Consider approximate nearest neighbor (ANN) algorithms
4. Batch embedding calls to reduce API costs

**Q: What happens if OpenAI's embedding API is down?**

A: Current code will throw an exception. Production improvements:
- Retry with exponential backoff
- Cache embeddings locally
- Have a fallback model (e.g., local sentence-transformers)

---

## Next Up

Section 5: `rag_chain.py` — the heart of the system, connecting retrieval to generation.
