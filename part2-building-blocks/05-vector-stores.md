# Part 2, Section 5: Vector Stores Deep Dive

## What You'll Learn
- How vector databases store and index embeddings
- ChromaDB vs Pinecone vs FAISS vs others
- When to use which database

---

## What is a Vector Store?

A specialized database optimized for:
1. **Storing** high-dimensional vectors (embeddings)
2. **Searching** for similar vectors quickly
3. **Filtering** by metadata

```
Traditional DB:          Vector DB:
┌──────────────────┐     ┌──────────────────────────────────┐
│ id │ name │ age  │     │ id │ vector           │ metadata │
├────┼──────┼──────┤     ├────┼──────────────────┼──────────┤
│ 1  │ John │ 30   │     │ 1  │ [0.1, 0.2, ...]  │ {src: a} │
│ 2  │ Jane │ 25   │     │ 2  │ [0.3, -0.1, ...] │ {src: b} │
└──────────────────┘     └──────────────────────────────────┘
         │                            │
         ▼                            ▼
    WHERE age > 25              Find nearest to [0.2, 0.1, ...]
```

---

## The Search Problem

Finding similar vectors is computationally expensive.

### Brute Force (Exact)
```
Compare query to ALL vectors:
- 1 million vectors × 1536 dimensions
- = 1.5 billion comparisons
- = SLOW
```

### Approximate Nearest Neighbor (ANN)
```
Build an index structure:
- Group similar vectors together
- Only search relevant groups
- 95-99% accuracy, 100x faster
```

---

## Popular Vector Stores

### 1. ChromaDB (What FinanceRAG Uses)

**Type:** Embedded (runs in your process) or Client-Server

```python
from langchain_community.vectorstores import Chroma

db = Chroma(
    collection_name="my_docs",
    embedding_function=embeddings,
    persist_directory="./data/chroma"  # Saves to disk
)
```

**Pros:**
- ✅ Zero setup — just pip install
- ✅ Persistent storage built-in
- ✅ Good for development & small-medium scale
- ✅ Open source, free

**Cons:**
- ❌ Single machine only
- ❌ Not designed for billions of vectors

**Best for:** Development, prototypes, <1M vectors

---

### 2. FAISS (Facebook AI Similarity Search)

**Type:** Library (not a database)

```python
from langchain_community.vectorstores import FAISS

db = FAISS.from_documents(documents, embeddings)
db.save_local("faiss_index")

# Load later
db = FAISS.load_local("faiss_index", embeddings)
```

**Pros:**
- ✅ Extremely fast (C++ core)
- ✅ Battle-tested at Facebook scale
- ✅ Multiple index types for different trade-offs

**Cons:**
- ❌ No built-in persistence (you manage saving/loading)
- ❌ No metadata filtering (workarounds needed)
- ❌ Single machine

**Best for:** Speed-critical applications, research

---

### 3. Pinecone

**Type:** Managed cloud service

```python
from langchain_pinecone import PineconeVectorStore
import pinecone

pinecone.init(api_key="xxx", environment="us-west1-gcp")

db = PineconeVectorStore.from_documents(
    documents,
    embeddings,
    index_name="my-index"
)
```

**Pros:**
- ✅ Fully managed — no infrastructure
- ✅ Scales to billions of vectors
- ✅ Built-in filtering, namespaces
- ✅ Real-time updates

**Cons:**
- ❌ Costs money ($70+/month for production)
- ❌ Data leaves your infrastructure
- ❌ Vendor lock-in

**Best for:** Production at scale, teams without DevOps

---

### 4. Weaviate

**Type:** Self-hosted or cloud

```python
from langchain_weaviate.vectorstores import WeaviateVectorStore
import weaviate

client = weaviate.Client("http://localhost:8080")

db = WeaviateVectorStore.from_documents(
    documents,
    embeddings,
    client=client,
    index_name="Document"
)
```

**Pros:**
- ✅ GraphQL API
- ✅ Hybrid search (vector + keyword) built-in
- ✅ Good filtering capabilities
- ✅ Open source with cloud option

**Cons:**
- ❌ More complex setup
- ❌ Heavier resource usage

**Best for:** Complex queries, hybrid search needs

---

### 5. Qdrant

**Type:** Self-hosted or cloud

```python
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient

client = QdrantClient("localhost", port=6333)

db = QdrantVectorStore.from_documents(
    documents,
    embeddings,
    collection_name="my_docs",
    client=client
)
```

**Pros:**
- ✅ Written in Rust (fast!)
- ✅ Rich filtering
- ✅ Good documentation
- ✅ Active development

**Cons:**
- ❌ Newer, smaller community

**Best for:** Performance-sensitive, self-hosted

---

### 6. pgvector (PostgreSQL Extension)

**Type:** PostgreSQL extension

```python
from langchain_postgres import PGVector

db = PGVector.from_documents(
    documents,
    embeddings,
    connection="postgresql://user:pass@localhost/db",
    collection_name="my_docs"
)
```

**Pros:**
- ✅ Use existing Postgres infrastructure
- ✅ SQL + vectors together
- ✅ ACID transactions
- ✅ Familiar tooling

**Cons:**
- ❌ Not as fast as dedicated vector DBs
- ❌ Index management more manual

**Best for:** Already using Postgres, want unified stack

---

## Comparison Matrix

| Feature | ChromaDB | FAISS | Pinecone | Weaviate | Qdrant |
|---------|----------|-------|----------|----------|--------|
| Setup | Easy | Easy | Easy | Medium | Medium |
| Managed | ❌ | ❌ | ✅ | Optional | Optional |
| Scale | 1M | 1B | ∞ | 1B | 1B |
| Filtering | ✅ | ❌ | ✅ | ✅ | ✅ |
| Hybrid | ❌ | ❌ | ❌ | ✅ | ✅ |
| Cost | Free | Free | $70+/mo | Free/Paid | Free/Paid |
| Persistence | ✅ | Manual | ✅ | ✅ | ✅ |

---

## ChromaDB Deep Dive (FinanceRAG)

### Collections

Like tables in a database:

```python
# Create/get a collection
db = Chroma(
    collection_name="finance_docs",
    embedding_function=embeddings,
    persist_directory="./data/chroma"
)

# Each collection is independent
regulations_db = Chroma(collection_name="regulations", ...)
policies_db = Chroma(collection_name="policies", ...)
```

### Adding Documents

```python
from langchain_core.documents import Document

docs = [
    Document(
        page_content="Basel III requires...",
        metadata={"source": "basel.md", "page": 1}
    ),
    Document(
        page_content="AML policy states...",
        metadata={"source": "aml.md", "page": 1}
    )
]

# Add to collection
ids = db.add_documents(docs)
print(ids)  # ['abc123', 'def456']
```

### Similarity Search

```python
# Basic search
results = db.similarity_search(
    query="capital requirements",
    k=5  # Return top 5
)

for doc in results:
    print(doc.page_content[:100])
    print(doc.metadata)
```

### Search with Scores

```python
# Get similarity scores too
results = db.similarity_search_with_score(
    query="capital requirements",
    k=5
)

for doc, score in results:
    print(f"Score: {score:.4f}")  # Lower = more similar
    print(doc.page_content[:100])
```

**Note:** ChromaDB returns **distance** (lower = better), not similarity (higher = better).

### Metadata Filtering

```python
# Only search documents from specific source
results = db.similarity_search(
    query="capital requirements",
    k=5,
    filter={"source": "basel.md"}
)

# Multiple conditions (AND)
results = db.similarity_search(
    query="requirements",
    filter={
        "source": "basel.md",
        "page": {"$gt": 5}  # Page > 5
    }
)
```

**Filter operators:**
- `$eq` — equals (default)
- `$ne` — not equals
- `$gt`, `$gte` — greater than
- `$lt`, `$lte` — less than
- `$in` — in list

### Getting the Retriever

```python
# LangChain's retriever interface
retriever = db.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 5}
)

# Use in chains
docs = retriever.invoke("What is Basel III?")
```

### Persistence

```python
# Data auto-saves to persist_directory
db = Chroma(
    collection_name="my_docs",
    persist_directory="./data/chroma"  # Survives restarts!
)

# To clear:
db._client.delete_collection("my_docs")
```

---

## Index Types (Advanced)

### HNSW (Default in Most DBs)

**Hierarchical Navigable Small World**

```
        Top Layer (few nodes)
             │
        ┌────┴────┐
        │         │
    Layer 2    Layer 2
     │   │      │   │
    L3  L3     L3  L3    (more nodes)
    ...        ...
    Bottom Layer (all nodes)
```

- Start at top, navigate down
- ~O(log N) search time
- 95-99% recall

### IVF (FAISS)

**Inverted File Index**

```
Cluster 1: [vectors close to centroid 1]
Cluster 2: [vectors close to centroid 2]
...

Query → Find nearest centroids → Search those clusters only
```

- Good for very large datasets
- Tunable accuracy/speed trade-off

---

## Code in FinanceRAG

From `src/tools/vector_store.py`:

```python
class VectorStoreManager:
    def __init__(self):
        self.settings = get_settings()
        self.embeddings = OpenAIEmbeddings(
            model=self.settings.embedding_model,
            openai_api_key=self.settings.openai_api_key
        )
        self._vector_store: Optional[Chroma] = None
    
    @property
    def vector_store(self) -> Chroma:
        """Lazy initialization."""
        if self._vector_store is None:
            self._vector_store = Chroma(
                collection_name=self.settings.collection_name,
                embedding_function=self.embeddings,
                persist_directory=self.settings.chroma_persist_dir
            )
        return self._vector_store
    
    def similarity_search_with_scores(self, query: str, k: int = 5):
        return self.vector_store.similarity_search_with_score(query, k=k)
```

---

## Interview Questions

**Q: When would you choose Pinecone over ChromaDB?**

A: 
- Need to scale beyond millions of vectors
- Don't want to manage infrastructure
- Need high availability / SLA guarantees
- Budget allows ($70+/month)

**Q: What's the trade-off between exact and approximate search?**

A:
- **Exact**: 100% recall, O(N) time, doesn't scale
- **Approximate**: 95-99% recall, O(log N) time, scales to billions

For most RAG applications, approximate is fine — you're retrieving top-K anyway.

**Q: How would you migrate from ChromaDB to Pinecone?**

A:
1. Export documents and metadata from Chroma
2. Re-embed documents (or export existing vectors)
3. Upload to Pinecone in batches
4. Update application code to use Pinecone client
5. Test thoroughly before switching

**Q: How do you handle updates to documents?**

A: Options:
1. **Delete & re-add**: Simple, works for small updates
2. **Versioning**: Keep old version, add new with version metadata
3. **Upsert**: Update in place (not all DBs support)

**Q: What happens if your embedding model changes?**

A: You must **re-embed everything**. Vectors from different models are incompatible. Plan for this in production:
- Store original text, not just vectors
- Have a re-indexing pipeline ready

---

## Quick Reference

### ChromaDB
```python
from langchain_community.vectorstores import Chroma
db = Chroma(collection_name="docs", persist_directory="./data")
```

### FAISS
```python
from langchain_community.vectorstores import FAISS
db = FAISS.from_documents(docs, embeddings)
```

### Pinecone
```python
from langchain_pinecone import PineconeVectorStore
db = PineconeVectorStore.from_documents(docs, embeddings, index_name="idx")
```

### Common Operations
```python
# Add
db.add_documents(docs)

# Search
db.similarity_search("query", k=5)

# Search with filter
db.similarity_search("query", filter={"source": "file.md"})

# Get retriever
retriever = db.as_retriever(search_kwargs={"k": 5})
```

---

## Next Up

Section 6: Retrieval Strategies — going beyond basic similarity search.
