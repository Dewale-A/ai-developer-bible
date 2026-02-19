# 1.3 Vector Databases & Similarity Search

## The Problem

You've got 100,000 document chunks. Each is embedded as a 1536-dimension vector. A user asks a question.

Now what?

The naive approach: compare the query vector to all 100,000 document vectors, calculate cosine similarity for each, return the top 5.

This works. But it's O(n) — linear time. With millions of documents, every query takes seconds. That's not acceptable for production.

**Vector databases solve this with clever indexing and approximate search algorithms.**

---

## What Is a Vector Database?

A vector database is a specialized database designed to:

1. **Store** high-dimensional vectors alongside metadata
2. **Index** vectors for fast retrieval
3. **Search** for similar vectors efficiently (sub-linear time)
4. **Scale** to millions or billions of vectors

Think of it like a regular database, but instead of querying with SQL (`WHERE name = 'John'`), you query with a vector ("find the 5 vectors most similar to this one").

---

## How Vector Search Works (The Intuition)

### The Library Analogy

Imagine a library with 1 million books. You want to find books similar to one you're holding.

**Naive approach:** Walk through every shelf, compare every book. Takes forever.

**Smart approach:** The library organizes books into sections (Fiction, Science, History), then subsections (Sci-Fi, Fantasy, Mystery). You go directly to the relevant section and search there.

Vector databases do something similar. They organize vectors into clusters or trees, so searches only compare against a subset of vectors.

---

## Indexing Algorithms

### 1. IVF (Inverted File Index)

**How it works:**
1. Cluster all vectors into K groups (using K-means)
2. When searching, identify which cluster(s) the query is closest to
3. Only search vectors in those clusters

```
100,000 vectors → 100 clusters of ~1,000 vectors each
Query: identify 3 closest clusters → search 3,000 vectors instead of 100,000
```

**Trade-off:** Faster search, but might miss relevant vectors in neighboring clusters. You tune `nprobe` (how many clusters to check) to balance speed vs accuracy.

### 2. HNSW (Hierarchical Navigable Small World)

**How it works:**
- Build a multi-layer graph where each vector connects to its approximate nearest neighbors
- Higher layers have fewer connections (long-range jumps)
- Lower layers have more connections (fine-grained search)
- Search starts at the top, navigates down

Think of it like: fly across the country (high layer), drive to the neighborhood (middle layer), walk to the house (low layer).

**Trade-off:** Very fast queries, but uses more memory and slower to build.

### 3. Flat (Brute Force)

No indexing — compare query to every vector. 

**When to use:** Small datasets (<10,000 vectors) where exact results matter more than speed.

---

## Popular Vector Databases

| Database | Type | Best For | Notes |
|----------|------|----------|-------|
| **ChromaDB** | Embedded | Prototyping, small-medium projects | Easy setup, Python-native |
| **Pinecone** | Managed Cloud | Production at scale | Fully managed, pay-per-use |
| **Weaviate** | Self-hosted/Cloud | When you need graph + vector | Hybrid search built-in |
| **Milvus** | Self-hosted | Large scale, high performance | Open source, complex setup |
| **FAISS** | Library | Maximum control, local | Facebook's library, not a full DB |
| **pgvector** | Extension | Already using Postgres | Add vectors to existing DB |

### For FinanceRAG, We Use ChromaDB

Why?
- Easy to set up (single pip install)
- Good enough for thousands of documents
- Persists to disk
- Works well with LangChain
- Can migrate to bigger solutions later

---

## ChromaDB Deep Dive

### Installation

```bash
pip install chromadb
```

### Basic Usage

```python
import chromadb

# Create a client (persisted to disk)
client = chromadb.PersistentClient(path="./chroma_db")

# Create a collection (like a table)
collection = client.get_or_create_collection(
    name="financial_docs",
    metadata={"hnsw:space": "cosine"}  # Use cosine similarity
)

# Add documents
collection.add(
    ids=["doc1", "doc2", "doc3"],
    documents=[
        "The loan approval process requires credit verification.",
        "Mortgage rates are determined by market conditions.",
        "Investment portfolios should be diversified."
    ],
    metadatas=[
        {"source": "loan_policy.pdf", "page": 1},
        {"source": "mortgage_guide.pdf", "page": 5},
        {"source": "investment_101.pdf", "page": 3}
    ]
)

# Query
results = collection.query(
    query_texts=["How do I get approved for a loan?"],
    n_results=2
)

print(results['documents'])
# [['The loan approval process requires credit verification.',
#   'Mortgage rates are determined by market conditions.']]
```

### What Just Happened?

1. ChromaDB automatically embedded our documents (using a default model)
2. Stored vectors + metadata + original text
3. When we queried, it embedded our question and found similar vectors
4. Returned the original text of the matches

---

## Key Concepts

### Collections
Like tables in a regular database. You might have:
- `policy_documents` collection
- `customer_faq` collection
- `regulatory_filings` collection

### Metadata
Extra information stored alongside vectors. Crucial for:
- **Filtering**: "Only search documents from 2024"
- **Attribution**: "This answer came from loan_policy.pdf, page 5"
- **Debugging**: Tracking where chunks came from

```python
# Query with metadata filter
results = collection.query(
    query_texts=["loan requirements"],
    n_results=5,
    where={"source": "loan_policy.pdf"}  # Only this source
)
```

### Distance Metrics

How similarity is measured:

| Metric | Formula | Range | Use When |
|--------|---------|-------|----------|
| **Cosine** | 1 - cos(θ) | 0 to 2 | Text embeddings (most common) |
| **L2 (Euclidean)** | √Σ(a-b)² | 0 to ∞ | When magnitude matters |
| **Inner Product** | Σ(a×b) | -∞ to ∞ | Normalized vectors |

For text, **cosine** is almost always the right choice.

---

## Approximate vs Exact Search

Vector databases trade **accuracy** for **speed** using approximate nearest neighbor (ANN) search.

**Exact search:** Guaranteed to find the true top-K most similar vectors.
**Approximate search:** Finds vectors that are *probably* in the top-K, much faster.

In practice, ANN algorithms achieve 95-99% recall (finding 95-99% of the true top results) while being 10-100x faster.

For RAG, this is fine. If you're retrieving 5 documents and 1 of them is slightly suboptimal, the LLM can still generate a good answer.

---

## Scaling Considerations

### Small (< 100K vectors) — ChromaDB, FAISS
- Single machine is fine
- Embedded or local database
- Millisecond queries

### Medium (100K - 10M vectors) — Pinecone, Weaviate, Milvus
- Consider managed services
- May need dedicated infrastructure
- Still sub-second queries

### Large (> 10M vectors) — Pinecone, Milvus, custom solutions
- Distributed systems
- Significant infrastructure investment
- Query latency becomes a design consideration

---

## Common Operations

### Adding Vectors

```python
# With pre-computed embeddings
collection.add(
    ids=["id1"],
    embeddings=[[0.1, 0.2, ...]],  # Your vector
    documents=["Original text"],
    metadatas=[{"source": "file.pdf"}]
)

# Let ChromaDB compute embeddings
collection.add(
    ids=["id1"],
    documents=["Original text"],  # ChromaDB embeds this
    metadatas=[{"source": "file.pdf"}]
)
```

### Querying

```python
# By text (ChromaDB embeds the query)
results = collection.query(
    query_texts=["my question"],
    n_results=5
)

# By embedding (you provide the vector)
results = collection.query(
    query_embeddings=[[0.1, 0.2, ...]],
    n_results=5
)
```

### Updating

```python
collection.update(
    ids=["id1"],
    documents=["Updated text"],
    metadatas=[{"source": "file.pdf", "updated": True}]
)
```

### Deleting

```python
collection.delete(ids=["id1", "id2"])

# Or delete by metadata filter
collection.delete(where={"source": "old_file.pdf"})
```

---

## How This Fits Into FinanceRAG

In our FinanceRAG project:

1. **Ingestion**: Process documents → chunk → embed → store in ChromaDB
2. **Query**: User question → embed → search ChromaDB → get relevant chunks
3. **Generate**: Pass chunks to LLM for answer

The vector store is the "memory" of our RAG system.

---

## What's Next?

You understand embeddings (converting text to vectors) and vector stores (storing and searching vectors).

Now let's put it all together: the complete **Retrieval-Generation Pipeline** — from user question to final answer.

→ [1.4 The Retrieval-Generation Pipeline](./04-retrieval-generation-pipeline.md)

---

## 💡 Interview Tip

> **Q: "Why use a vector database instead of a regular database?"**
>
> **A:** "Traditional databases are optimized for exact matches — find rows where column X equals Y. Vector databases are optimized for similarity search — find vectors closest to a query vector in high-dimensional space. They use specialized indexing algorithms like HNSW or IVF that allow sub-linear search time, making it practical to search millions of vectors in milliseconds. For RAG, this means we can quickly find document chunks semantically similar to a user's question, even when they don't share exact keywords."

> **Q: "What trade-offs do approximate nearest neighbor algorithms make?"**
>
> **A:** "ANN algorithms trade a small amount of accuracy for significant speed improvements. Instead of guaranteeing the exact top-K results, they find results that are very likely to be in the top-K. In practice, algorithms like HNSW achieve 95-99% recall while being orders of magnitude faster than brute force. For RAG applications, this is an acceptable trade-off — we're usually retrieving 3-5 documents, and if one is slightly suboptimal, the LLM can still generate a good answer."

