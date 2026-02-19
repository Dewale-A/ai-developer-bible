# Part 2, Section 4: Embedding Models

## What You'll Learn
- How embedding models turn text into vectors
- Comparing OpenAI vs open-source models
- Choosing the right model for your use case

---

## What Are Embeddings?

An embedding model converts text into a list of numbers (vector) that captures its meaning.

```
"The bank approved the loan"
        │
        ▼
[0.023, -0.156, 0.891, 0.045, ..., -0.234]
        │
        └── 1536 numbers (for text-embedding-3-small)
```

**Key insight:** Similar meanings → similar vectors.

```
"The bank approved the loan"     → [0.12, 0.45, -0.23, ...]
"The financial institution       → [0.11, 0.47, -0.21, ...]  ← Very close!
 granted the credit"

"I went fishing by the river     → [-0.67, 0.12, 0.89, ...]  ← Very different!
 bank"
```

---

## How Embedding Models Work

### Training Process (Simplified)

```
1. Feed model BILLIONS of text pairs
   
   Similar pairs:          Different pairs:
   ("cat", "kitten")       ("cat", "airplane")
   ("bank loan", "credit") ("bank loan", "river bank")

2. Model learns to:
   - Push similar texts CLOSER in vector space
   - Push different texts FARTHER apart

3. Result: A function that maps text → meaningful vectors
```

### The Math (Optional)

```
Cosine Similarity = (A · B) / (||A|| × ||B||)

Where:
- A · B = dot product (multiply corresponding elements, sum)
- ||A|| = magnitude (square root of sum of squares)

Result: -1 (opposite) to +1 (identical)
```

---

## Popular Embedding Models

### OpenAI Models

| Model | Dimensions | Max Tokens | Cost | Best For |
|-------|------------|------------|------|----------|
| `text-embedding-3-small` | 1536 | 8191 | $0.02/1M tokens | General use, cost-effective |
| `text-embedding-3-large` | 3072 | 8191 | $0.13/1M tokens | Higher accuracy needs |
| `text-embedding-ada-002` | 1536 | 8191 | $0.10/1M tokens | Legacy (still good) |

**FinanceRAG uses:** `text-embedding-3-small` — best balance of quality and cost.

### Open-Source Models

| Model | Dimensions | Runs Locally | Best For |
|-------|------------|--------------|----------|
| `all-MiniLM-L6-v2` | 384 | ✅ | Fast, lightweight |
| `all-mpnet-base-v2` | 768 | ✅ | Better quality |
| `e5-large-v2` | 1024 | ✅ | State-of-the-art open |
| `bge-large-en-v1.5` | 1024 | ✅ | Excellent for retrieval |
| `mxbai-embed-large` | 1024 | ✅ | New, very good |

---

## Using OpenAI Embeddings

### Basic Usage

```python
from langchain_openai import OpenAIEmbeddings

# Initialize
embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small",
    openai_api_key="sk-..."
)

# Embed a single text
vector = embeddings.embed_query("What is Basel III?")
print(len(vector))  # 1536

# Embed multiple texts (more efficient)
vectors = embeddings.embed_documents([
    "Basel III capital requirements",
    "AML compliance procedures",
    "Loan underwriting guidelines"
])
print(len(vectors))  # 3
print(len(vectors[0]))  # 1536 each
```

### Cost Calculation

```
1 million tokens ≈ 750,000 words ≈ 3,000 pages

For text-embedding-3-small at $0.02/1M tokens:
- 100 documents (avg 1000 words each) = ~133K tokens = $0.003
- 10,000 documents = $0.27
- 1 million documents = $27
```

**Tip:** Embeddings are one-time cost per document. Queries are cheap too!

---

## Using Open-Source Embeddings

### With Sentence Transformers

```python
from langchain_community.embeddings import HuggingFaceEmbeddings

# Initialize (downloads model on first use)
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-mpnet-base-v2"
)

# Same interface as OpenAI
vector = embeddings.embed_query("What is Basel III?")
print(len(vector))  # 768
```

### With Ollama (Local)

```python
from langchain_community.embeddings import OllamaEmbeddings

embeddings = OllamaEmbeddings(
    model="mxbai-embed-large"
)

vector = embeddings.embed_query("What is Basel III?")
```

---

## Choosing the Right Model

### Decision Matrix

| Factor | OpenAI | Open-Source |
|--------|--------|-------------|
| **Quality** | Excellent | Good to Excellent |
| **Speed** | Fast (API) | Depends on hardware |
| **Cost** | Pay per token | Free (compute cost) |
| **Privacy** | Data sent to API | Stays local |
| **Offline** | ❌ | ✅ |
| **Setup** | Easy | More complex |

### When to Use OpenAI

- ✅ Getting started quickly
- ✅ Don't want to manage infrastructure
- ✅ Data isn't sensitive
- ✅ Need consistent, reliable quality

### When to Use Open-Source

- ✅ Data privacy is critical (healthcare, finance)
- ✅ High volume (cost savings)
- ✅ Need offline capability
- ✅ Want to fine-tune for your domain

---

## Embedding Dimensions: Does Size Matter?

```
Higher dimensions = More information captured
                  = Larger storage
                  = Slower search

Lower dimensions  = Faster search
                  = Less storage
                  = May lose nuance
```

**Practical guidance:**
- 384-768: Fine for most use cases
- 1024-1536: Good balance
- 3072+: Only if you need maximum precision

**Storage calculation:**
```
1 million vectors × 1536 dimensions × 4 bytes/float
= 6.14 GB

With 384 dimensions:
= 1.54 GB
```

---

## Batching for Efficiency

### Bad: One at a Time

```python
# SLOW - one API call per document
vectors = []
for doc in documents:
    v = embeddings.embed_query(doc)
    vectors.append(v)
```

### Good: Batch Processing

```python
# FAST - one API call for all
vectors = embeddings.embed_documents(documents)
```

**OpenAI limits:** 2048 texts per batch, 8191 tokens per text.

### Chunked Batching

```python
def batch_embed(texts, batch_size=100):
    """Embed texts in batches."""
    all_vectors = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        vectors = embeddings.embed_documents(batch)
        all_vectors.extend(vectors)
    return all_vectors

# Embed 10,000 documents in batches of 100
vectors = batch_embed(documents, batch_size=100)
```

---

## Handling Long Texts

Embedding models have token limits (usually 512-8192).

### Strategy 1: Truncate

```python
# Simple but loses information
text = text[:8000]  # Rough character limit
vector = embeddings.embed_query(text)
```

### Strategy 2: Chunk First (Recommended)

```python
# Split into chunks, embed each
from langchain_text_splitters import RecursiveCharacterTextSplitter

splitter = RecursiveCharacterTextSplitter(chunk_size=1000)
chunks = splitter.split_text(long_document)
vectors = embeddings.embed_documents(chunks)
```

### Strategy 3: Pool Chunk Embeddings

```python
import numpy as np

# Embed chunks, then average
chunk_vectors = embeddings.embed_documents(chunks)
document_vector = np.mean(chunk_vectors, axis=0)
```

---

## Code in FinanceRAG

From `src/tools/vector_store.py`:

```python
from langchain_openai import OpenAIEmbeddings

class VectorStoreManager:
    def __init__(self):
        self.settings = get_settings()
        
        # Initialize embedding model
        self.embeddings = OpenAIEmbeddings(
            model=self.settings.embedding_model,  # "text-embedding-3-small"
            openai_api_key=self.settings.openai_api_key
        )
```

The embeddings object is passed to ChromaDB, which automatically:
1. Calls `embed_documents()` when adding documents
2. Calls `embed_query()` when searching

---

## Interview Questions

**Q: What's the difference between `embed_query()` and `embed_documents()`?**

A: Functionally identical for most models. The distinction exists because some models (like E5) use different prefixes:
- Query: "query: What is Basel III?"
- Document: "passage: Basel III is a framework..."

**Q: Can you use different embedding models for indexing vs querying?**

A: **No!** Vectors must be comparable. If you embed documents with Model A, you must query with Model A. Switching models requires re-indexing everything.

**Q: How do you handle multilingual documents?**

A: Use a multilingual embedding model:
- OpenAI's models handle 100+ languages
- Open-source: `paraphrase-multilingual-mpnet-base-v2`

**Q: What if your domain has specialized vocabulary?**

A: Options:
1. Fine-tune an open-source model on your domain
2. Use a larger model (captures more nuance)
3. Combine embeddings with keyword search (hybrid)

**Q: How do you evaluate embedding quality?**

A: 
1. **Retrieval metrics**: Recall@K, MRR, NDCG
2. **Manual inspection**: Do top results make sense?
3. **A/B testing**: Compare models on real queries

---

## Quick Reference

### OpenAI Setup
```python
from langchain_openai import OpenAIEmbeddings
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
```

### HuggingFace Setup
```python
from langchain_community.embeddings import HuggingFaceEmbeddings
embeddings = HuggingFaceEmbeddings(model_name="all-mpnet-base-v2")
```

### Embed & Compare
```python
v1 = embeddings.embed_query("bank loan")
v2 = embeddings.embed_query("credit facility")

# Cosine similarity
from numpy import dot
from numpy.linalg import norm
similarity = dot(v1, v2) / (norm(v1) * norm(v2))
print(f"Similarity: {similarity:.2%}")  # ~85%+
```

---

## Next Up

Section 5: Vector Stores Deep Dive — where embeddings live and how to search them.
