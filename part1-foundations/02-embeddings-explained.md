# 1.2 Embeddings Explained

## The Big Idea

Computers are great with numbers. They're terrible with meaning.

The sentence "The bank approved my loan" and "My mortgage application was accepted" mean almost the same thing to a human. But to a computer doing string matching, they're completely different — no shared words.

**Embeddings solve this problem.**

An embedding converts text into a list of numbers (a "vector") that captures the *meaning* of the text. Similar meanings → similar numbers → can be compared mathematically.

---

## What Is a Vector?

A vector is just a list of numbers. That's it.

```python
# A simple 3-dimensional vector
[0.2, -0.5, 0.8]

# An OpenAI embedding (1536 dimensions)
[0.0023, -0.0142, 0.0087, ..., 0.0012]  # 1536 numbers
```

Each number represents something about the text — but not in a way humans can easily interpret. It's a learned representation.

Think of it like GPS coordinates. The numbers themselves (51.5074, -0.1278) don't mean much to you, but they precisely locate London. Embeddings are "semantic coordinates" — they locate text in meaning-space.

---

## How Embeddings Capture Meaning

### The Intuition

Imagine plotting words in 3D space where:
- X-axis = how "financial" vs "medical" something is
- Y-axis = how "positive" vs "negative" the sentiment is  
- Z-axis = how "formal" vs "casual" the language is

In this space:
- "The loan was approved" might be at (0.9, 0.7, 0.8) — financial, positive, formal
- "The mortgage was rejected" might be at (0.9, -0.6, 0.8) — financial, negative, formal
- "The surgery went well" might be at (-0.8, 0.7, 0.8) — medical, positive, formal

The first two are close on X and Z (both financial and formal) but far on Y (opposite sentiment).

**Real embeddings work the same way, but with 1536 dimensions instead of 3.** Each dimension captures some aspect of meaning that the model learned during training.

---

## The Math: Measuring Similarity

Once text is converted to vectors, we can measure how similar two pieces of text are.

### Cosine Similarity

The most common measure is **cosine similarity** — the cosine of the angle between two vectors.

```
Cosine Similarity = (A · B) / (||A|| × ||B||)
```

Where:
- A · B is the dot product (multiply corresponding elements, sum them up)
- ||A|| is the magnitude (length) of vector A

**Results range from -1 to 1:**
- 1.0 = identical meaning (vectors point same direction)
- 0.0 = unrelated (vectors are perpendicular)
- -1.0 = opposite meaning (vectors point opposite directions)

In practice, most text embeddings land between 0.0 and 1.0.

### Python Example

```python
import numpy as np

def cosine_similarity(vec_a, vec_b):
    """Calculate cosine similarity between two vectors."""
    dot_product = np.dot(vec_a, vec_b)
    magnitude_a = np.linalg.norm(vec_a)
    magnitude_b = np.linalg.norm(vec_b)
    return dot_product / (magnitude_a * magnitude_b)

# Example with simple vectors
vec1 = np.array([1, 2, 3])
vec2 = np.array([1, 2, 3.1])  # Very similar
vec3 = np.array([-1, -2, -3])  # Opposite

print(cosine_similarity(vec1, vec2))  # ~0.9998 (very similar)
print(cosine_similarity(vec1, vec3))  # -1.0 (opposite)
```

---

## Generating Embeddings in Practice

### Using OpenAI's API

```python
from openai import OpenAI

client = OpenAI()

def get_embedding(text: str) -> list[float]:
    """Convert text to an embedding vector."""
    response = client.embeddings.create(
        model="text-embedding-ada-002",
        input=text
    )
    return response.data[0].embedding

# Generate embeddings
query = "What is the loan approval process?"
doc1 = "The mortgage application procedure involves credit checks and income verification."
doc2 = "Our restaurant serves excellent Italian cuisine."

query_vec = get_embedding(query)
doc1_vec = get_embedding(doc1)
doc2_vec = get_embedding(doc2)

print(f"Query length: {len(query_vec)}")  # 1536

# Calculate similarities
sim1 = cosine_similarity(np.array(query_vec), np.array(doc1_vec))
sim2 = cosine_similarity(np.array(query_vec), np.array(doc2_vec))

print(f"Similarity to doc1 (relevant): {sim1:.4f}")  # ~0.85
print(f"Similarity to doc2 (irrelevant): {sim2:.4f}")  # ~0.70
```

The loan-related query is much more similar to the mortgage document than the restaurant document — even though they share no exact words.

---

## Key Embedding Models

| Model | Dimensions | Provider | Notes |
|-------|-----------|----------|-------|
| text-embedding-ada-002 | 1536 | OpenAI | Good balance of quality/cost |
| text-embedding-3-small | 1536 | OpenAI | Newer, slightly better |
| text-embedding-3-large | 3072 | OpenAI | Highest quality, more expensive |
| all-MiniLM-L6-v2 | 384 | HuggingFace | Free, runs locally, fast |
| bge-large-en | 1024 | HuggingFace | High quality open-source |

### Trade-offs

- **More dimensions** = captures more nuance, but uses more storage and is slower to search
- **OpenAI models** = easy to use, high quality, but costs money and requires API calls
- **Open-source models** = free, can run locally (privacy!), but may need more setup

---

## What Makes a Good Embedding?

### 1. Semantic Consistency
Similar meanings should produce similar vectors:
```
"car" ≈ "automobile" ≈ "vehicle"
```

### 2. Contextual Understanding
The same word in different contexts should produce different vectors:
```
"bank" (financial) ≠ "bank" (river)
```

Modern embedding models handle this well because they look at surrounding words.

### 3. Domain Awareness
A model trained on general web text might not understand finance-specific terms perfectly. This is why:
- Chunking matters (include enough context)
- You might fine-tune embedding models for specialized domains
- Testing with your actual data is important

---

## Common Pitfalls

### 1. Embedding Model Mismatch
If you embed documents with model A and embed queries with model B, similarity comparisons are meaningless. **Always use the same model for documents and queries.**

### 2. Text Too Long
Most embedding models have a maximum input length (e.g., 8191 tokens for ada-002). Longer text gets truncated. This is why we **chunk** documents.

### 3. Ignoring the Query-Document Gap
Users ask questions: "What is the policy on X?"
Documents state facts: "The policy on X states that..."

The phrasing is different. Good embedding models bridge this gap, but it's not perfect. This is why **hybrid search** (combining semantic + keyword) often works better.

---

## Visualizing Embeddings

You can't visualize 1536 dimensions, but you can use dimensionality reduction (PCA, t-SNE, UMAP) to project down to 2D or 3D.

```python
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

# Assume we have embeddings for many documents
embeddings = [...]  # List of 1536-dim vectors
labels = [...]  # Document labels

# Reduce to 2D
tsne = TSNE(n_components=2, random_state=42)
reduced = tsne.fit_transform(np.array(embeddings))

# Plot
plt.scatter(reduced[:, 0], reduced[:, 1])
for i, label in enumerate(labels):
    plt.annotate(label, (reduced[i, 0], reduced[i, 1]))
plt.show()
```

This helps you see if similar documents cluster together — a sanity check for your embeddings.

---

## How This Fits Into RAG

In a RAG system:

1. **Ingestion**: Each document chunk → embedding → stored in vector database
2. **Query**: User question → embedding → search vector database for similar embeddings
3. **Retrieval**: Return the original text of the most similar chunks
4. **Generation**: Pass chunks + question to LLM

The embedding is the bridge between human text and mathematical search.

---

## What's Next?

Now you understand embeddings — the magical conversion of text to searchable numbers.

But where do we store millions of these vectors? How do we search them efficiently? Regular databases aren't designed for this.

Enter: **Vector Databases**.

→ [1.3 Vector Databases & Similarity Search](./03-vector-databases.md)

---

## 💡 Interview Tip

> **Q: "What is an embedding and why do we use them in RAG?"**
>
> **A:** "An embedding is a numerical representation of text — typically a vector of hundreds or thousands of floating-point numbers. The key property is that semantically similar text produces similar vectors. We use embeddings in RAG because they enable semantic search: instead of matching keywords, we can find documents that are conceptually related to a user's question, even if they use different words. The user's query is embedded, then we search the vector store for document chunks with similar embeddings."

> **Q: "How do you measure similarity between embeddings?"**
>
> **A:** "The most common measure is cosine similarity, which calculates the cosine of the angle between two vectors. A value of 1 means identical direction (very similar), 0 means perpendicular (unrelated), and -1 means opposite. We use cosine similarity rather than Euclidean distance because it's normalized — it measures the angle regardless of vector magnitude, which is more robust for comparing text of different lengths."

