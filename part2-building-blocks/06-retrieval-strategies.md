# Part 2, Section 6: Retrieval Strategies

## What You'll Learn
- Beyond basic similarity search
- MMR, hybrid search, and reranking
- Choosing the right strategy for your use case

---

## Why Retrieval Strategy Matters

**The problem:** Basic similarity search can return redundant results.

```
Query: "What are the capital requirements?"

Basic Search Returns:
1. "Basel III CET1 requirement is 4.5%..."     ← Great!
2. "The CET1 capital requirement is 4.5%..."   ← Same info!
3. "CET1 ratio must be at least 4.5%..."       ← Still same!
4. "Tier 1 capital includes CET1..."           ← Different!
5. "Capital buffers add 2.5%..."               ← Different!
```

You wanted diverse information, but got 3 chunks saying the same thing.

---

## Strategy 1: Basic Similarity Search

**How it works:** Return K nearest vectors.

```python
# LangChain
results = db.similarity_search(query, k=5)

# Returns: Top 5 most similar, regardless of diversity
```

**Pros:**
- Simple, fast
- Returns most relevant results

**Cons:**
- Results may be redundant
- Misses diverse perspectives

**When to use:** Simple queries, small document sets.

---

## Strategy 2: Maximum Marginal Relevance (MMR)

**How it works:** Balance relevance AND diversity.

```
For each result:
  Score = λ × similarity(query, doc) - (1-λ) × max_similarity(doc, already_selected)
          ─────────────────────────     ──────────────────────────────────────────
               Relevance                        Avoid redundancy
```

**In plain English:**
1. First result = most similar to query
2. Second result = similar to query BUT different from first
3. Third result = similar to query BUT different from first two
4. ...

```python
# LangChain
retriever = db.as_retriever(
    search_type="mmr",
    search_kwargs={
        "k": 5,                    # Return 5 results
        "fetch_k": 20,             # Consider top 20 candidates
        "lambda_mult": 0.5         # Balance: 0=diversity, 1=relevance
    }
)

results = retriever.invoke("capital requirements")
```

**Parameters:**
- `fetch_k`: How many candidates to consider (higher = better diversity, slower)
- `lambda_mult`: Trade-off (0.5 is usually good)

**Pros:**
- More diverse results
- Better coverage of topic

**Cons:**
- Slightly slower
- May include less relevant results

**When to use:** Complex queries, summarization, research.

---

## Strategy 3: Hybrid Search

**How it works:** Combine vector search + keyword search.

```
Vector Search:  Finds semantically similar
                "bank loan" finds "credit facility"

Keyword Search: Finds exact matches
                "CET1" finds documents with "CET1"

Hybrid:         Best of both
                - Semantic understanding
                - Exact term matching
```

### Implementation with BM25

```python
from langchain.retrievers import BM25Retriever, EnsembleRetriever

# Keyword retriever (BM25)
bm25_retriever = BM25Retriever.from_documents(documents)
bm25_retriever.k = 5

# Vector retriever
vector_retriever = db.as_retriever(search_kwargs={"k": 5})

# Combine them
hybrid_retriever = EnsembleRetriever(
    retrievers=[bm25_retriever, vector_retriever],
    weights=[0.3, 0.7]  # 30% keyword, 70% vector
)

results = hybrid_retriever.invoke("Basel III CET1 requirements")
```

**How weights work:**
```
Final_score = 0.3 × BM25_score + 0.7 × Vector_score
```

**Pros:**
- Catches exact terms that embedding might miss
- More robust overall

**Cons:**
- More complex setup
- Two indexes to maintain

**When to use:** Technical domains, acronyms, specific terminology.

---

## Strategy 4: Reranking

**How it works:** Two-stage retrieval.

```
Stage 1: Fast retrieval (get top 20-50 candidates)
         Uses embedding similarity

Stage 2: Slow reranking (score each candidate)
         Uses a cross-encoder model
```

### Why Reranking Helps

**Bi-encoder (embedding):**
```
Query embedding:    [0.1, 0.2, ...]
Document embedding: [0.15, 0.18, ...]
Score: cosine_similarity(query, doc)  ← No interaction!
```

**Cross-encoder (reranker):**
```
Input: "Query: What is CET1? Document: CET1 is Common Equity Tier 1..."
Score: model("[Query] [Document]")    ← Sees both together!
```

Cross-encoders are more accurate but slower (can't pre-compute).

### Implementation

```python
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import CrossEncoderReranker
from langchain_community.cross_encoders import HuggingFaceCrossEncoder

# Initialize cross-encoder
model = HuggingFaceCrossEncoder(model_name="cross-encoder/ms-marco-MiniLM-L-6-v2")
reranker = CrossEncoderReranker(model=model, top_n=5)

# Wrap your retriever
base_retriever = db.as_retriever(search_kwargs={"k": 20})  # Get 20 candidates

reranking_retriever = ContextualCompressionRetriever(
    base_compressor=reranker,
    base_retriever=base_retriever
)

results = reranking_retriever.invoke("What is CET1?")  # Returns top 5 after reranking
```

**Pros:**
- Significantly better relevance
- Catches nuances bi-encoders miss

**Cons:**
- Slower (additional model inference)
- More complex setup

**When to use:** When precision matters most, production systems.

---

## Strategy 5: Query Expansion

**How it works:** Generate multiple query variations.

```
Original: "capital requirements"

Expanded:
- "capital requirements"
- "minimum capital ratio"
- "regulatory capital standards"
- "CET1 Tier 1 requirements"
```

### Implementation with LLM

```python
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

# Generate variations
expand_prompt = ChatPromptTemplate.from_template("""
Generate 3 alternative phrasings for this search query.
Return only the alternatives, one per line.

Query: {query}
""")

llm = ChatOpenAI(model="gpt-4o-mini")
chain = expand_prompt | llm

variations = chain.invoke({"query": "capital requirements"})
# "minimum capital ratio\nregulatory capital standards\nbank capital rules"

# Search with each variation
all_results = []
for q in [original_query] + variations.split("\n"):
    results = db.similarity_search(q, k=3)
    all_results.extend(results)

# Deduplicate and rank
unique_results = deduplicate(all_results)
```

**Pros:**
- Better recall (finds more relevant docs)
- Handles query ambiguity

**Cons:**
- More API calls
- Can introduce noise

**When to use:** Short queries, vague queries.

---

## Strategy 6: Self-Query (Metadata Filtering)

**How it works:** LLM extracts filter conditions from natural language.

```
Query: "Show me AML policies from 2023"

LLM extracts:
- Search query: "AML policies"
- Filter: {"year": 2023, "type": "policy"}
```

### Implementation

```python
from langchain.retrievers.self_query.base import SelfQueryRetriever
from langchain.chains.query_constructor.base import AttributeInfo

# Define filterable attributes
metadata_field_info = [
    AttributeInfo(
        name="source",
        description="The source file name",
        type="string"
    ),
    AttributeInfo(
        name="year",
        description="The year the document was published",
        type="integer"
    ),
    AttributeInfo(
        name="type",
        description="Document type: policy, regulation, guideline",
        type="string"
    )
]

retriever = SelfQueryRetriever.from_llm(
    llm=ChatOpenAI(model="gpt-4o-mini"),
    vectorstore=db,
    document_content_description="Financial regulatory documents",
    metadata_field_info=metadata_field_info
)

# Natural language query with implicit filter
results = retriever.invoke("Show me AML policies from 2023")
```

**Pros:**
- Natural language filtering
- Powerful for structured queries

**Cons:**
- Requires good metadata
- LLM can misinterpret

**When to use:** Rich metadata, user-facing search.

---

## Comparison Matrix

| Strategy | Speed | Precision | Diversity | Complexity |
|----------|-------|-----------|-----------|------------|
| Basic Similarity | ⚡⚡⚡ | Good | Low | Simple |
| MMR | ⚡⚡ | Good | High | Simple |
| Hybrid | ⚡⚡ | Better | Medium | Medium |
| Reranking | ⚡ | Best | Medium | Medium |
| Query Expansion | ⚡ | Better | High | Medium |
| Self-Query | ⚡⚡ | Good | Medium | Complex |

---

## Combining Strategies

The best systems combine multiple strategies:

```python
# Production retrieval pipeline

def retrieve(query: str, k: int = 5) -> List[Document]:
    # 1. Expand query (if short)
    if len(query.split()) < 5:
        queries = expand_query(query)
    else:
        queries = [query]
    
    # 2. Hybrid search for each query
    candidates = []
    for q in queries:
        bm25_results = bm25_retriever.invoke(q)
        vector_results = vector_retriever.invoke(q)
        candidates.extend(bm25_results + vector_results)
    
    # 3. Deduplicate
    unique = deduplicate(candidates)
    
    # 4. Rerank
    reranked = reranker.rerank(query, unique)
    
    # 5. Return top K
    return reranked[:k]
```

---

## FinanceRAG: Current Strategy

From `src/tools/vector_store.py`:

```python
def similarity_search_with_scores(self, query: str, k: int = 5):
    """Basic similarity search with scores."""
    return self.vector_store.similarity_search_with_score(query, k=k)
```

**Current:** Basic similarity search.

**Potential improvements:**
1. Add MMR for diversity
2. Add BM25 for hybrid search (regulatory terms, acronyms)
3. Add reranking for precision

---

## Interview Questions

**Q: When would you use MMR over basic similarity search?**

A: When you need diverse results:
- Summarization tasks (want different aspects)
- Research queries (explore the topic)
- When documents are very similar

**Q: What's the trade-off with hybrid search?**

A: 
- **Pro:** Catches exact terms embeddings might miss
- **Con:** More complex, two indexes, tuning weights

Worth it for technical domains with specific terminology.

**Q: How do you choose reranker top_n vs retriever k?**

A:
- Retriever k should be 3-5x reranker top_n
- Example: Retrieve 20, rerank to top 5
- More candidates = better reranking, but slower

**Q: How would you evaluate retrieval quality?**

A: Metrics:
- **Recall@K:** % of relevant docs in top K
- **MRR (Mean Reciprocal Rank):** Where does first relevant doc appear?
- **NDCG:** Measures ranking quality

Manual evaluation matters too — do results make sense?

**Q: What if retrieval is good but answers are bad?**

A: The problem is likely:
1. **Prompt engineering** — adjust instructions
2. **Context length** — too much/too little context
3. **LLM choice** — model may not be suitable
4. **Chunking** — chunks may not be coherent

---

## Quick Reference

### MMR
```python
retriever = db.as_retriever(
    search_type="mmr",
    search_kwargs={"k": 5, "fetch_k": 20, "lambda_mult": 0.5}
)
```

### Hybrid (Ensemble)
```python
from langchain.retrievers import EnsembleRetriever
retriever = EnsembleRetriever(
    retrievers=[bm25_retriever, vector_retriever],
    weights=[0.3, 0.7]
)
```

### Reranking
```python
from langchain.retrievers import ContextualCompressionRetriever
retriever = ContextualCompressionRetriever(
    base_compressor=reranker,
    base_retriever=base_retriever
)
```

---

## 🎉 Part 2 Complete!

You now understand all the building blocks:

1. **LangChain Architecture** — how pieces fit together
2. **Document Loading** — getting documents in
3. **Chunking Strategies** — splitting intelligently
4. **Embedding Models** — text to vectors
5. **Vector Stores** — storing and searching
6. **Retrieval Strategies** — finding the best results

**Next:** Part 4 — Production Patterns (Docker, observability, scaling).
