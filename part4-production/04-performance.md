# Part 4, Section 4: Performance Optimization

## What You'll Learn
- Where RAG systems are slow
- Caching strategies
- Batching and async patterns

---

## Understanding RAG Latency

Typical query breakdown:

```
┌────────────────────────────────────────────────────────────┐
│ Total: ~2000ms                                              │
├────────────────────────────────────────────────────────────┤
│ Embed Query:    │████████│ 200ms         (10%)              │
│ Vector Search:  │██│ 50ms               (2.5%)              │
│ Fetch Chunks:   │██│ 50ms               (2.5%)              │
│ LLM Generation: │████████████████████████████│ 1700ms (85%) │
└────────────────────────────────────────────────────────────┘
```

**Key insight:** LLM generation dominates. But don't ignore the rest!

---

## Strategy 1: Caching

### Query Cache

Cache identical queries:

```python
from functools import lru_cache
import hashlib

# Simple in-memory cache
query_cache = {}

def get_cache_key(question: str, k: int) -> str:
    return hashlib.md5(f"{question}:{k}".encode()).hexdigest()

def query_with_cache(question: str, k: int = 5):
    cache_key = get_cache_key(question, k)
    
    if cache_key in query_cache:
        logger.info("cache_hit", question_preview=question[:30])
        return query_cache[cache_key]
    
    result = rag_chain.query(question, k)
    query_cache[cache_key] = result
    
    return result
```

### Embedding Cache

Don't re-embed the same text:

```python
from langchain.embeddings import CacheBackedEmbeddings
from langchain.storage import LocalFileStore

# Create cache
store = LocalFileStore("./cache/embeddings")

cached_embeddings = CacheBackedEmbeddings.from_bytes_store(
    underlying_embeddings=OpenAIEmbeddings(),
    document_embedding_cache=store,
    namespace="financerag"
)

# Now use cached_embeddings instead of OpenAIEmbeddings
```

**Result:** Second time you embed "Basel III capital requirements" = instant.

### Redis Cache for Production

```python
import redis
import json

redis_client = redis.Redis(host='localhost', port=6379, db=0)
CACHE_TTL = 3600  # 1 hour

def query_with_redis_cache(question: str, k: int = 5):
    cache_key = f"rag:{get_cache_key(question, k)}"
    
    # Try cache
    cached = redis_client.get(cache_key)
    if cached:
        return json.loads(cached)
    
    # Compute and cache
    result = rag_chain.query(question, k)
    redis_client.setex(cache_key, CACHE_TTL, json.dumps(result))
    
    return result
```

---

## Strategy 2: Batching

### Batch Embeddings

```python
# SLOW: One at a time
for doc in documents:
    embedding = embeddings.embed_query(doc.page_content)  # API call
    
# FAST: All at once
texts = [doc.page_content for doc in documents]
all_embeddings = embeddings.embed_documents(texts)  # Single API call
```

**Savings:** 100 documents = 100 API calls → 1 API call

### Batch Database Operations

```python
# SLOW: Insert one by one
for chunk in chunks:
    vector_store.add_documents([chunk])

# FAST: Batch insert
vector_store.add_documents(chunks)  # Single operation
```

---

## Strategy 3: Async Operations

### Async API Calls

```python
import asyncio
from langchain_openai import ChatOpenAI

async def query_async(question: str, k: int = 5):
    # Async embedding
    embedding = await embeddings.aembed_query(question)
    
    # Async search
    docs = await vector_store.asimilarity_search(question, k=k)
    
    # Async LLM call
    answer = await llm.ainvoke(prompt)
    
    return {"answer": answer, "sources": docs}

# Run multiple queries concurrently
async def query_batch(questions: List[str]):
    tasks = [query_async(q) for q in questions]
    results = await asyncio.gather(*tasks)
    return results
```

### FastAPI Async Endpoints

```python
@app.post("/query")
async def query_endpoint(request: QueryRequest):
    # This is already async - FastAPI handles concurrency
    result = await rag_chain.aquery(request.question)
    return result
```

---

## Strategy 4: Optimize Retrieval

### Reduce K When Possible

```python
# More docs = more tokens = slower generation
result = retriever.search(query, k=10)  # Slow, expensive

# Often 3-5 is enough
result = retriever.search(query, k=4)   # Faster, cheaper
```

### Use Metadata Filters

```python
# Instead of searching everything
results = db.similarity_search(query, k=5)

# Filter first, search less
results = db.similarity_search(
    query,
    k=5,
    filter={"document_type": "policy"}  # Smaller search space
)
```

### Approximate Search Settings

```python
# For ChromaDB with HNSW
# Increase ef_search for better recall (slower)
# Decrease ef_search for faster search (less accurate)

chroma_settings = Settings(
    chroma_db_impl="duckdb+parquet",
    anonymized_telemetry=False,
    allow_reset=True
)
```

---

## Strategy 5: Optimize Generation

### Shorter Context = Faster Response

```python
def optimize_context(docs: List[Document], max_chars: int = 4000):
    """Keep context short but relevant."""
    context = ""
    for doc in docs:
        if len(context) + len(doc.page_content) > max_chars:
            break
        context += doc.page_content + "\n\n"
    return context
```

### Faster Model

```
Model           Speed       Cost        Quality
─────────────────────────────────────────────────
gpt-4o          Slow        $$$         Best
gpt-4o-mini     Fast        $           Good
gpt-3.5-turbo   Faster      $           Okay
```

Use `gpt-4o-mini` for most RAG queries. Reserve `gpt-4o` for complex reasoning.

### Streaming Responses

Return results as they generate:

```python
from langchain_openai import ChatOpenAI
from fastapi.responses import StreamingResponse

llm = ChatOpenAI(model="gpt-4o-mini", streaming=True)

@app.post("/query/stream")
async def query_stream(request: QueryRequest):
    async def generate():
        async for chunk in llm.astream(prompt):
            yield chunk.content
    
    return StreamingResponse(generate(), media_type="text/plain")
```

**User experience:** Sees text appearing immediately instead of waiting 2+ seconds.

---

## Strategy 6: Pre-computation

### Pre-compute Common Queries

```python
# During off-hours, compute answers for common queries
common_queries = [
    "What is CET1?",
    "What are the AML requirements?",
    "Explain Basel III",
]

for query in common_queries:
    result = rag_chain.query(query)
    cache.set(query, result, ttl=86400)  # Cache for 24 hours
```

### Pre-summarize Documents

```python
# At ingestion time, create summaries
for doc in documents:
    summary = llm.summarize(doc.content)
    doc.metadata["summary"] = summary
    
# At query time, search summaries first
results = db.similarity_search(query, filter={"type": "summary"})
```

---

## Benchmarking

### Measure Before Optimizing

```python
import time
from statistics import mean, stdev

def benchmark_query(query: str, iterations: int = 10):
    latencies = []
    
    for _ in range(iterations):
        start = time.time()
        result = rag_chain.query(query)
        latencies.append(time.time() - start)
    
    return {
        "mean_ms": mean(latencies) * 1000,
        "std_ms": stdev(latencies) * 1000,
        "min_ms": min(latencies) * 1000,
        "max_ms": max(latencies) * 1000
    }

# Run benchmark
stats = benchmark_query("What is Basel III?")
print(f"Mean: {stats['mean_ms']:.0f}ms ± {stats['std_ms']:.0f}ms")
```

### Component-Level Timing

```python
def query_with_timing(question: str):
    timings = {}
    
    # Embed
    start = time.time()
    embedding = embeddings.embed_query(question)
    timings['embed_ms'] = (time.time() - start) * 1000
    
    # Search
    start = time.time()
    docs = vector_store.similarity_search(question)
    timings['search_ms'] = (time.time() - start) * 1000
    
    # Generate
    start = time.time()
    answer = llm.invoke(prompt)
    timings['generate_ms'] = (time.time() - start) * 1000
    
    timings['total_ms'] = sum(timings.values())
    
    return {"answer": answer, "timings": timings}
```

---

## Performance Checklist

### Quick Wins (Do First)

- [ ] Use `gpt-4o-mini` instead of `gpt-4o`
- [ ] Reduce K from 10 to 5
- [ ] Add query caching
- [ ] Batch embedding operations

### Medium Effort

- [ ] Add Redis cache
- [ ] Implement async endpoints
- [ ] Use streaming responses
- [ ] Add metadata filters

### Advanced

- [ ] Pre-compute common queries
- [ ] Fine-tune embedding model
- [ ] Use local embedding model
- [ ] Implement hybrid search with reranking

---

## Interview Questions

**Q: How would you reduce RAG latency from 3 seconds to under 1 second?**

A: Analyze where time is spent:
1. **If embedding slow:** Cache embeddings, use faster model
2. **If retrieval slow:** Reduce K, add filters, check index
3. **If generation slow:** Use faster model (gpt-4o-mini), reduce context, stream response

**Q: When would you use caching vs not?**

A: Cache when:
- Same queries repeat often
- Answers don't change frequently
- Latency matters more than freshness

Don't cache when:
- Every query is unique
- Real-time data matters
- Storage is limited

**Q: How do you handle high query volume?**

A:
1. Horizontal scaling (multiple replicas)
2. Aggressive caching
3. Request queuing
4. Rate limiting per user
5. Use managed vector DB (Pinecone) for scale

**Q: What's the trade-off between retrieval quality and speed?**

A:
- Higher K = better recall, more tokens, slower
- Reranking = better precision, extra latency
- Hybrid search = best of both, more complex

Find the sweet spot for your use case.

---

## Quick Reference

### LRU Cache
```python
from functools import lru_cache

@lru_cache(maxsize=1000)
def cached_query(question: str):
    return rag_chain.query(question)
```

### Async Pattern
```python
async def query_async(question: str):
    embedding = await embeddings.aembed_query(question)
    docs = await vector_store.asimilarity_search(question)
    answer = await llm.ainvoke(prompt)
    return answer
```

### Streaming
```python
async for chunk in llm.astream(prompt):
    yield chunk.content
```

### Timing
```python
import time
start = time.time()
# ... operation ...
duration_ms = (time.time() - start) * 1000
```

---

## 🎉 Part 4 Complete!

You now understand production patterns:

1. **Docker** — Containerization and deployment
2. **Observability** — Logging and monitoring
3. **Error Handling** — Resilient systems
4. **Performance** — Making it fast

**Next:** Part 5 — Interview Ready (Q&A, system design, common gotchas).
