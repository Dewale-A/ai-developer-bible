# Part 4, Section 2: Observability & Logging

## What You'll Learn
- Structured logging with structlog
- Key metrics to track in RAG systems
- Debugging retrieval and generation issues

---

## Why Observability Matters

**Without observability:**
```
User: "The answers are wrong"
You: "🤷 Let me check... somewhere..."
```

**With observability:**
```
User: "The answers are wrong"
You: "I see query X retrieved documents A, B, C with scores 0.3, 0.2, 0.1.
      The low relevance scores explain the poor answer.
      Let me adjust the chunking strategy."
```

---

## The Three Pillars

| Pillar | What It Is | Example |
|--------|------------|---------|
| **Logs** | Events that happened | "Query received", "Document chunked" |
| **Metrics** | Numbers over time | Latency, token count, error rate |
| **Traces** | Request journey | Query → Embed → Search → Generate |

---

## Structured Logging with structlog

**Unstructured (bad):**
```python
print(f"Query: {query}, Results: {len(results)}, Time: {elapsed}")
# Output: Query: What is Basel?, Results: 5, Time: 0.234
```

**Structured (good):**
```python
logger.info("query_complete", query=query, results=len(results), time=elapsed)
# Output: {"event": "query_complete", "query": "What is Basel?", "results": 5, "time": 0.234}
```

**Why structured?**
- Easy to parse (JSON)
- Easy to search (filter by field)
- Easy to aggregate (count events by type)

---

## structlog Setup

```python
import structlog

# Configure structlog
structlog.configure(
    processors=[
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.UnicodeDecoder(),
        structlog.processors.JSONRenderer()  # JSON output
    ],
    wrapper_class=structlog.stdlib.BoundLogger,
    context_class=dict,
    logger_factory=structlog.stdlib.LoggerFactory(),
    cache_logger_on_first_use=True,
)

logger = structlog.get_logger()
```

### Using the Logger

```python
# Basic logging
logger.info("server_started", host="0.0.0.0", port=8000)

# With context
logger.info("document_processed", 
    filename="policy.md",
    chunks=15,
    duration_ms=234
)

# Errors with exceptions
try:
    process_document(doc)
except Exception as e:
    logger.error("document_error", 
        filename=doc.name,
        error=str(e),
        exc_info=True  # Include stack trace
    )
```

---

## Key Events to Log in RAG

### 1. Document Ingestion

```python
def process_documents(source_path: str):
    logger.info("ingestion_started", source=source_path)
    
    chunks = processor.process_documents(source_path)
    
    logger.info("ingestion_complete",
        source=source_path,
        documents=len(source_files),
        chunks=len(chunks),
        duration_ms=elapsed
    )
```

### 2. Retrieval

```python
def similarity_search(query: str, k: int):
    start = time.time()
    
    results = self.vector_store.similarity_search_with_score(query, k)
    
    scores = [score for doc, score in results]
    
    logger.info("retrieval_complete",
        query_preview=query[:50],
        k=k,
        results=len(results),
        avg_score=sum(scores)/len(scores) if scores else 0,
        min_score=min(scores) if scores else 0,
        max_score=max(scores) if scores else 0,
        duration_ms=(time.time() - start) * 1000
    )
```

### 3. Generation

```python
def query(question: str, k: int):
    # ... retrieval ...
    
    start = time.time()
    answer = chain.invoke({"context": context, "question": question})
    
    logger.info("generation_complete",
        question_preview=question[:50],
        context_length=len(context),
        answer_length=len(answer),
        model=self.settings.llm_model,
        duration_ms=(time.time() - start) * 1000
    )
```

### 4. API Requests

```python
@app.middleware("http")
async def log_requests(request: Request, call_next):
    start = time.time()
    
    response = await call_next(request)
    
    logger.info("http_request",
        method=request.method,
        path=request.url.path,
        status=response.status_code,
        duration_ms=(time.time() - start) * 1000,
        client_ip=request.client.host
    )
    
    return response
```

---

## FinanceRAG Logging

From `src/tools/rag_chain.py`:

```python
logger.info("rag_query_complete", 
    question_preview=question[:50],
    docs_retrieved=len(docs),
    avg_score=round(sum(scores)/len(scores), 4)
)
```

**Output:**
```json
{
  "event": "rag_query_complete",
  "timestamp": "2024-02-19T18:30:00Z",
  "question_preview": "What are the capital requirements under Basel",
  "docs_retrieved": 5,
  "avg_score": 0.2345
}
```

---

## Key Metrics to Track

### Retrieval Metrics

| Metric | What It Measures | Alert If |
|--------|------------------|----------|
| `retrieval_latency_ms` | Time to search vectors | > 500ms |
| `avg_relevance_score` | Quality of retrieved docs | < 0.3 |
| `empty_results_rate` | Queries with no results | > 5% |

### Generation Metrics

| Metric | What It Measures | Alert If |
|--------|------------------|----------|
| `generation_latency_ms` | Time for LLM response | > 5000ms |
| `token_count` | Tokens used per request | Budget exceeded |
| `error_rate` | Failed generations | > 1% |

### System Metrics

| Metric | What It Measures | Alert If |
|--------|------------------|----------|
| `request_rate` | Requests per second | Unusual spike |
| `memory_usage` | Container memory | > 80% |
| `vector_store_size` | Number of documents | Unexpected drop |

---

## Debugging Common Issues

### Issue: Poor Answer Quality

**Check logs for:**
```python
# Low relevance scores
{"event": "retrieval_complete", "avg_score": 0.65}  # Distance, lower is better
# But if using similarity: 0.35 is low!

# Few documents retrieved
{"event": "retrieval_complete", "results": 1}  # Expected 5
```

**Debug steps:**
1. Check what documents were retrieved
2. Verify query embedding matches document embeddings
3. Try different chunking strategy

### Issue: Slow Responses

**Check logs for:**
```python
{"event": "retrieval_complete", "duration_ms": 2500}  # Slow retrieval
{"event": "generation_complete", "duration_ms": 8000}  # Slow LLM
```

**Debug steps:**
1. If retrieval slow: Check vector index, reduce K
2. If generation slow: Reduce context length, use faster model

### Issue: Errors Spiking

**Check logs for:**
```python
{"event": "query_error", "error": "Rate limit exceeded"}
{"event": "embedding_error", "error": "Connection timeout"}
```

**Debug steps:**
1. Check OpenAI status page
2. Implement retry logic with backoff
3. Add caching for repeated queries

---

## Adding Metrics with Prometheus

```python
from prometheus_client import Counter, Histogram, start_http_server

# Define metrics
QUERY_COUNTER = Counter('rag_queries_total', 'Total RAG queries')
QUERY_LATENCY = Histogram('rag_query_latency_seconds', 'Query latency')
RETRIEVAL_SCORE = Histogram('rag_retrieval_score', 'Retrieval relevance scores')

# Use in code
@QUERY_LATENCY.time()
def query(question: str):
    QUERY_COUNTER.inc()
    
    results = retriever.search(question)
    for doc, score in results:
        RETRIEVAL_SCORE.observe(score)
    
    # ... rest of query ...

# Start metrics server
start_http_server(9090)  # Prometheus scrapes this
```

---

## Log Aggregation

In production, send logs to a central system:

### Option 1: stdout + Container Orchestrator

```python
# Logs go to stdout
structlog.processors.JSONRenderer()

# Docker/K8s captures stdout
# Send to: Elasticsearch, CloudWatch, Datadog
```

### Option 2: Direct to Service

```python
import logging
from pythonjsonlogger import jsonlogger

# Send to Logstash/Fluentd
handler = logging.handlers.SocketHandler('logstash', 5000)
handler.setFormatter(jsonlogger.JsonFormatter())
```

---

## Request Tracing

Track a request through the entire pipeline:

```python
import uuid

@app.middleware("http")
async def add_request_id(request: Request, call_next):
    request_id = str(uuid.uuid4())[:8]
    
    # Add to logger context
    structlog.contextvars.bind_contextvars(request_id=request_id)
    
    response = await call_next(request)
    response.headers["X-Request-ID"] = request_id
    
    return response
```

**Now all logs include request_id:**
```json
{"event": "query_received", "request_id": "abc123"}
{"event": "retrieval_complete", "request_id": "abc123"}
{"event": "generation_complete", "request_id": "abc123"}
```

---

## Interview Questions

**Q: What metrics would you track for a RAG system?**

A: Key metrics:
1. **Latency**: Retrieval time, generation time, total time
2. **Quality**: Relevance scores, empty result rate
3. **Cost**: Token usage, API calls
4. **Errors**: Error rate by type, timeout rate

**Q: How would you debug a slow RAG query?**

A:
1. Check logs for timing breakdown (retrieval vs generation)
2. If retrieval slow: vector DB performance, index issues
3. If generation slow: context too long, model overloaded
4. Check for cold start issues (first query slower)

**Q: How do you handle sensitive data in logs?**

A:
1. Never log full queries/answers (might contain PII)
2. Log previews: `question[:50]`
3. Mask sensitive fields
4. Set appropriate retention policies

**Q: What's the difference between logs and traces?**

A:
- **Logs**: Individual events ("query received")
- **Traces**: Connected journey (request → embed → search → generate → respond)
- Traces help understand the flow; logs help understand each step

---

## Quick Reference

### structlog Setup
```python
import structlog
logger = structlog.get_logger()
logger.info("event_name", key="value", number=123)
```

### Key Events
```python
logger.info("ingestion_complete", docs=10, chunks=150)
logger.info("retrieval_complete", results=5, avg_score=0.85)
logger.info("generation_complete", tokens=500, duration_ms=1200)
logger.error("query_error", error=str(e), exc_info=True)
```

### Timing Helper
```python
import time

start = time.time()
# ... do work ...
duration_ms = (time.time() - start) * 1000
```

---

## Next Up

Section 3: Error Handling & Edge Cases — building resilient RAG systems.
