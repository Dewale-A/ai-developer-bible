# Part 3, Section 6: API Endpoints

## What You'll Learn
- Building REST APIs with FastAPI
- Request/response validation with Pydantic
- Production-ready endpoint patterns

---

## Why FastAPI?

| Feature | Benefit |
|---------|---------|
| Auto-documentation | Swagger UI generated free |
| Type hints | Pydantic validates requests automatically |
| Async support | Non-blocking I/O for LLM calls |
| Easy to learn | Feels like Flask, performs like Node.js |

---

## File: `src/api/main.py`

### Part 1: Imports & App Setup

```python
from typing import List, Optional, Dict, Any
from pathlib import Path
import structlog

from fastapi import FastAPI, HTTPException, UploadFile, File, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from src.config.settings import get_settings
from src.tools.document_processor import DocumentProcessor
from src.tools.vector_store import get_vector_store_manager
from src.tools.rag_chain import get_rag_chain

logger = structlog.get_logger()
```

**Key imports:**

| Import | Purpose |
|--------|---------|
| `FastAPI` | The web framework |
| `HTTPException` | Return proper error responses |
| `BaseModel` | Define request/response schemas |
| `CORSMiddleware` | Allow cross-origin requests |

---

```python
# Initialize FastAPI app
app = FastAPI(
    title="FinanceRAG API",
    description="Production-grade RAG system for financial document Q&A",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)
```

**What this creates:**
- `GET /docs` → Swagger UI (interactive)
- `GET /redoc` → ReDoc (pretty documentation)
- `GET /openapi.json` → OpenAPI spec

---

```python
# CORS middleware - allow requests from any origin
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],        # Any domain can call this API
    allow_credentials=True,
    allow_methods=["*"],        # GET, POST, PUT, DELETE, etc.
    allow_headers=["*"],
)
```

**Why CORS?** Without it, browsers block requests from other domains. For development, `"*"` is fine. In production, specify exact domains:
```python
allow_origins=["https://myapp.com", "https://admin.myapp.com"]
```

---

### Part 2: Request/Response Models

```python
class QueryRequest(BaseModel):
    """Request model for RAG queries."""
    question: str = Field(..., description="The question to answer", min_length=3)
    k: Optional[int] = Field(default=5, description="Number of documents to retrieve", ge=1, le=20)
```

**Field validation:**

| Attribute | Meaning |
|-----------|---------|
| `...` | Required field |
| `min_length=3` | At least 3 characters |
| `ge=1, le=20` | Between 1 and 20 |

**What happens with invalid input?**
```json
// Request
{"question": "Hi"}  // Too short!

// Response (automatic!)
{
  "detail": [
    {
      "loc": ["body", "question"],
      "msg": "ensure this value has at least 3 characters",
      "type": "value_error"
    }
  ]
}
```

---

```python
class SourceDocument(BaseModel):
    """Source document information."""
    filename: str
    chunk_index: int
    relevance_score: float
    preview: str


class QueryResponse(BaseModel):
    """Response model for RAG queries."""
    answer: str
    sources: List[SourceDocument]
    query: str
    documents_retrieved: int
    model: Optional[str] = None
    error: Optional[str] = None
```

**Why define response models?**
1. Documentation shows exact response shape
2. Pydantic serializes objects automatically
3. Type hints help IDE catch bugs

---

### Part 3: Health Check Endpoint

```python
@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint with system status."""
    try:
        settings = get_settings()
        vector_store = get_vector_store_manager()
        stats = vector_store.get_collection_stats()
        
        return HealthResponse(
            status="healthy",
            collection_stats=stats,
            settings={
                "llm_model": settings.llm_model,
                "embedding_model": settings.embedding_model,
                "chunk_size": settings.chunk_size,
                "top_k": settings.top_k_results
            }
        )
    except Exception as e:
        return HealthResponse(
            status="unhealthy",
            collection_stats={"error": str(e)},
            settings={}
        )
```

**Anatomy of an endpoint:**

```python
@app.get("/health", response_model=HealthResponse)
#     │      │               │
#     │      │               └── Response type (for docs + validation)
#     │      └── URL path
#     └── HTTP method

async def health_check():  # async = can await I/O operations
```

**Why health check?**
- Load balancers ping this to know if server is up
- Kubernetes uses it for readiness probes
- Quick sanity check that database is connected

---

### Part 4: Query Endpoint (The Main One!)

```python
@app.post("/query", response_model=QueryResponse)
async def query_documents(request: QueryRequest):
    """
    Query the RAG system with a question.
    
    Returns an answer based on the ingested financial documents,
    along with source references.
    """
    try:
        rag_chain = get_rag_chain()
        result = rag_chain.query(request.question, k=request.k)
        
        return QueryResponse(
            answer=result["answer"],
            sources=[SourceDocument(**s) for s in result["sources"]],
            query=result["query"],
            documents_retrieved=result["documents_retrieved"],
            model=result.get("model"),
            error=result.get("error")
        )
    except Exception as e:
        logger.error("query_error", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))
```

**Request flow:**

```
POST /query
Content-Type: application/json

{"question": "What is Basel III?", "k": 5}
          │
          ▼
    QueryRequest(question="...", k=5)  ← Pydantic validates
          │
          ▼
    rag_chain.query(question, k)       ← Business logic
          │
          ▼
    QueryResponse(answer=..., sources=...) ← Pydantic serializes
          │
          ▼
    200 OK
    {"answer": "...", "sources": [...]}
```

**The `**s` trick:**
```python
# source dict
s = {"filename": "a.md", "chunk_index": 0, "relevance_score": 0.8, "preview": "..."}

# **s unpacks it:
SourceDocument(**s)
# Equivalent to:
SourceDocument(filename="a.md", chunk_index=0, relevance_score=0.8, preview="...")
```

---

### Part 5: Document Ingestion Endpoint

```python
@app.post("/ingest", response_model=IngestResponse)
async def ingest_documents(request: IngestRequest):
    """
    Ingest documents from a file or directory path.
    """
    try:
        processor = DocumentProcessor()
        vector_store = get_vector_store_manager()
        
        # Process documents
        chunks = processor.process_documents(request.source_path)
        
        if not chunks:
            return IngestResponse(
                success=False,
                documents_processed=0,
                chunks_created=0,
                message="No documents found to process"
            )
        
        # Add to vector store
        ids = vector_store.add_documents(chunks)
        
        # Count unique source files
        source_files = set(chunk.metadata.get("filename", "") for chunk in chunks)
        
        return IngestResponse(
            success=True,
            documents_processed=len(source_files),
            chunks_created=len(ids),
            message=f"Successfully ingested {len(source_files)} documents"
        )
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
```

**Error handling pattern:**
- `FileNotFoundError` → 404 (Not Found)
- Other exceptions → 500 (Server Error)

---

### Part 6: File Upload Endpoint

```python
@app.post("/ingest/upload", response_model=IngestResponse)
async def upload_and_ingest(file: UploadFile = File(...)):
    """
    Upload and ingest a single document file.
    """
    try:
        # Save uploaded file temporarily
        temp_path = Path(f"/tmp/{file.filename}")
        
        with open(temp_path, "wb") as f:
            content = await file.read()  # async read!
            f.write(content)
        
        # Process the file
        processor = DocumentProcessor()
        vector_store = get_vector_store_manager()
        
        chunks = processor.process_documents(str(temp_path))
        ids = vector_store.add_documents(chunks)
        
        # Clean up temp file
        temp_path.unlink()
        
        return IngestResponse(
            success=True,
            documents_processed=1,
            chunks_created=len(ids),
            message=f"Successfully ingested {file.filename}"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
```

**File upload handling:**
```python
file: UploadFile = File(...)
#        │             │
#        │             └── Required file parameter
#        └── FastAPI's file upload type
```

**Usage:**
```bash
curl -X POST http://localhost:8000/ingest/upload \
  -F "file=@my_document.pdf"
```

---

### Part 7: Utility Endpoints

```python
@app.get("/stats")
async def get_stats():
    """Get vector store statistics."""
    vector_store = get_vector_store_manager()
    return vector_store.get_collection_stats()


@app.delete("/collection")
async def clear_collection():
    """Clear all documents from the collection."""
    vector_store = get_vector_store_manager()
    success = vector_store.clear_collection()
    
    if success:
        return {"message": "Collection cleared successfully"}
    else:
        raise HTTPException(status_code=500, detail="Failed to clear collection")
```

**HTTP methods matter:**
- `GET /stats` → Read data (safe, cacheable)
- `DELETE /collection` → Destructive action

---

## All Endpoints Summary

| Method | Path | Purpose |
|--------|------|---------|
| GET | `/` | API info |
| GET | `/health` | System status |
| GET | `/docs` | Swagger UI |
| POST | `/query` | Ask a question |
| POST | `/chat` | Query with history |
| POST | `/ingest` | Add docs from path |
| POST | `/ingest/upload` | Upload & add file |
| GET | `/stats` | Collection stats |
| DELETE | `/collection` | Clear all docs |

---

## Testing with cURL

```bash
# Health check
curl http://localhost:8000/health

# Query
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "What is Basel III?"}'

# Ingest documents
curl -X POST http://localhost:8000/ingest \
  -H "Content-Type: application/json" \
  -d '{"source_path": "./sample_docs"}'

# Upload a file
curl -X POST http://localhost:8000/ingest/upload \
  -F "file=@policy.pdf"

# Get stats
curl http://localhost:8000/stats

# Clear collection (careful!)
curl -X DELETE http://localhost:8000/collection
```

---

## Interview Questions

**Q: Why use FastAPI over Flask?**

A:
- Built-in async support (better for I/O-bound tasks like LLM calls)
- Automatic validation with Pydantic
- Auto-generated OpenAPI docs
- Type hints everywhere = fewer bugs

**Q: What's the difference between `@app.get` and `@app.post`?**

A:
- **GET**: Retrieve data, no body, can be cached, bookmarkable
- **POST**: Send data in body, not cached, for creating/updating

**Q: How would you add authentication?**

A:
```python
from fastapi import Depends, Header

def verify_api_key(x_api_key: str = Header(...)):
    if x_api_key != "secret123":
        raise HTTPException(status_code=401, detail="Invalid API key")

@app.post("/query")
async def query(request: QueryRequest, _: str = Depends(verify_api_key)):
    # Only runs if API key is valid
    ...
```

**Q: How would you handle rate limiting?**

A:
```python
from slowapi import Limiter
from slowapi.util import get_remote_address

limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter

@app.post("/query")
@limiter.limit("10/minute")
async def query(request: QueryRequest, request_obj: Request):
    ...
```

**Q: How would you add request logging?**

A:
```python
@app.middleware("http")
async def log_requests(request: Request, call_next):
    start = time.time()
    response = await call_next(request)
    duration = time.time() - start
    
    logger.info("request",
        path=request.url.path,
        method=request.method,
        status=response.status_code,
        duration_ms=round(duration * 1000)
    )
    return response
```

---

## Running the Server

```bash
# Development (with auto-reload)
uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000

# Production (multiple workers)
uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --workers 4

# Or use the CLI
python main.py serve
```

---

## 🎉 Part 3 Complete!

You now understand the entire FinanceRAG codebase:

1. **Settings** — Configuration management
2. **Document Processor** — Loading and chunking
3. **Vector Store** — ChromaDB operations
4. **RAG Chain** — Retrieval + generation
5. **API** — REST endpoints

**Next:** Part 4 will cover production patterns — Docker, monitoring, scaling.
