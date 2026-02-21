# Part 4, Section 5: API Deployment for Agent Systems

## Why Add an API Layer?

CLI tools are great for development and batch processing, but real-world systems need:

```
CLI Only:                          With API:
┌─────────────────┐               ┌─────────────────┐
│  Terminal User  │               │   Web App       │
│  (one at a time)│               │   Mobile App    │
│                 │               │   Other Services│
└────────┬────────┘               │   Webhooks      │
         │                        └────────┬────────┘
         ▼                                 ▼
┌─────────────────┐               ┌─────────────────┐
│   python main.py│               │  FastAPI Server │
│                 │               │  (always running)│
└────────┬────────┘               └────────┬────────┘
         │                                 │
         ▼                                 ▼
┌─────────────────┐               ┌─────────────────┐
│   CrewAI Agents │               │   CrewAI Agents │
└─────────────────┘               └─────────────────┘
```

**Key Benefits:**
- **Integration** - Other systems can trigger agent workflows
- **Scalability** - Handle multiple concurrent requests
- **Monitoring** - Health checks, metrics endpoints
- **Async Processing** - Long-running jobs don't block clients
- **Documentation** - Auto-generated Swagger/OpenAPI docs

---

## What to Build: API Architecture

### Core Endpoints Pattern

Every agent system should expose:

| Endpoint | Purpose | Why |
|----------|---------|-----|
| `GET /health` | Health check | Load balancers, k8s probes |
| `GET /items` | List processable items | UI listing, job discovery |
| `GET /items/{id}` | Get item details | Pre-flight validation |
| `POST /process` | Sync processing | Simple integrations |
| `POST /process/async` | Async processing | Long-running jobs |
| `GET /jobs/{id}` | Job status | Polling for completion |

### Request/Response Design

Always use Pydantic models for type safety and auto-documentation:

```python
from pydantic import BaseModel, Field
from typing import Optional

class ProcessRequest(BaseModel):
    """Request to process an item through the agent workflow."""
    item_id: str = Field(..., description="ID of item to process")
    verbose: bool = Field(default=False, description="Enable detailed output")
    
class ProcessResponse(BaseModel):
    """Response from processing."""
    item_id: str
    status: str  # completed, failed
    duration_seconds: float
    result_summary: str
    output_file: Optional[str] = None
```

---

## How to Implement: FastAPI + CrewAI

### Step 1: Project Structure

Add an `api/` module to your existing structure:

```
your_project/
├── main.py              # CLI entry point
├── src/
│   ├── api/             # NEW: API layer
│   │   ├── __init__.py
│   │   └── main.py      # FastAPI app
│   ├── agents/
│   ├── tasks/
│   ├── tools/
│   └── crew.py
├── Dockerfile           # NEW: Container
└── docker-compose.yml   # NEW: Orchestration
```

### Step 2: FastAPI Application

```python
# src/api/main.py
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from datetime import datetime
import uuid

from src.crew import YourCrew
from src.models.your_model import YourModel

app = FastAPI(
    title="Your Agent System API",
    description="Multi-agent AI system for X",
    version="1.0.0",
)

# CORS for web clients
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory job tracking (use Redis in production)
jobs: dict[str, dict] = {}


@app.get("/api/v1/health")
async def health_check():
    """Health check for load balancers and k8s."""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "version": "1.0.0",
    }


@app.post("/api/v1/process")
async def process_sync(request: ProcessRequest):
    """
    Synchronous processing - blocks until complete.
    Use for short jobs (<30 seconds).
    """
    start = datetime.now()
    
    # Load your data model
    model = YourModel.from_json(f"data/{request.item_id}.json")
    
    # Run the crew
    crew = YourCrew(model, verbose=request.verbose)
    result = crew.run()
    
    duration = (datetime.now() - start).total_seconds()
    
    return {
        "item_id": request.item_id,
        "status": "completed",
        "duration_seconds": duration,
        "result_summary": str(result)[:500],
    }


@app.post("/api/v1/process/async")
async def process_async(request: ProcessRequest, background_tasks: BackgroundTasks):
    """
    Asynchronous processing - returns immediately with job ID.
    Poll /api/v1/jobs/{job_id} for status.
    """
    job_id = str(uuid.uuid4())
    
    jobs[job_id] = {
        "status": "pending",
        "item_id": request.item_id,
        "created_at": datetime.now().isoformat(),
    }
    
    # Run in background thread
    background_tasks.add_task(run_job, job_id, request)
    
    return {
        "job_id": job_id,
        "status": "pending",
        "message": "Poll /api/v1/jobs/{job_id} for status",
    }


def run_job(job_id: str, request: ProcessRequest):
    """Background job runner."""
    jobs[job_id]["status"] = "processing"
    
    try:
        model = YourModel.from_json(f"data/{request.item_id}.json")
        crew = YourCrew(model, verbose=request.verbose)
        result = crew.run()
        
        jobs[job_id]["status"] = "completed"
        jobs[job_id]["result"] = str(result)[:500]
    except Exception as e:
        jobs[job_id]["status"] = "failed"
        jobs[job_id]["error"] = str(e)


@app.get("/api/v1/jobs/{job_id}")
async def get_job_status(job_id: str):
    """Check status of async job."""
    if job_id not in jobs:
        raise HTTPException(404, f"Job not found: {job_id}")
    return jobs[job_id]
```

### Step 3: Dockerfile

```dockerfile
FROM python:3.11-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source
COPY . .

# Expose port
EXPOSE 8000

# Run API server
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

### Step 4: Docker Compose

```yaml
version: '3.8'

services:
  agent-api:
    build: .
    ports:
      - "8000:8000"
    environment:
      - OPENAI_API_KEY=${OPENAI_API_KEY}
    volumes:
      - ./data:/app/data:ro
      - ./output:/app/output
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/api/v1/health"]
      interval: 30s
      timeout: 10s
      retries: 3
```

---

## Sync vs Async: When to Use Each

### Use Sync (`POST /process`) When:
- Processing takes < 30 seconds
- Client can wait for response
- Simple request-response flow
- Internal service-to-service calls

### Use Async (`POST /process/async`) When:
- Processing takes > 30 seconds (agent workflows often do)
- Client shouldn't block (mobile apps, UIs)
- Need progress tracking
- Webhook callbacks on completion

```python
# Client-side async pattern
import httpx
import time

# Submit job
resp = httpx.post("http://api/process/async", json={"item_id": "ITEM001"})
job_id = resp.json()["job_id"]

# Poll for completion
while True:
    status = httpx.get(f"http://api/jobs/{job_id}").json()
    
    if status["status"] == "completed":
        print("Done!", status["result"])
        break
    elif status["status"] == "failed":
        print("Error:", status["error"])
        break
    
    time.sleep(5)  # Poll every 5 seconds
```

---

## Production Readiness Checklist

### Tier 1: MVP/Demo (What We Built)
- [x] FastAPI with Pydantic models
- [x] Health check endpoint
- [x] Sync + async processing
- [x] In-memory job tracking
- [x] Dockerfile + docker-compose
- [x] Basic tests
- [x] Swagger docs (auto-generated)

### Tier 2: Internal Production
- [ ] API key authentication
- [ ] Rate limiting
- [ ] Structured logging (JSON)
- [ ] Redis for job queue (multi-instance)
- [ ] PostgreSQL for job persistence
- [ ] Retry logic for transient failures

### Tier 3: Enterprise Scale
- [ ] OAuth2/JWT authentication
- [ ] Prometheus metrics
- [ ] OpenTelemetry tracing
- [ ] Kubernetes deployment
- [ ] Horizontal auto-scaling
- [ ] Circuit breakers
- [ ] Dead letter queue for failed jobs

---

## Common Patterns

### 1. Webhook Callbacks

Instead of polling, have the API call back when done:

```python
class AsyncRequest(BaseModel):
    item_id: str
    callback_url: Optional[str] = None  # POST result here when done

def run_job_with_callback(job_id: str, request: AsyncRequest):
    # ... run crew ...
    
    if request.callback_url:
        httpx.post(request.callback_url, json={
            "job_id": job_id,
            "status": "completed",
            "result": result,
        })
```

### 2. Streaming Progress

For long jobs, stream progress updates:

```python
from fastapi.responses import StreamingResponse
import json

@app.post("/api/v1/process/stream")
async def process_stream(request: ProcessRequest):
    async def generate():
        yield json.dumps({"status": "starting"}) + "\n"
        
        # ... each agent completes ...
        yield json.dumps({"status": "agent_1_complete"}) + "\n"
        yield json.dumps({"status": "agent_2_complete"}) + "\n"
        
        yield json.dumps({"status": "completed", "result": "..."}) + "\n"
    
    return StreamingResponse(generate(), media_type="application/x-ndjson")
```

### 3. Batch Processing

Process multiple items in one request:

```python
@app.post("/api/v1/process/batch")
async def process_batch(requests: list[ProcessRequest]):
    job_ids = []
    for req in requests:
        job_id = str(uuid.uuid4())
        # ... queue each job ...
        job_ids.append(job_id)
    
    return {"job_ids": job_ids}
```

---

## Testing API Endpoints

```python
# tests/test_api.py
from fastapi.testclient import TestClient
from src.api.main import app

client = TestClient(app)

def test_health():
    resp = client.get("/api/v1/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "healthy"

def test_process_invalid_item():
    resp = client.post("/api/v1/process", json={"item_id": "INVALID"})
    assert resp.status_code == 404

def test_job_not_found():
    resp = client.get("/api/v1/jobs/nonexistent")
    assert resp.status_code == 404
```

Run tests:
```bash
pytest tests/test_api.py -v
```

---

## Key Takeaways

1. **API layer is essential** for production multi-agent systems
2. **Use async for long jobs** - agent workflows often take minutes
3. **Health checks matter** - load balancers and k8s need them
4. **Pydantic models** give you validation + documentation for free
5. **Start simple** - in-memory job tracking works for demos
6. **Scale later** - add Redis/PostgreSQL when you need persistence

The pattern is the same across all agent systems - the specific endpoints change, but the architecture remains consistent.
