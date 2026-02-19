# Part 4, Section 1: Containerization with Docker

## What You'll Learn
- Why containerize RAG applications
- Docker basics for ML applications
- The FinanceRAG Dockerfile explained

---

## Why Docker?

**Without Docker:**
```
"It works on my machine!"

Developer's laptop:
- Python 3.11
- ChromaDB 0.4.15
- LangChain 0.2.0

Production server:
- Python 3.9
- ChromaDB 0.4.10
- LangChain 0.1.5
  
= 🔥 Problems
```

**With Docker:**
```
Same environment everywhere.

Container includes:
- Exact Python version
- Exact dependencies
- Exact configuration

Developer = Staging = Production
```

---

## Docker Concepts

### Image vs Container

```
Image (Recipe)              Container (Running Instance)
┌─────────────────┐        ┌─────────────────┐
│ Python 3.11     │        │ Python 3.11     │
│ Requirements    │ ─run─► │ Your app        │
│ Your code       │        │ Actually running│
└─────────────────┘        └─────────────────┘
    Immutable                   Can be many
    Shareable                   Temporary
```

### Layers

Docker images are built in layers:

```dockerfile
FROM python:3.11-slim    # Layer 1: Base OS + Python
COPY requirements.txt .  # Layer 2: Requirements file
RUN pip install -r ...   # Layer 3: Installed packages
COPY . .                 # Layer 4: Your code
```

**Why layers matter:** Each layer is cached. Change code? Only layer 4 rebuilds.

---

## FinanceRAG Dockerfile

```dockerfile
# syntax=docker/dockerfile:1

# ============================================================
# Stage 1: Builder
# ============================================================
FROM python:3.11-slim as builder

WORKDIR /app

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Create virtual environment
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt
```

### Builder Stage Explained

| Line | Purpose |
|------|---------|
| `FROM python:3.11-slim as builder` | Start with minimal Python image |
| `WORKDIR /app` | Set working directory |
| `apt-get install build-essential` | C compiler for some packages |
| `rm -rf /var/lib/apt/lists/*` | Clean up to reduce image size |
| `python -m venv /opt/venv` | Create isolated environment |
| `pip install --no-cache-dir` | Don't cache pip downloads |

---

```dockerfile
# ============================================================
# Stage 2: Runtime
# ============================================================
FROM python:3.11-slim

WORKDIR /app

# Copy virtual environment from builder
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy application code
COPY src/ ./src/
COPY main.py .
COPY sample_docs/ ./sample_docs/

# Create data directory for ChromaDB
RUN mkdir -p /app/data

# Environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Run the application
CMD ["python", "main.py", "serve"]
```

### Runtime Stage Explained

| Line | Purpose |
|------|---------|
| `FROM python:3.11-slim` | Fresh slim image (no build tools) |
| `COPY --from=builder /opt/venv` | Copy only the venv |
| `PYTHONUNBUFFERED=1` | Print logs immediately |
| `EXPOSE 8000` | Document which port (doesn't open it) |
| `HEALTHCHECK` | Docker checks if container is healthy |
| `CMD` | Default command when container starts |

---

## Multi-Stage Builds

**Why two stages?**

```
Builder stage:
- Has gcc, build tools
- ~800MB

Runtime stage:
- Just Python + your code
- ~300MB
```

Result: Smaller, more secure production image.

---

## Docker Compose

For local development with multiple services:

```yaml
# docker-compose.yml
version: '3.8'

services:
  financerag:
    build: .
    ports:
      - "8000:8000"
    volumes:
      - ./data:/app/data          # Persist ChromaDB
      - ./sample_docs:/app/sample_docs  # Hot reload docs
    environment:
      - OPENAI_API_KEY=${OPENAI_API_KEY}
      - LLM_MODEL=gpt-4o-mini
      - EMBEDDING_MODEL=text-embedding-3-small
    restart: unless-stopped
```

### Running with Compose

```bash
# Start
docker-compose up -d

# View logs
docker-compose logs -f financerag

# Stop
docker-compose down

# Rebuild after code changes
docker-compose up -d --build
```

---

## Volume Mounts

**Problem:** ChromaDB data is inside the container. Container dies = data gone.

**Solution:** Mount a host directory:

```yaml
volumes:
  - ./data:/app/data    # Host:Container
```

Now `./data/chroma/` on your machine persists across container restarts.

---

## Environment Variables

**Never hardcode secrets!**

```dockerfile
# BAD
ENV OPENAI_API_KEY=sk-abc123...

# GOOD - pass at runtime
docker run -e OPENAI_API_KEY=$OPENAI_API_KEY ...
```

### Using .env File

```bash
# .env
OPENAI_API_KEY=sk-abc123...
LLM_MODEL=gpt-4o-mini
```

```yaml
# docker-compose.yml
services:
  financerag:
    env_file:
      - .env
```

---

## Common Docker Commands

```bash
# Build image
docker build -t financerag:latest .

# Run container
docker run -d -p 8000:8000 --name rag financerag:latest

# View logs
docker logs -f rag

# Execute command in running container
docker exec -it rag bash

# Stop and remove
docker stop rag && docker rm rag

# List images
docker images

# Clean up
docker system prune -a  # Remove unused images/containers
```

---

## Optimizing Docker Images

### 1. Use .dockerignore

```
# .dockerignore
.git
.env
__pycache__
*.pyc
.pytest_cache
.venv
data/
*.md
!README.md
```

### 2. Order Matters

```dockerfile
# GOOD - dependencies change less often
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .

# BAD - any code change invalidates pip cache
COPY . .
RUN pip install -r requirements.txt
```

### 3. Pin Versions

```
# requirements.txt
langchain==0.2.0
chromadb==0.4.15
fastapi==0.110.0
```

Not:
```
langchain
chromadb
fastapi
```

---

## Production Deployment

### Running with Gunicorn

For production, use a proper WSGI server:

```dockerfile
CMD ["gunicorn", "src.api.main:app", "-w", "4", "-k", "uvicorn.workers.UvicornWorker", "-b", "0.0.0.0:8000"]
```

- `-w 4`: 4 worker processes
- `-k uvicorn.workers.UvicornWorker`: Async worker class

### With Docker Swarm/Kubernetes

```yaml
# For Kubernetes
apiVersion: apps/v1
kind: Deployment
metadata:
  name: financerag
spec:
  replicas: 3
  template:
    spec:
      containers:
      - name: financerag
        image: financerag:latest
        ports:
        - containerPort: 8000
        env:
        - name: OPENAI_API_KEY
          valueFrom:
            secretKeyRef:
              name: openai-secret
              key: api-key
```

---

## Interview Questions

**Q: Why use multi-stage builds?**

A: 
1. Smaller final image (no build tools)
2. More secure (fewer attack surfaces)
3. Faster deployments (less to transfer)

**Q: How do you handle secrets in Docker?**

A: Never in Dockerfile or image. Options:
1. Environment variables at runtime
2. Docker secrets (Swarm)
3. Kubernetes secrets
4. External secret managers (Vault, AWS Secrets Manager)

**Q: How would you scale this RAG service?**

A:
1. Run multiple container replicas
2. Load balancer in front
3. Shared vector database (e.g., Pinecone instead of local ChromaDB)
4. Or: ChromaDB in separate container, shared volume/network

**Q: What's the difference between COPY and ADD?**

A:
- `COPY`: Just copies files
- `ADD`: Can also extract tar files, fetch URLs

Use `COPY` unless you need `ADD`'s features.

---

## Quick Reference

```bash
# Build
docker build -t myapp:v1 .

# Run
docker run -d -p 8000:8000 -e API_KEY=xxx myapp:v1

# Compose up
docker-compose up -d

# View logs
docker logs -f container_name

# Shell into container
docker exec -it container_name bash

# Clean up
docker system prune -a
```

---

## Next Up

Section 2: Observability & Logging — seeing what's happening inside your RAG system.
