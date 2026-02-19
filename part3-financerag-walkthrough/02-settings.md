# Part 3, Section 2: Settings (Configuration)

## What You'll Learn
- How to manage configuration with Pydantic
- Environment variables for secrets
- Caching settings for performance

---

## The Complete File

**File:** `src/config/settings.py` (~40 lines)

```python
"""Configuration settings for FinanceRAG."""

from pydantic_settings import BaseSettings
from pydantic import Field
from functools import lru_cache
```

### Lines 1-5: Imports

| Import | What It Does |
|--------|--------------|
| `BaseSettings` | Pydantic class that reads from environment variables |
| `Field` | Define field metadata (defaults, validation) |
| `lru_cache` | Cache function results (only runs once) |

---

```python
class Settings(BaseSettings):
    """Application settings loaded from environment variables."""
    
    # OpenAI
    openai_api_key: str = Field(..., env="OPENAI_API_KEY")
    embedding_model: str = Field(default="text-embedding-ada-002", env="EMBEDDING_MODEL")
    llm_model: str = Field(default="gpt-4o-mini", env="LLM_MODEL")
    llm_temperature: float = Field(default=0.1, env="LLM_TEMPERATURE")
```

### Lines 7-14: OpenAI Settings

| Setting | Default | Purpose |
|---------|---------|---------|
| `openai_api_key` | **Required** (`...`) | Your API key |
| `embedding_model` | `text-embedding-ada-002` | Model for vectors |
| `llm_model` | `gpt-4o-mini` | Model for answers |
| `llm_temperature` | `0.1` | Low = more factual |

**Key Point:** `Field(...)` means "required" — app won't start without it.

---

```python
    # ChromaDB
    chroma_persist_dir: str = Field(default="./data/chroma", env="CHROMA_PERSIST_DIR")
    collection_name: str = Field(default="finance_docs", env="COLLECTION_NAME")
```

### Lines 16-18: Database Settings

| Setting | Default | Purpose |
|---------|---------|---------|
| `chroma_persist_dir` | `./data/chroma` | Where vectors are saved |
| `collection_name` | `finance_docs` | Name of the vector collection |

---

```python
    # Chunking
    chunk_size: int = Field(default=1000, env="CHUNK_SIZE")
    chunk_overlap: int = Field(default=200, env="CHUNK_OVERLAP")
```

### Lines 20-22: Chunking Settings

| Setting | Default | Purpose |
|---------|---------|---------|
| `chunk_size` | 1000 | Characters per chunk |
| `chunk_overlap` | 200 | Overlap between chunks |

**Why overlap?** So sentences at chunk boundaries don't get cut off mid-thought.

---

```python
    # Retrieval
    top_k_results: int = Field(default=5, env="TOP_K_RESULTS")
    
    # API
    api_host: str = Field(default="0.0.0.0", env="API_HOST")
    api_port: int = Field(default=8000, env="API_PORT")
```

### Lines 24-29: Retrieval & API Settings

| Setting | Default | Purpose |
|---------|---------|---------|
| `top_k_results` | 5 | How many chunks to retrieve |
| `api_host` | `0.0.0.0` | Listen on all interfaces |
| `api_port` | 8000 | HTTP port |

---

```python
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
```

### Lines 31-33: Config Class

Tells Pydantic: "Read from `.env` file if it exists."

Your `.env` file might look like:
```
OPENAI_API_KEY=sk-proj-xxx...
LLM_MODEL=gpt-4o-mini
CHUNK_SIZE=1000
```

---

```python
@lru_cache()
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()
```

### Lines 36-39: Cached Settings Function

**What `@lru_cache()` does:**
1. First call → creates `Settings()` object
2. Second call → returns the SAME object (no recreation)

**Why?** Reading environment variables is slow. Do it once, reuse everywhere.

---

## How Other Files Use This

```python
from src.config.settings import get_settings

settings = get_settings()

# Now use any setting:
print(settings.openai_api_key)
print(settings.chunk_size)
print(settings.llm_model)
```

---

## Interview Questions

**Q: Why use Pydantic Settings instead of plain environment variables?**

A: Three reasons:
1. **Type validation** — catches mistakes early (e.g., port must be int)
2. **Default values** — don't need to set everything
3. **Documentation** — code shows all available settings

**Q: What happens if OPENAI_API_KEY is missing?**

A: Pydantic raises `ValidationError` at startup — app won't run. This is good! Fail fast, not silently.

**Q: Why cache settings with `@lru_cache`?**

A: Environment variable reads are I/O operations. Caching makes subsequent calls instant.

---

## Next Up

Section 3: `document_processor.py` — loading and chunking documents.
