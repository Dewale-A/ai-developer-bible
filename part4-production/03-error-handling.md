# Part 4, Section 3: Error Handling & Edge Cases

## What You'll Learn
- Common failure modes in RAG systems
- Graceful degradation strategies
- Retry patterns and circuit breakers

---

## Why RAG Systems Fail

RAG has multiple external dependencies:

```
User Query
    ↓
┌─────────────────┐
│ Embedding API   │ ← Can fail (rate limit, timeout)
└────────┬────────┘
         ↓
┌─────────────────┐
│ Vector Database │ ← Can fail (connection, corruption)
└────────┬────────┘
         ↓
┌─────────────────┐
│ LLM API         │ ← Can fail (rate limit, timeout, content filter)
└────────┬────────┘
         ↓
    Response
```

**Each component can fail independently.**

---

## Common Failure Modes

### 1. API Rate Limits

```python
# OpenAI response
{
  "error": {
    "message": "Rate limit reached for gpt-4o-mini",
    "type": "rate_limit_error",
    "code": "rate_limit_exceeded"
  }
}
```

**Solution: Retry with exponential backoff**

```python
import time
from tenacity import retry, stop_after_attempt, wait_exponential

@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=1, max=10)
)
def call_openai(prompt: str):
    return openai.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}]
    )
```

**Exponential backoff:**
- Attempt 1: Wait 1 second
- Attempt 2: Wait 2 seconds
- Attempt 3: Wait 4 seconds

### 2. Timeouts

```python
# Connection hangs...
requests.exceptions.ReadTimeout: Read timed out
```

**Solution: Set timeouts everywhere**

```python
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(
    model="gpt-4o-mini",
    timeout=30,           # 30 second timeout
    max_retries=2         # Retry twice on failure
)
```

### 3. Empty Results

```python
# No documents match the query
results = vector_store.similarity_search(query, k=5)
# results = []
```

**Solution: Handle gracefully**

```python
def query(question: str):
    docs = retriever.search(question, k=5)
    
    if not docs:
        return {
            "answer": "I couldn't find relevant information in the knowledge base. "
                      "Could you rephrase your question or ask about a different topic?",
            "sources": [],
            "fallback": True
        }
    
    # Continue with normal flow...
```

### 4. Context Too Long

```python
# Token limit exceeded
openai.BadRequestError: This model's maximum context length is 128000 tokens
```

**Solution: Truncate context intelligently**

```python
def prepare_context(docs: List[Document], max_tokens: int = 4000):
    """Build context within token limit."""
    context_parts = []
    current_tokens = 0
    
    for doc in docs:
        doc_tokens = count_tokens(doc.page_content)
        
        if current_tokens + doc_tokens > max_tokens:
            break
        
        context_parts.append(doc.page_content)
        current_tokens += doc_tokens
    
    return "\n\n".join(context_parts)
```

### 5. Content Filtering

```python
# LLM refuses to answer
{
  "error": {
    "code": "content_filter",
    "message": "The response was filtered due to content safety"
  }
}
```

**Solution: Handle filtered responses**

```python
def generate_answer(question: str, context: str):
    try:
        response = llm.invoke(prompt)
        return response
    except openai.ContentFilterError:
        return "I'm unable to provide a response to this query. " \
               "Please rephrase or ask a different question."
```

---

## Implementing Retry Logic

### Using tenacity

```python
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type
)
import openai

@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=1, max=10),
    retry=retry_if_exception_type((
        openai.RateLimitError,
        openai.APITimeoutError,
        openai.APIConnectionError
    ))
)
def embed_with_retry(texts: List[str]):
    return embeddings.embed_documents(texts)
```

### Custom Retry with Logging

```python
import time
import structlog

logger = structlog.get_logger()

def retry_with_backoff(func, max_retries=3, base_delay=1):
    """Retry function with exponential backoff."""
    
    for attempt in range(max_retries):
        try:
            return func()
        except Exception as e:
            if attempt == max_retries - 1:
                logger.error("retry_exhausted", 
                    function=func.__name__,
                    attempts=max_retries,
                    error=str(e)
                )
                raise
            
            delay = base_delay * (2 ** attempt)
            logger.warning("retry_attempt",
                function=func.__name__,
                attempt=attempt + 1,
                delay=delay,
                error=str(e)
            )
            time.sleep(delay)
```

---

## Circuit Breaker Pattern

**Problem:** Failing service causes cascading failures.

```
Every request tries OpenAI → All fail → System overloaded
```

**Solution:** Stop trying temporarily.

```python
from circuitbreaker import circuit

class OpenAICircuitOpen(Exception):
    pass

@circuit(failure_threshold=5, recovery_timeout=60)
def call_openai(prompt: str):
    """Call OpenAI with circuit breaker protection."""
    return llm.invoke(prompt)

# Usage
try:
    response = call_openai(prompt)
except OpenAICircuitOpen:
    # Circuit is open - use fallback
    response = fallback_response()
```

**How it works:**
1. Normal: Calls go through
2. 5 failures: Circuit opens (stops calling)
3. After 60 seconds: Circuit half-opens (tries one call)
4. If success: Circuit closes (normal again)

---

## Graceful Degradation

When something fails, provide partial service:

```python
def query(question: str) -> QueryResponse:
    # Try full RAG pipeline
    try:
        docs = retriever.search(question, k=5)
        answer = llm.generate(docs, question)
        return QueryResponse(answer=answer, sources=docs)
    
    except RetrievalError:
        # Fallback: LLM without context
        logger.warning("degraded_mode", reason="retrieval_failed")
        answer = llm.generate([], question)
        return QueryResponse(
            answer=answer + "\n\n(Note: Unable to search knowledge base)",
            sources=[],
            degraded=True
        )
    
    except LLMError:
        # Fallback: Return sources only
        logger.warning("degraded_mode", reason="llm_failed")
        return QueryResponse(
            answer="I'm currently unable to generate an answer. "
                   "Here are relevant documents:",
            sources=docs,
            degraded=True
        )
    
    except Exception as e:
        # Complete failure
        logger.error("query_failed", error=str(e))
        return QueryResponse(
            answer="Service temporarily unavailable. Please try again.",
            sources=[],
            error=True
        )
```

---

## Validation & Sanitization

### Input Validation

```python
from pydantic import BaseModel, Field, validator

class QueryRequest(BaseModel):
    question: str = Field(..., min_length=3, max_length=1000)
    k: int = Field(default=5, ge=1, le=20)
    
    @validator('question')
    def sanitize_question(cls, v):
        # Remove potential prompt injection
        v = v.replace("Ignore previous instructions", "")
        # Remove excessive whitespace
        v = " ".join(v.split())
        return v
```

### Output Validation

```python
def validate_response(response: str) -> str:
    """Validate LLM response before returning."""
    
    # Check for empty response
    if not response or len(response.strip()) < 10:
        raise ValueError("Response too short")
    
    # Check for error patterns
    error_patterns = [
        "I cannot",
        "I'm not able to",
        "Error:",
    ]
    
    if any(pattern in response for pattern in error_patterns):
        logger.warning("response_flagged", response_preview=response[:100])
    
    return response
```

---

## Error Responses in API

```python
from fastapi import HTTPException
from fastapi.responses import JSONResponse

class RAGError(Exception):
    """Base exception for RAG errors."""
    def __init__(self, message: str, code: str):
        self.message = message
        self.code = code

class RetrievalError(RAGError):
    pass

class GenerationError(RAGError):
    pass

@app.exception_handler(RAGError)
async def rag_error_handler(request: Request, exc: RAGError):
    return JSONResponse(
        status_code=500,
        content={
            "error": {
                "code": exc.code,
                "message": exc.message
            }
        }
    )

@app.post("/query")
async def query(request: QueryRequest):
    try:
        result = rag_chain.query(request.question)
        return result
    except RetrievalError as e:
        raise HTTPException(status_code=503, detail="Search service unavailable")
    except GenerationError as e:
        raise HTTPException(status_code=503, detail="Generation service unavailable")
```

---

## Testing Error Handling

```python
import pytest
from unittest.mock import patch, MagicMock

def test_handles_empty_results():
    """Test graceful handling of no results."""
    with patch.object(vector_store, 'similarity_search', return_value=[]):
        result = rag_chain.query("unknown topic xyz")
        
        assert "couldn't find" in result['answer'].lower()
        assert result['sources'] == []

def test_handles_llm_timeout():
    """Test retry on timeout."""
    mock_llm = MagicMock()
    mock_llm.invoke.side_effect = [
        openai.APITimeoutError("timeout"),
        openai.APITimeoutError("timeout"),
        "Success response"
    ]
    
    with patch.object(rag_chain, 'llm', mock_llm):
        result = rag_chain.query("test question")
        
        assert mock_llm.invoke.call_count == 3
        assert result['answer'] == "Success response"

def test_circuit_breaker_opens():
    """Test circuit breaker after failures."""
    for _ in range(5):
        with pytest.raises(openai.APIError):
            call_openai("test")
    
    # Circuit should be open now
    with pytest.raises(OpenAICircuitOpen):
        call_openai("test")
```

---

## Interview Questions

**Q: How would you handle OpenAI rate limits in production?**

A:
1. Implement exponential backoff retry
2. Add request queuing for batch operations
3. Monitor usage, set alerts before limits
4. Consider caching for repeated queries
5. Have fallback model (e.g., switch to GPT-3.5)

**Q: What's the circuit breaker pattern and when would you use it?**

A: Circuit breaker stops calling a failing service to prevent cascade failures. Use when:
- External API might be down
- Want to fail fast instead of timeout
- Need to protect downstream services
- Want automatic recovery when service returns

**Q: How do you handle prompt injection attacks?**

A:
1. Sanitize user input (remove instruction-like patterns)
2. Use system prompts that establish clear boundaries
3. Validate output for unexpected patterns
4. Log suspicious queries for review
5. Consider content filtering

**Q: What's graceful degradation?**

A: Providing reduced but functional service when components fail:
- If retrieval fails → answer without context
- If LLM fails → return just the sources
- If both fail → helpful error message

Better than complete failure.

---

## Quick Reference

### Retry with tenacity
```python
from tenacity import retry, stop_after_attempt, wait_exponential

@retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=10))
def api_call():
    ...
```

### Handle Empty Results
```python
if not results:
    return {"answer": "No relevant documents found", "fallback": True}
```

### Timeout Configuration
```python
llm = ChatOpenAI(timeout=30, max_retries=2)
```

### Custom Exception
```python
class RAGError(Exception):
    def __init__(self, message, code):
        self.message = message
        self.code = code
```

---

## Next Up

Section 4: Performance Optimization — making RAG systems fast.
