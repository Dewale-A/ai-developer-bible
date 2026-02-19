# Part 3, Section 5: RAG Chain

## What You'll Learn
- How retrieval and generation connect
- Prompt engineering for RAG
- Building the query pipeline

---

## The RAG Chain: Heart of the System

```
Question
    │
    ▼
┌─────────────┐
│ Vector      │──→ Relevant Chunks
│ Store       │
└─────────────┘
    │
    ▼
┌─────────────┐
│ Prompt      │──→ "Context: {chunks}\nQuestion: {query}"
│ Template    │
└─────────────┘
    │
    ▼
┌─────────────┐
│ LLM         │──→ Generated Answer
│ (GPT-4o)    │
└─────────────┘
    │
    ▼
Answer + Sources
```

---

## File: `src/tools/rag_chain.py`

### Part 1: Imports

```python
from typing import List, Dict, Any, Optional
import structlog

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

from src.config.settings import get_settings
from src.tools.vector_store import get_vector_store_manager

logger = structlog.get_logger()
```

**Key imports:**

| Import | Purpose |
|--------|---------|
| `ChatOpenAI` | LLM wrapper for OpenAI chat models |
| `ChatPromptTemplate` | Build prompts with variables |
| `StrOutputParser` | Extract string from LLM response |

---

### Part 2: The System Prompt

This is crucial! It tells the LLM how to behave:

```python
SYSTEM_PROMPT = """You are a knowledgeable financial services assistant specializing in regulatory compliance, risk management, and banking operations. Your role is to provide accurate, helpful answers based on the provided context documents.

Guidelines:
1. Base your answers strictly on the provided context
2. If the context doesn't contain enough information, say so clearly
3. Cite specific sections or documents when possible
4. Use precise financial and regulatory terminology
5. If asked about specific thresholds, ratios, or requirements, quote them exactly
6. Highlight any important compliance considerations or risks

Context from financial documents:
{context}

Question: {question}

Provide a comprehensive answer based on the context above. If the context doesn't contain sufficient information to answer the question, state that clearly and explain what information would be needed."""
```

**Anatomy of a RAG Prompt:**

| Section | Purpose |
|---------|---------|
| Role definition | "You are a financial services assistant..." |
| Guidelines | How to behave (cite sources, be precise) |
| `{context}` | Retrieved chunks go here |
| `{question}` | User's question |
| Instructions | How to structure the answer |

**Key guideline:** "If the context doesn't contain enough information, say so" — prevents hallucination!

---

### Part 3: Class Initialization

```python
class RAGChain:
    """Retrieval-Augmented Generation chain for financial documents."""
    
    def __init__(self):
        self.settings = get_settings()
        self.vector_store_manager = get_vector_store_manager()
        
        self.llm = ChatOpenAI(
            model=self.settings.llm_model,
            temperature=self.settings.llm_temperature,
            openai_api_key=self.settings.openai_api_key
        )
        
        self.prompt = ChatPromptTemplate.from_template(SYSTEM_PROMPT)
```

**ChatOpenAI parameters:**

| Parameter | Value | Why |
|-----------|-------|-----|
| `model` | "gpt-4o-mini" | Fast, cheap, good enough |
| `temperature` | 0.1 | Low = factual, high = creative |

---

### Part 4: Format Retrieved Documents

```python
def _format_docs(self, docs: List[Document]) -> str:
    """Format retrieved documents for the prompt."""
    formatted = []
    for i, doc in enumerate(docs, 1):
        source = doc.metadata.get("filename", "Unknown source")
        content = doc.page_content.strip()
        formatted.append(f"[Document {i}: {source}]\n{content}")
    
    return "\n\n---\n\n".join(formatted)
```

**Input:**
```python
[
    Document(page_content="Basel III requires...", metadata={"filename": "basel.md"}),
    Document(page_content="CET1 ratio is...", metadata={"filename": "basel.md"})
]
```

**Output:**
```
[Document 1: basel.md]
Basel III requires...

---

[Document 2: basel.md]
CET1 ratio is...
```

**Why format this way?**
- Clear document boundaries
- LLM can cite "Document 1" or "Document 2"
- Filename helps with attribution

---

### Part 5: The Query Method (Core Logic!)

```python
def query(
    self, 
    question: str, 
    k: Optional[int] = None
) -> Dict[str, Any]:
    """
    Execute RAG query and return answer with sources.
    """
    k = k or self.settings.top_k_results
    
    try:
        # Step 1: Retrieve relevant documents
        docs_with_scores = self.vector_store_manager.similarity_search_with_scores(
            query=question,
            k=k
        )
```

**Step 1:** Search vector store for relevant chunks.

```python
        # Handle empty results
        if not docs_with_scores:
            return {
                "answer": "No relevant documents found in the knowledge base.",
                "sources": [],
                "query": question,
                "documents_retrieved": 0
            }
        
        # Separate documents and scores
        docs = [doc for doc, score in docs_with_scores]
        scores = [score for doc, score in docs_with_scores]
```

**Step 2:** Handle edge case (empty database) gracefully.

```python
        # Step 3: Format context for the prompt
        context = self._format_docs(docs)
        
        # Step 4: Build and run the chain
        chain = self.prompt | self.llm | StrOutputParser()
        answer = chain.invoke({
            "context": context,
            "question": question
        })
```

**The Chain Explained:**

```
self.prompt | self.llm | StrOutputParser()
     │            │            │
     ▼            ▼            ▼
Template    →   LLM    →   Extract String
             
Input: {"context": "...", "question": "..."}
                 │
                 ▼
         "Based on the context..."
```

The `|` operator is LangChain's "pipe" — output of one becomes input of next.

---

```python
        # Step 5: Prepare sources with metadata
        sources = []
        for doc, score in docs_with_scores:
            sources.append({
                "filename": doc.metadata.get("filename", "Unknown"),
                "chunk_index": doc.metadata.get("chunk_index", 0),
                "relevance_score": round(1 - score, 4),  # Convert distance → similarity
                "preview": doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content
            })
```

**Score conversion:** `1 - score`
- ChromaDB returns **distance** (lower = more similar)
- We want **relevance** (higher = more similar)
- So: `relevance = 1 - distance`

```python
        logger.info("rag_query_complete", 
                   question_preview=question[:50],
                   docs_retrieved=len(docs),
                   avg_score=round(sum(scores)/len(scores), 4))
        
        return {
            "answer": answer,
            "sources": sources,
            "query": question,
            "documents_retrieved": len(docs),
            "model": self.settings.llm_model
        }
        
    except Exception as e:
        logger.error("rag_query_error", error=str(e))
        return {
            "answer": f"An error occurred: {str(e)}",
            "sources": [],
            "query": question,
            "error": str(e)
        }
```

**Return structure:**
```python
{
    "answer": "Basel III requires a minimum CET1 ratio of 4.5%...",
    "sources": [
        {"filename": "basel.md", "chunk_index": 0, "relevance_score": 0.85, "preview": "..."},
        {"filename": "basel.md", "chunk_index": 2, "relevance_score": 0.72, "preview": "..."}
    ],
    "query": "What is the CET1 ratio requirement?",
    "documents_retrieved": 5,
    "model": "gpt-4o-mini"
}
```

---

### Part 6: Query with Chat History

```python
def query_with_history(
    self,
    question: str,
    chat_history: List[Dict[str, str]],
    k: Optional[int] = None
) -> Dict[str, Any]:
    """
    Execute RAG query with conversation history.
    """
    history_context = ""
    if chat_history:
        recent_history = chat_history[-4:]  # Last 2 exchanges
        history_parts = []
        for msg in recent_history:
            role = "User" if msg["role"] == "user" else "Assistant"
            history_parts.append(f"{role}: {msg['content']}")
        history_context = "Recent conversation:\n" + "\n".join(history_parts) + "\n\nCurrent question: "
    
    enhanced_question = history_context + question if history_context else question
    
    return self.query(enhanced_question, k=k)
```

**Why include history?**

User: "What's the CET1 ratio?"
Assistant: "CET1 minimum is 4.5%..."
User: "And what about Tier 1?"  ← Needs context!

The history helps the system understand "what about" refers to capital ratios.

---

### Part 7: Singleton & Convenience Function

```python
_rag_chain: Optional[RAGChain] = None

def get_rag_chain() -> RAGChain:
    """Get singleton RAG chain instance."""
    global _rag_chain
    if _rag_chain is None:
        _rag_chain = RAGChain()
    return _rag_chain

def query(question: str, k: Optional[int] = None) -> Dict[str, Any]:
    """Convenience function to query the RAG chain."""
    chain = get_rag_chain()
    return chain.query(question, k=k)
```

**Two ways to use:**
```python
# Method 1: Direct function call
from src.tools.rag_chain import query
result = query("What is Basel III?")

# Method 2: Get the chain instance
from src.tools.rag_chain import get_rag_chain
chain = get_rag_chain()
result = chain.query("What is Basel III?")
result2 = chain.query_with_history("And Tier 1?", history)
```

---

## Complete Flow Visualization

```
User: "What are the capital requirements under Basel III?"
                            │
                            ▼
            ┌───────────────────────────────┐
            │     similarity_search()        │
            │  "capital requirements Basel"  │
            └───────────────┬───────────────┘
                            │
              ┌─────────────┼─────────────┐
              ▼             ▼             ▼
          [Chunk 1]    [Chunk 2]    [Chunk 3]
          "CET1 4.5%"  "Tier 1 6%"  "Buffers"
                            │
                            ▼
            ┌───────────────────────────────┐
            │       _format_docs()           │
            │  "[Document 1: basel.md]       │
            │   CET1 minimum is 4.5%..."     │
            └───────────────┬───────────────┘
                            │
                            ▼
            ┌───────────────────────────────┐
            │      ChatPromptTemplate        │
            │  "You are a financial...       │
            │   Context: {formatted_docs}    │
            │   Question: {question}"        │
            └───────────────┬───────────────┘
                            │
                            ▼
            ┌───────────────────────────────┐
            │         ChatOpenAI             │
            │      (gpt-4o-mini)             │
            └───────────────┬───────────────┘
                            │
                            ▼
            ┌───────────────────────────────┐
            │      StrOutputParser           │
            │  Extract string from response  │
            └───────────────┬───────────────┘
                            │
                            ▼
    {
      "answer": "Under Basel III, the capital requirements are...",
      "sources": [...],
      "documents_retrieved": 5
    }
```

---

## Interview Questions

**Q: Why use `temperature=0.1` instead of 0?**

A: 
- `0` = deterministic (same answer every time)
- `0.1` = tiny bit of variation, sounds more natural
- For RAG, you want factual answers, so low temperature is correct

**Q: How would you improve retrieval quality?**

A:
1. **Hybrid search**: Combine vector + keyword (BM25)
2. **Reranking**: Use a cross-encoder to re-score results
3. **Query expansion**: Rephrase query multiple ways
4. **Metadata filtering**: Narrow by date, document type

**Q: What if the LLM hallucinates despite having context?**

A:
1. Make prompt stricter: "ONLY use information from the context"
2. Add citation requirements: "Quote the exact text"
3. Post-process: Verify claims against source chunks
4. Use a more capable model (GPT-4 vs GPT-3.5)

**Q: How would you handle multiple languages?**

A:
1. Use multilingual embeddings (e.g., multilingual-e5)
2. Detect query language, retrieve in same language
3. Or translate everything to English before embedding

**Q: What's the trade-off with K (number of retrieved docs)?**

A:
- **Higher K**: More context, better recall, but more tokens (cost + latency)
- **Lower K**: Faster, cheaper, but might miss relevant info
- Sweet spot: Usually 3-7 for most RAG applications

---

## Next Up

Section 6: `api/main.py` — exposing RAG as REST endpoints with FastAPI.
