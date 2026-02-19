# Part 5, Section 3: Code Explanation Practice

## How to Explain Your Code in Interviews

---

## The Framework

When asked about your code:

1. **Start high-level** — what does it do, why does it exist
2. **Explain the flow** — data in, data out
3. **Highlight key decisions** — why this approach
4. **Mention trade-offs** — what you'd do differently

---

## FinanceRAG Code Walkthrough

### Opening Statement

> "FinanceRAG is a production-grade RAG system I built for financial document Q&A. It uses LangChain, ChromaDB, and OpenAI to let users ask questions about regulatory documents like Basel III requirements or AML policies. Let me walk you through the key components."

---

### Document Processor

**File:** `src/tools/document_processor.py`

```python
class DocumentProcessor:
    def __init__(self):
        self.settings = get_settings()
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.settings.chunk_size,
            chunk_overlap=self.settings.chunk_overlap,
            separators=["\n\n", "\n", " ", ""]
        )
```

**How to explain:**

> "The DocumentProcessor handles ingestion. It loads files and splits them into chunks for embedding.
> 
> I'm using RecursiveCharacterTextSplitter because it tries to keep semantic units together — it first tries to split on paragraphs, then sentences, then words. This gives better retrieval than just cutting every N characters.
> 
> The chunk_size and overlap come from settings, so they're configurable without code changes. I typically use 1000 characters with 200 overlap — that's a good balance for financial docs."

**Follow-up questions to expect:**

Q: *"Why 1000 characters?"*

> "It's a balance. Smaller chunks give more precise retrieval but may lose context. Larger chunks include more context but might retrieve irrelevant info. For financial docs with dense information, 1000 works well. I'd tune this based on evaluation."

Q: *"What about PDFs with tables?"*

> "Good question. The current loader handles basic PDF text. For complex tables, I'd use a specialized parser like Unstructured or Tabula to preserve table structure, then embed that separately."

---

### Vector Store Manager

**File:** `src/tools/vector_store.py`

```python
class VectorStoreManager:
    def __init__(self):
        self.settings = get_settings()
        self.embeddings = OpenAIEmbeddings(
            model=self.settings.embedding_model,
            openai_api_key=self.settings.openai_api_key
        )
        self._vector_store: Optional[Chroma] = None
    
    @property
    def vector_store(self) -> Chroma:
        if self._vector_store is None:
            self._vector_store = Chroma(
                collection_name=self.settings.collection_name,
                embedding_function=self.embeddings,
                persist_directory=self.settings.chroma_persist_dir
            )
        return self._vector_store
```

**How to explain:**

> "This class manages all interactions with ChromaDB. There are two key patterns here:
> 
> First, lazy initialization with the @property decorator. The vector store isn't created until it's first accessed. This makes startup faster and lets us fail gracefully if there's a config issue.
> 
> Second, it's a singleton — there's a module-level `get_vector_store_manager()` function that ensures we only have one instance. Opening multiple database connections would waste resources and could cause consistency issues."

**Follow-up questions:**

Q: *"Why ChromaDB instead of Pinecone?"*

> "For this project, ChromaDB made sense because it's zero-setup — just pip install and go. It persists to disk, so data survives restarts. For a production system with millions of documents, I'd switch to Pinecone or Qdrant for better scaling. But ChromaDB is perfect for development and small-to-medium scale."

Q: *"What happens if OpenAI is down?"*

> "Currently, it would throw an exception. In production, I'd add retry logic with exponential backoff, maybe a circuit breaker, and ideally a fallback embedding model. The architecture makes that easy to add since embeddings are injected."

---

### RAG Chain

**File:** `src/tools/rag_chain.py`

```python
SYSTEM_PROMPT = """You are a knowledgeable financial services assistant...

Context from financial documents:
{context}

Question: {question}

Provide a comprehensive answer based on the context above..."""

class RAGChain:
    def query(self, question: str, k: int = 5) -> Dict[str, Any]:
        # Retrieve
        docs_with_scores = self.vector_store_manager.similarity_search_with_scores(
            query=question, k=k
        )
        
        # Format context
        context = self._format_docs(docs)
        
        # Generate
        chain = self.prompt | self.llm | StrOutputParser()
        answer = chain.invoke({"context": context, "question": question})
        
        return {"answer": answer, "sources": sources}
```

**How to explain:**

> "This is the core RAG logic. The query method does three things:
> 
> First, retrieval — it searches the vector store for the K most relevant chunks. I use similarity_search_with_scores to get both the documents and their relevance scores, which is useful for debugging and confidence thresholds.
> 
> Second, context building — the _format_docs method formats the chunks into a string with clear boundaries. Each chunk is labeled '[Document 1: filename.md]' so the LLM can cite sources.
> 
> Third, generation — I use LangChain's pipe syntax to chain the prompt template, LLM, and output parser. It's clean and easy to modify."

**Key point about the prompt:**

> "The system prompt is critical. It explicitly says 'base your answers strictly on the provided context' and 'if the context doesn't contain enough information, say so clearly.' This reduces hallucinations significantly."

**Follow-up questions:**

Q: *"Why K=5?"*

> "Five is a reasonable default. More chunks means more context for the LLM, but also more tokens and cost. I'd tune this based on query types — simple factual questions might need only 2-3, while complex analysis might need 7-10."

Q: *"How do you handle when the question isn't in the documents?"*

> "The prompt instructs the LLM to say it doesn't have enough information. Additionally, I return the relevance scores — if they're all low (say, below 0.5), that's a signal the documents aren't relevant. You could add a confidence threshold that returns a fallback message."

---

### API Layer

**File:** `src/api/main.py`

```python
@app.post("/query", response_model=QueryResponse)
async def query_documents(request: QueryRequest):
    try:
        rag_chain = get_rag_chain()
        result = rag_chain.query(request.question, k=request.k)
        
        return QueryResponse(
            answer=result["answer"],
            sources=[SourceDocument(**s) for s in result["sources"]],
            query=result["query"],
            documents_retrieved=result["documents_retrieved"]
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
```

**How to explain:**

> "The API uses FastAPI, which I chose for three reasons:
> 
> First, automatic validation — the QueryRequest model ensures the question is at least 3 characters and K is between 1 and 20. Invalid requests get rejected with helpful error messages.
> 
> Second, auto-generated documentation — /docs gives you Swagger UI for free, which is great for testing and onboarding.
> 
> Third, async support — although the current implementation isn't fully async, FastAPI handles concurrent requests well, and I can add async LLM calls later."

---

## Common Code Questions

### "What would you change?"

Good answers show self-awareness:

> "A few things I'd improve:
> 
> 1. **Add hybrid search** — financial documents have lots of acronyms (CET1, LCR) that pure vector search might miss. Combining with keyword search would help.
> 
> 2. **Add reranking** — for better precision, I'd retrieve more candidates (say 20) and rerank with a cross-encoder to get the best 5.
> 
> 3. **Better error handling** — add retry logic, circuit breakers, and graceful degradation.
> 
> 4. **Caching** — common queries should be cached to reduce latency and cost."

### "How would you test this?"

> "Several levels:
> 
> **Unit tests** — test chunking logic, metadata extraction, response formatting
> 
> **Integration tests** — test the full query pipeline with a test database
> 
> **Retrieval evaluation** — build a test set of queries with known relevant documents, measure recall@k
> 
> **End-to-end evaluation** — test answer quality, possibly with LLM-as-judge comparing to expected answers"

### "Walk me through a request"

> "Let's trace a query:
> 
> 1. POST /query with {"question": "What is CET1?", "k": 5}
> 2. FastAPI validates the request
> 3. query_documents calls rag_chain.query()
> 4. The question gets embedded via OpenAI
> 5. ChromaDB finds 5 most similar chunks
> 6. Chunks are formatted into context
> 7. GPT-4o-mini generates answer
> 8. Response includes answer + sources
> 9. Total time: ~2 seconds (mostly LLM generation)"

---

## Tips for Code Explanation

1. **Know your code** — be able to explain any line
2. **Explain the WHY** — decisions matter more than syntax
3. **Acknowledge limitations** — shows maturity
4. **Use concrete numbers** — "1000 characters", "~2 seconds"
5. **Connect to business value** — "This ensures accuracy for compliance"

---

## Practice Exercise

Pick any function in FinanceRAG and practice explaining:
- What it does (one sentence)
- How it works (walk through the code)
- Why you made key decisions
- What you'd improve

---

## Next Up

Section 4: Common Gotchas & How to Handle Them.
