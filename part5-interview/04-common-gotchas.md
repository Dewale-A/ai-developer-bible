# Part 5, Section 4: Common Gotchas & How to Handle Them

## The Pitfalls That Trip Up RAG Systems

---

## Gotcha 1: "It Works on My Test Query"

### The Problem

Your RAG system answers your test questions perfectly, but fails on real user queries.

```
Your test:    "What is the CET1 ratio requirement under Basel III?"
User's query: "how much capital do banks need"
              "whats the minimum tier 1"
              "basel requirements for capital"
```

### Why It Happens

- Test queries are precise; user queries are vague
- You tested with the same vocabulary as your documents
- Real users don't know the "right" words

### The Fix

1. **Test with diverse queries**
   ```python
   test_queries = [
       # Precise
       "What is the CET1 ratio requirement under Basel III?",
       # Vague
       "how much capital do banks need",
       # Misspelled
       "basil III requirements",
       # Synonym
       "bank capital minimums",
       # Conversational
       "explain the capital rules to me",
   ]
   ```

2. **Add query expansion**
   ```python
   def expand_query(query: str) -> List[str]:
       """Generate alternative phrasings."""
       prompt = f"Generate 3 alternative ways to ask: {query}"
       variations = llm.generate(prompt)
       return [query] + variations
   ```

3. **Hybrid search** — keyword search catches exact terms

---

## Gotcha 2: Chunking at the Wrong Boundaries

### The Problem

Your chunks split mid-concept:

```
Chunk 1: "The minimum CET1 requirement is"
Chunk 2: "4.5% of risk-weighted assets. Additionally..."
```

Query: "What is the CET1 requirement?"
Retrieved: Chunk 1 (says "is" but not the number!)

### Why It Happens

- Fixed-size chunking doesn't respect document structure
- No overlap, or overlap too small

### The Fix

1. **Add meaningful overlap**
   ```python
   splitter = RecursiveCharacterTextSplitter(
       chunk_size=1000,
       chunk_overlap=200,  # 20% overlap
   )
   ```

2. **Respect document structure**
   ```python
   splitter = RecursiveCharacterTextSplitter(
       separators=[
           "\n## ",      # Headers first
           "\n### ",
           "\n\n",       # Then paragraphs
           "\n",         # Then lines
           ". ",         # Then sentences
           " ",          # Then words
       ],
       chunk_size=1000,
   )
   ```

3. **Semantic chunking** — use LLM to find natural break points

---

## Gotcha 3: Embedding Model Mismatch

### The Problem

You embed documents with Model A, then query with Model B.

```python
# Ingestion (last week)
embeddings = OpenAIEmbeddings(model="text-embedding-ada-002")
vector_store.add_documents(docs)

# Query (today, after "upgrading")
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")  # Different!
results = vector_store.similarity_search(query)  # Returns garbage
```

### Why It Happens

- Model upgrade without re-indexing
- Different team members using different configs
- Environment variable differences

### The Fix

1. **Version your index**
   ```python
   collection_name = f"finance_docs_v{EMBEDDING_MODEL_VERSION}"
   ```

2. **Store model info with the collection**
   ```python
   metadata = {
       "embedding_model": "text-embedding-3-small",
       "indexed_at": "2024-02-19",
       "chunk_size": 1000,
   }
   ```

3. **Validate on startup**
   ```python
   def validate_embedding_model():
       stored_model = get_collection_metadata()["embedding_model"]
       current_model = settings.embedding_model
       if stored_model != current_model:
           raise ValueError(f"Model mismatch: {stored_model} vs {current_model}")
   ```

---

## Gotcha 4: Context Window Overflow

### The Problem

You retrieve 10 documents, each 1000 characters. Context = 10,000 chars + prompt + question. Then:

```
openai.BadRequestError: This model's maximum context length is 128000 tokens.
You requested 135000 tokens.
```

### Why It Happens

- K is too high
- Chunks are too large
- Didn't account for prompt overhead

### The Fix

1. **Calculate tokens before sending**
   ```python
   import tiktoken
   
   def count_tokens(text: str, model: str = "gpt-4o-mini") -> int:
       encoding = tiktoken.encoding_for_model(model)
       return len(encoding.encode(text))
   
   def build_context(docs: List[Document], max_tokens: int = 4000):
       context = ""
       total_tokens = 0
       
       for doc in docs:
           doc_tokens = count_tokens(doc.page_content)
           if total_tokens + doc_tokens > max_tokens:
               break
           context += doc.page_content + "\n\n"
           total_tokens += doc_tokens
       
       return context
   ```

2. **Reduce K dynamically**
   ```python
   # Start with more, trim to fit
   docs = retriever.search(query, k=10)
   docs = trim_to_token_limit(docs, max_tokens=4000)
   ```

---

## Gotcha 5: Hallucination Despite Good Context

### The Problem

You retrieve the right documents, but the LLM invents information:

```
Context: "CET1 minimum is 4.5%"
Question: "What about Tier 2?"
Answer: "Tier 2 minimum is 2%"  ← MADE UP!
```

### Why It Happens

- LLM tries to be helpful even without information
- Prompt doesn't clearly say "only use context"
- Temperature too high

### The Fix

1. **Strict prompt**
   ```python
   SYSTEM_PROMPT = """You are a financial assistant.
   
   IMPORTANT RULES:
   - ONLY use information from the provided context
   - If the answer is not in the context, say "I don't have information about this"
   - Never make up facts, numbers, or requirements
   - When uncertain, express uncertainty
   
   Context:
   {context}
   
   Question: {question}
   """
   ```

2. **Low temperature**
   ```python
   llm = ChatOpenAI(temperature=0.1)  # More deterministic
   ```

3. **Require citations**
   ```python
   # Add to prompt:
   "For each claim, cite the specific document and section."
   ```

4. **Post-process validation**
   ```python
   def validate_answer(answer: str, context: str) -> bool:
       """Check if answer claims are in context."""
       # Use another LLM call or rule-based check
       return verify_claims(answer, context)
   ```

---

## Gotcha 6: Slow Cold Start

### The Problem

First query takes 10+ seconds because:
- Loading embedding model
- Connecting to vector DB
- Initializing LLM client

### Why It Happens

- Lazy initialization delays cost to first request
- No connection pooling
- Large model downloads

### The Fix

1. **Warm up on startup**
   ```python
   @app.on_event("startup")
   async def warmup():
       # Initialize everything
       vector_store = get_vector_store_manager()
       rag_chain = get_rag_chain()
       
       # Run a dummy query
       rag_chain.query("warmup query")
   ```

2. **Keep connections alive**
   ```python
   # Use connection pooling for external APIs
   # Keep vector store connection open
   ```

3. **Health check endpoint**
   ```python
   @app.get("/ready")
   async def ready():
       # Returns 200 only when fully initialized
       if not rag_chain_initialized:
           raise HTTPException(503, "Not ready")
       return {"status": "ready"}
   ```

---

## Gotcha 7: Duplicate/Outdated Content

### The Problem

Same information indexed multiple times:
- Document updated but old version still in index
- Same content in multiple files
- Test data mixed with production

### Why It Happens

- No deduplication during ingestion
- No cleanup of old versions
- Shared index between environments

### The Fix

1. **Content hashing**
   ```python
   import hashlib
   
   def get_content_hash(text: str) -> str:
       return hashlib.md5(text.encode()).hexdigest()
   
   def add_if_new(doc: Document):
       hash = get_content_hash(doc.page_content)
       if not exists_in_store(hash):
           vector_store.add_documents([doc])
   ```

2. **Version tracking**
   ```python
   doc.metadata["version"] = "2024-02-19"
   doc.metadata["source_hash"] = hash(source_file)
   
   # On update: remove old, add new
   def update_document(source_path: str):
       vector_store.delete(filter={"source": source_path})
       new_docs = process(source_path)
       vector_store.add_documents(new_docs)
   ```

3. **Separate environments**
   ```python
   collection_name = f"finance_docs_{ENVIRONMENT}"  # dev, staging, prod
   ```

---

## Gotcha 8: "My Relevance Scores Are Low"

### The Problem

Everything has scores like 0.2-0.4 (for similarity) or 0.6-0.8 (for distance).

### Why It Happens

- That's normal! Embedding similarity isn't 0-1 range like you expect
- Different models have different score distributions
- Domain-specific text vs general embeddings

### The Fix

1. **Understand your baseline**
   ```python
   # Test known similar/different pairs
   similar = cosine_similarity(
       embed("bank loan"), 
       embed("credit facility")
   )  # Maybe 0.7
   
   different = cosine_similarity(
       embed("bank loan"),
       embed("pizza recipe")
   )  # Maybe 0.2
   
   # Your threshold should be between these
   ```

2. **Use relative ranking, not absolute scores**
   ```python
   # Don't: if score > 0.8
   # Do: take top K, regardless of absolute score
   ```

3. **Monitor score distributions**
   ```python
   # Track scores over time
   # Alert if distribution changes significantly
   ```

---

## Quick Reference: Gotcha Checklist

Before deploying, verify:

- [ ] Tested with vague/misspelled queries
- [ ] Chunking respects document structure
- [ ] Embedding model version is tracked
- [ ] Context fits in token limit
- [ ] Prompt prevents hallucination
- [ ] First request isn't painfully slow
- [ ] Duplicates are handled
- [ ] Score thresholds are reasonable

---

## Interview Tip

When discussing gotchas, show you've experienced them:

> "One issue I ran into was embedding model mismatch. I upgraded from ada-002 to text-embedding-3-small and forgot to re-index. Queries returned completely wrong results. Now I version my collections and validate the model on startup."

This shows:
1. You've built real systems
2. You learned from mistakes
3. You implemented preventive measures

---

## 🎉 Congratulations!

You've completed **The AI Developer Bible: RAG Edition**.

You now have:
- ✅ Deep understanding of RAG concepts
- ✅ Hands-on experience with FinanceRAG
- ✅ Production-ready patterns
- ✅ Interview preparation

**Go build something great!** 🚀
