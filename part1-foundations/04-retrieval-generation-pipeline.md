# 1.4 The Retrieval-Generation Pipeline

## Bringing It All Together

You now understand the pieces:
- **Embeddings** convert text to searchable vectors
- **Vector databases** store and search those vectors efficiently

Now let's see how they combine into a complete RAG pipeline.

---

## The Two Phases of RAG

### Phase 1: Ingestion (Offline)

This happens **before** any user asks a question. It's preparation.

```
Documents → Load → Chunk → Embed → Store
```

1. **Load**: Read documents (PDF, DOCX, Markdown, etc.)
2. **Chunk**: Split into smaller pieces (why? see below)
3. **Embed**: Convert each chunk to a vector
4. **Store**: Save vectors + text + metadata in vector database

This might run once, or on a schedule when documents change.

### Phase 2: Query (Online)

This happens **every time** a user asks a question.

```
Question → Embed → Retrieve → Augment → Generate → Answer
```

1. **Embed**: Convert user question to a vector
2. **Retrieve**: Find similar chunks in vector database
3. **Augment**: Build a prompt with retrieved context
4. **Generate**: Send to LLM for answer
5. **Answer**: Return response to user

---

## Why Chunking Matters

### The Problem

A 50-page policy document is way too long to:
- Embed as a single vector (loses detail, may exceed token limits)
- Pass to an LLM (context window limits, relevance dilution)

### The Solution: Chunking

Break documents into smaller, semantically meaningful pieces.

```
50-page document → ~100 chunks of ~500 tokens each
```

Now:
- Each chunk can be embedded independently
- Search can find the specific relevant section, not the whole document
- Only relevant chunks go to the LLM

### Chunking Strategies (Preview)

| Strategy | How It Works | Best For |
|----------|--------------|----------|
| **Fixed Size** | Every N characters/tokens | Simple, predictable |
| **Recursive** | Split by paragraphs, then sentences | Preserves structure |
| **Semantic** | Split at topic boundaries | Best quality, more complex |
| **Sentence** | Each sentence is a chunk | Q&A over short facts |

We'll dive deep into chunking strategies in Part 2.3.

---

## The Complete Flow (Code)

Here's a minimal but complete RAG pipeline:

```python
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA

# ============ PHASE 1: INGESTION ============

# 1. Load documents
loader = PyPDFLoader("loan_policy.pdf")
documents = loader.load()
print(f"Loaded {len(documents)} pages")

# 2. Chunk documents
splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,      # ~250 words per chunk
    chunk_overlap=200,    # Overlap prevents cutting mid-sentence
    separators=["\n\n", "\n", ". ", " "]
)
chunks = splitter.split_documents(documents)
print(f"Split into {len(chunks)} chunks")

# 3. Embed and store
embeddings = OpenAIEmbeddings()
vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    persist_directory="./chroma_db"
)
print("Stored in vector database")


# ============ PHASE 2: QUERY ============

# 4. Set up retriever
retriever = vectorstore.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 4}  # Return top 4 chunks
)

# 5. Set up LLM
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# 6. Create the chain
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",  # "Stuff" all context into one prompt
    retriever=retriever,
    return_source_documents=True
)

# 7. Ask a question
question = "What documents are required for a loan application?"
result = qa_chain.invoke({"query": question})

print("Answer:", result["result"])
print("\nSources:")
for doc in result["source_documents"]:
    print(f"  - {doc.metadata.get('source', 'Unknown')}, Page {doc.metadata.get('page', '?')}")
```

---

## What Happens Inside the Chain

Let's trace a single query:

### User Asks:
```
"What documents are required for a loan application?"
```

### Step 1: Embed Query
```python
query_vector = embeddings.embed_query("What documents are required...")
# Returns: [0.012, -0.034, 0.056, ...] (1536 floats)
```

### Step 2: Retrieve Similar Chunks
```python
similar_chunks = vectorstore.similarity_search(query_vector, k=4)
# Returns 4 Document objects with text and metadata
```

Retrieved chunks might include:
- "Required documents include: proof of income, bank statements..."
- "The application process begins with document submission..."
- "Income verification requires two years of tax returns..."
- "Identity documents must include government-issued photo ID..."

### Step 3: Build Augmented Prompt
```
System: You are a helpful assistant. Answer based only on the provided context.

Context:
[Chunk 1]: Required documents include: proof of income, bank statements...
[Chunk 2]: The application process begins with document submission...
[Chunk 3]: Income verification requires two years of tax returns...
[Chunk 4]: Identity documents must include government-issued photo ID...

Question: What documents are required for a loan application?

Answer:
```

### Step 4: LLM Generates Response
```
Based on the loan policy, the following documents are required for a loan application:

1. Proof of income (pay stubs or employment letter)
2. Bank statements 
3. Two years of tax returns for income verification
4. Government-issued photo ID

The application process begins once all documents are submitted.
```

### Step 5: Return Answer + Sources
The user sees the answer, and you can show which documents it came from (important for trust and compliance).

---

## Chain Types Explained

LangChain offers different ways to combine retrieved documents:

### `stuff` (What We Used)
- Concatenate all retrieved chunks into one prompt
- Simple, works well when total context fits in context window
- **Use when:** Retrieved content < 4K tokens

### `map_reduce`
- Process each chunk separately, then combine answers
- Good for large amounts of retrieved content
- **Use when:** Many chunks, need to synthesize

### `refine`
- Process chunks one at a time, refining the answer iteratively
- Higher quality but slower (multiple LLM calls)
- **Use when:** Accuracy is critical

### `map_rerank`
- Generate answer from each chunk, score them, return best
- Good for finding the single best answer
- **Use when:** Looking for specific facts

For most RAG applications, `stuff` is the right starting point.

---

## Retrieval Parameters That Matter

### `k` (Number of Results)
How many chunks to retrieve?

- Too few (k=1): Might miss relevant information
- Too many (k=20): Irrelevant chunks dilute context, costs more tokens

**Start with k=4**, tune based on your documents and questions.

### `search_type`
How to select chunks:

| Type | How It Works | When to Use |
|------|--------------|-------------|
| `similarity` | Top K most similar | Default, works well |
| `mmr` | Maximize diversity among results | Avoid redundant chunks |
| `similarity_score_threshold` | Only return if above threshold | Filter low-quality matches |

### `score_threshold`
Minimum similarity score to include a result. Filters out irrelevant matches.

```python
retriever = vectorstore.as_retriever(
    search_type="similarity_score_threshold",
    search_kwargs={"score_threshold": 0.7}
)
```

---

## Common Failure Modes

### 1. Poor Retrieval
**Symptom:** LLM gives wrong answer or says "I don't know" even though the answer is in your documents.

**Causes:**
- Chunking split relevant info across chunks
- Embedding model doesn't understand domain terms
- Query phrasing differs from document phrasing

**Fixes:**
- Increase chunk overlap
- Try larger chunks
- Use hybrid search (semantic + keyword)
- Rephrase queries or add query expansion

### 2. Hallucination Despite Context
**Symptom:** LLM generates plausible but incorrect information not in the retrieved context.

**Causes:**
- Retrieved chunks are vaguely relevant but don't contain the answer
- Prompt doesn't strongly instruct to use only context

**Fixes:**
- Improve retrieval (see above)
- Strengthen prompt: "Answer ONLY based on the provided context. If the answer is not in the context, say 'I don't have information about that.'"
- Lower LLM temperature to 0

### 3. Lost in the Middle
**Symptom:** LLM ignores information in the middle of retrieved context.

**Causes:**
- LLMs pay more attention to the beginning and end of context
- Middle chunks get overlooked

**Fixes:**
- Reduce k (fewer chunks)
- Reorder chunks with most relevant first
- Use `refine` chain type for sequential processing

---

## Measuring RAG Quality

### Retrieval Metrics
- **Recall@K**: Did the relevant chunks appear in top K?
- **Precision@K**: What fraction of retrieved chunks were relevant?
- **MRR** (Mean Reciprocal Rank): How high did the relevant chunk rank?

### Generation Metrics
- **Faithfulness**: Does the answer match the retrieved context?
- **Relevance**: Does the answer address the question?
- **Completeness**: Does it include all relevant information?

We'll cover evaluation in more depth in Part 4.

---

## The RAG Pipeline in FinanceRAG

In our project, this pipeline is structured as:

```
FinanceRAG/
├── src/
│   ├── tools/
│   │   ├── document_processor.py  # Ingestion: load, chunk
│   │   ├── vector_store.py        # Embed, store, retrieve
│   │   └── rag_chain.py           # Query pipeline
│   └── api/
│       └── main.py                # FastAPI endpoints
```

We'll walk through each file in Part 3.

---

## Part 1 Summary

You've now built the mental model for RAG:

| Concept | What You Learned |
|---------|-----------------|
| **RAG** | Retrieve relevant docs, then generate answers using them |
| **Embeddings** | Text → vectors that capture meaning |
| **Vector DBs** | Store and search embeddings efficiently |
| **Pipeline** | Ingest (load→chunk→embed→store) + Query (embed→retrieve→augment→generate) |

**You can now explain RAG to an interviewer, a CEO, or a skeptical engineer.**

---

## What's Next?

Part 1 gave you the concepts. Part 2 goes deeper into each building block:

- How LangChain orchestrates everything
- Loading different document types
- Chunking strategies (this matters more than you think)
- Embedding model selection
- Advanced retrieval techniques

→ [Part 2.1: LangChain Architecture](../part2-building-blocks/01-langchain-architecture.md)

---

## 💡 Interview Tips

> **Q: "Walk me through how a RAG system processes a user query."**
>
> **A:** "When a user submits a question, the system first embeds the question into a vector using the same embedding model used during ingestion. It then performs a similarity search against the vector database to find the most relevant document chunks — typically the top 3-5. These chunks are combined with the original question into a prompt that's sent to the LLM. The prompt instructs the LLM to answer based only on the provided context. The LLM generates a response grounded in the retrieved documents, which is returned to the user along with source attribution."

> **Q: "What are the most common failure modes in RAG systems?"**
>
> **A:** "The main failure modes are: (1) Poor retrieval — the relevant information exists but isn't retrieved, often due to chunking issues or query-document mismatch; (2) Hallucination despite context — the LLM generates information not in the retrieved chunks, usually because the context is vaguely related but doesn't contain the answer; (3) Lost in the middle — relevant information in the middle of the context is ignored. Solutions include tuning chunking parameters, using hybrid search, strengthening the prompt to stay grounded, and reducing the number of retrieved chunks."

