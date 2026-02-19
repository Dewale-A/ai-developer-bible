# Part 5, Section 1: Core Concepts Q&A

## The Most Common RAG Interview Questions

---

## Foundational Questions

### Q: What is RAG and why would you use it?

**Good Answer:**

RAG (Retrieval-Augmented Generation) combines information retrieval with language model generation. Instead of relying solely on what the LLM learned during training, we:

1. **Retrieve** relevant documents from a knowledge base
2. **Augment** the prompt with this context
3. **Generate** an answer grounded in real data

**Why use it:**
- LLMs have knowledge cutoffs — RAG adds current information
- LLMs hallucinate — RAG grounds answers in real documents
- LLMs can't access private data — RAG brings in your documents
- Fine-tuning is expensive — RAG is cheaper and more flexible

**When to use:**
- Question-answering over documents
- Customer support with knowledge bases
- Legal/financial document analysis
- Any domain where accuracy matters

---

### Q: Explain how embeddings work.

**Good Answer:**

Embeddings convert text into dense numerical vectors that capture semantic meaning.

**The process:**
1. Text goes into a trained neural network
2. Network outputs a fixed-size vector (e.g., 1536 numbers)
3. Similar meanings → similar vectors (close in vector space)

**Example:**
```
"bank loan" → [0.12, -0.34, 0.56, ...]
"credit facility" → [0.11, -0.32, 0.58, ...] ← Close!
"river bank" → [-0.45, 0.78, -0.12, ...] ← Far!
```

**Key insight:** The model learned that "bank loan" and "credit facility" appear in similar contexts, so their embeddings are similar — even though the words are different.

---

### Q: What's the difference between vector similarity and keyword search?

**Good Answer:**

| Aspect | Keyword Search | Vector Search |
|--------|---------------|---------------|
| **How it works** | Match exact terms | Match meaning |
| **Example** | "bank loan" finds "bank loan" | "bank loan" finds "credit facility" |
| **Pros** | Fast, precise, explainable | Semantic understanding |
| **Cons** | Misses synonyms | Less precise, needs embeddings |

**Best practice:** Use both (hybrid search). Vector search for semantic understanding, keyword search for exact terms and acronyms.

---

### Q: What is chunking and why does it matter?

**Good Answer:**

Chunking splits documents into smaller pieces for:
1. **Embedding limits** — models have token limits (e.g., 8192)
2. **Retrieval precision** — find specific relevant sections
3. **Context efficiency** — don't waste tokens on irrelevant parts

**Chunking strategies:**
- **Fixed size:** Simple but may cut mid-sentence
- **Recursive:** Split on paragraphs, then sentences
- **Semantic:** Group related content together

**Key trade-offs:**
- Smaller chunks = more precise retrieval, may lose context
- Larger chunks = more context, may include irrelevant info

**Typical settings:** 500-1500 characters with 10-20% overlap.

---

## Architecture Questions

### Q: Walk me through a RAG query flow.

**Good Answer:**

```
1. USER QUERY
   "What are the capital requirements under Basel III?"
   
2. EMBED QUERY
   Query → Embedding model → [0.23, -0.12, 0.89, ...]
   
3. VECTOR SEARCH
   Find K nearest vectors in database
   Return: [chunk1, chunk2, chunk3, ...] with scores
   
4. BUILD PROMPT
   System: "You are a helpful assistant..."
   Context: [chunk1] [chunk2] [chunk3]
   Question: "What are the capital requirements..."
   
5. GENERATE ANSWER
   Send prompt to LLM → Get response
   
6. RETURN WITH SOURCES
   {answer: "...", sources: [chunk1, chunk2, chunk3]}
```

**Key components:**
- Embedding model (e.g., text-embedding-3-small)
- Vector database (e.g., ChromaDB, Pinecone)
- LLM (e.g., GPT-4o-mini)

---

### Q: How do you handle documents that are too long to embed?

**Good Answer:**

Three approaches:

1. **Chunking (most common)**
   - Split into smaller pieces
   - Embed each chunk separately
   - Retrieve relevant chunks at query time

2. **Summarization**
   - Create summary of long document
   - Embed the summary
   - Optionally, link to full document

3. **Hierarchical**
   - Chunk document
   - Create summary of chunks
   - Search summaries first, then drill into chunks

**Best practice:** Chunk with overlap (e.g., 200 characters) so context isn't lost at boundaries.

---

### Q: What's the difference between a bi-encoder and cross-encoder?

**Good Answer:**

**Bi-encoder (used for retrieval):**
```
Query:    "What is Basel III?" → [0.23, 0.45, ...]
Document: "Basel III is a..."  → [0.25, 0.43, ...]

Score = cosine_similarity(query_vec, doc_vec)
```
- Embeddings computed separately
- Can pre-compute document embeddings
- Fast retrieval (compare vectors)

**Cross-encoder (used for reranking):**
```
Input: "[Query] What is Basel III? [Document] Basel III is a..."
Output: Score = 0.95
```
- Sees query and document together
- More accurate (understands relationship)
- Slow (can't pre-compute)

**Best practice:** Use bi-encoder for fast retrieval (top 20), then cross-encoder to rerank (top 5).

---

## Practical Questions

### Q: How do you evaluate RAG system quality?

**Good Answer:**

**Retrieval metrics:**
- **Recall@K:** % of relevant docs in top K
- **MRR (Mean Reciprocal Rank):** Position of first relevant doc
- **NDCG:** Quality of ranking (relevant docs should be higher)

**Generation metrics:**
- **Faithfulness:** Is the answer supported by context?
- **Relevance:** Does it answer the question?
- **Completeness:** Does it cover all aspects?

**Practical evaluation:**
1. Build a test set (questions + expected answers)
2. Run queries, collect results
3. Human evaluation or LLM-as-judge
4. Track metrics over time

**Tools:** Ragas, LangSmith, custom evaluation scripts.

---

### Q: How do you handle hallucinations?

**Good Answer:**

**Prevention:**
1. **Strong prompts:** "Only use information from the context. If unsure, say so."
2. **Temperature = 0:** More deterministic, less creative
3. **Require citations:** "Cite the specific section for each claim."

**Detection:**
1. **Cross-check:** Verify claims against source documents
2. **Confidence scores:** Track retrieval relevance
3. **Human review:** For high-stakes domains

**Mitigation:**
1. **Fallback responses:** "I don't have enough information to answer this."
2. **Source highlighting:** Show users where answers come from
3. **Uncertainty language:** "Based on the documents, it appears that..."

---

### Q: How do you improve retrieval quality?

**Good Answer:**

**Quick wins:**
1. Better chunking (respect document structure)
2. Add metadata for filtering
3. Tune K (retrieve more, let LLM filter)

**Medium effort:**
4. Hybrid search (vector + keyword)
5. Reranking with cross-encoder
6. Query expansion (rephrase query multiple ways)

**Advanced:**
7. Fine-tune embedding model on your domain
8. HyDE (generate hypothetical answer, embed that)
9. Multi-hop retrieval (iterative refinement)

---

### Q: What would you do if retrieval is good but answers are bad?

**Good Answer:**

If I'm getting relevant documents but poor answers, the problem is in generation:

1. **Check the prompt:**
   - Is context being passed correctly?
   - Are instructions clear?
   - Try few-shot examples

2. **Check context length:**
   - Too long → LLM gets lost
   - Too short → missing information

3. **Check the model:**
   - Try a better model (GPT-4 vs GPT-3.5)
   - Adjust temperature

4. **Check context format:**
   - Clear document boundaries
   - Relevant metadata included

5. **Debug with manual inspection:**
   - Print the full prompt
   - See exactly what LLM receives

---

## Scenario Questions

### Q: How would you build a RAG system for a 10 million document corpus?

**Good Answer:**

**Key decisions:**

1. **Vector database:**
   - Not ChromaDB (single machine limit)
   - Use: Pinecone, Weaviate, or Qdrant cluster

2. **Embedding strategy:**
   - Batch processing (not one at a time)
   - Consider cheaper/faster model for initial index
   - Async processing pipeline

3. **Retrieval optimization:**
   - Metadata filtering (narrow search space)
   - Two-stage: coarse filter → fine retrieval
   - Caching for common queries

4. **Infrastructure:**
   - Separate ingestion from serving
   - Multiple retrieval replicas
   - CDN for static assets

5. **Cost management:**
   - Embed once, query many times
   - Cache aggressively
   - Use cheaper models where possible

---

### Q: A user says the system gives wrong answers. How do you debug?

**Good Answer:**

**Step 1: Reproduce**
- Get the exact query
- Run it, see what happens

**Step 2: Check retrieval**
- What documents were retrieved?
- Are they relevant? Check similarity scores.
- If not relevant → retrieval problem

**Step 3: Check context**
- What was passed to the LLM?
- Is the answer actually in the context?
- If not → need different documents

**Step 4: Check generation**
- Does the LLM answer match the context?
- If context is good but answer is wrong → prompt problem

**Step 5: Root cause**
- Wrong docs retrieved? → improve chunking/embedding
- Right docs but low ranked? → add reranking
- LLM misinterpreting? → improve prompt

---

## Quick Tips for Interviews

1. **Always explain trade-offs** — there's no perfect solution
2. **Mention scale considerations** — what works at 1K docs vs 1M
3. **Talk about production concerns** — error handling, monitoring
4. **Give concrete numbers** — chunk sizes, K values, latencies
5. **Admit uncertainty** — "I'd need to test, but my hypothesis is..."

---

## Next Up

Section 2: System Design Scenarios — whiteboard-style problems.
