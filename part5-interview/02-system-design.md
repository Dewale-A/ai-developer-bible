# Part 5, Section 2: System Design Scenarios

## How to Approach RAG System Design Questions

---

## The Framework

When asked to design a RAG system, follow this structure:

1. **Clarify Requirements** (2 min)
2. **High-Level Architecture** (5 min)
3. **Deep Dive on Components** (10 min)
4. **Scale & Production Concerns** (5 min)
5. **Trade-offs & Alternatives** (3 min)

---

## Scenario 1: Enterprise Document Q&A

**Prompt:** "Design a system that lets employees ask questions about company policies, procedures, and documentation."

### 1. Clarify Requirements

Ask:
- How many documents? (Say: 50,000 policies, ~100 pages each)
- How many users? (Say: 10,000 employees)
- Query volume? (Say: 1,000 queries/day)
- Update frequency? (Say: documents change weekly)
- Security requirements? (Say: role-based access, sensitive content)

### 2. High-Level Architecture

```
┌──────────────────────────────────────────────────────────┐
│                    INGESTION PIPELINE                     │
├──────────────────────────────────────────────────────────┤
│                                                          │
│  SharePoint/  ──► Document    ──► Chunker  ──► Embedding │
│  Confluence       Processor       +Meta       Service    │
│                                                   │      │
│                                                   ▼      │
│                                            Vector DB     │
│                                           (Pinecone)     │
└──────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────┐
│                     QUERY PIPELINE                        │
├──────────────────────────────────────────────────────────┤
│                                                          │
│  User Query  ──►  API   ──►  Auth   ──►  Retrieval      │
│  (Slack/Web)    Gateway     Layer       Service         │
│                                            │             │
│                                            ▼             │
│                              ┌──────────────────┐        │
│                              │ LLM Service      │        │
│                              │ (OpenAI/Azure)   │        │
│                              └────────┬─────────┘        │
│                                       │                  │
│                                       ▼                  │
│                              Answer + Sources            │
└──────────────────────────────────────────────────────────┘
```

### 3. Component Deep Dive

**Document Processing:**
```python
# Handle multiple formats
loaders = {
    ".pdf": PyPDFLoader,
    ".docx": Docx2txtLoader,
    ".pptx": UnstructuredPowerPointLoader,
    ".html": BSHTMLLoader,  # Confluence pages
}

# Extract metadata
metadata = {
    "source": "HR Policies",
    "department": "Human Resources",
    "last_updated": "2024-01-15",
    "access_level": "all_employees",
}
```

**Chunking Strategy:**
- Respect document structure (sections, headers)
- 1000 chars with 200 overlap
- Keep metadata with each chunk

**Access Control:**
```python
def get_filtered_retriever(user: User):
    # Filter based on user's permissions
    allowed_departments = get_user_permissions(user)
    
    return vector_store.as_retriever(
        filter={"department": {"$in": allowed_departments}}
    )
```

### 4. Scale Considerations

- **Vector DB:** Pinecone (managed, scales automatically)
- **Caching:** Redis for common queries (80/20 rule)
- **API:** Multiple replicas behind load balancer
- **Ingestion:** Async worker queue (Celery/SQS)

### 5. Trade-offs

- **Pinecone vs self-hosted:** Higher cost, less operational burden
- **One big index vs per-department:** Simpler vs better filtering
- **Real-time vs batch updates:** Freshness vs cost

---

## Scenario 2: Customer Support Bot

**Prompt:** "Design a RAG-powered support bot that answers customer questions using help docs and past support tickets."

### 1. Clarify Requirements

- Query volume? (Say: 10,000/day)
- Response time SLA? (Say: < 3 seconds)
- Escalation needed? (Say: yes, to human agents)
- Multiple languages? (Say: English + Spanish)

### 2. High-Level Architecture

```
┌───────────────────────────────────────────────────────┐
│                  KNOWLEDGE SOURCES                     │
├───────────────────────────────────────────────────────┤
│  Help Center    Support Tickets    Product Docs       │
│  (Zendesk)      (Historical)       (Notion)           │
│      │               │                 │              │
│      └───────────────┼─────────────────┘              │
│                      ▼                                │
│            Unified Vector Store                       │
└───────────────────────────────────────────────────────┘

┌───────────────────────────────────────────────────────┐
│                   QUERY HANDLING                       │
├───────────────────────────────────────────────────────┤
│                                                       │
│  Customer  ──►  Intent    ──►  RAG or    ──►  Response│
│  Message       Classifier     Direct?        Generator│
│                    │                              │    │
│                    ▼                              ▼    │
│              "Returns"  ───────────────►  Policy RAG  │
│              "Order status" ──────────►  API Call     │
│              "Angry" ─────────────────►  Escalate     │
│                                                       │
└───────────────────────────────────────────────────────┘
```

### 3. Key Design Decisions

**Intent Classification First:**
```python
def handle_message(message: str, user: User):
    intent = classify_intent(message)
    
    if intent == "order_status":
        return get_order_status(user.order_id)
    elif intent == "angry_or_urgent":
        return escalate_to_human(message, user)
    elif intent == "general_question":
        return rag_query(message)
```

**Multi-source Retrieval:**
```python
# Combine sources with weights
retrievers = EnsembleRetriever(
    retrievers=[
        help_docs_retriever,      # Official docs
        ticket_retriever,         # Past solutions
    ],
    weights=[0.7, 0.3]  # Prefer official docs
)
```

**Confidence Threshold:**
```python
def generate_response(query: str):
    docs = retriever.search(query)
    
    if not docs or max_score(docs) < 0.5:
        return {
            "answer": "I'm not sure about this. Let me connect you with an agent.",
            "escalate": True
        }
    
    return rag_chain.query(query, docs)
```

### 4. Performance Requirements

- **< 3s response:** Use streaming, gpt-4o-mini
- **Caching:** Cache common questions (password reset, returns)
- **Warm start:** Keep connections alive

---

## Scenario 3: Legal Document Analysis

**Prompt:** "Design a system for lawyers to search and analyze contracts."

### 1. Key Challenges

- **Precision critical:** Wrong answer = liability
- **Long documents:** Contracts can be 100+ pages
- **Sensitive data:** Can't send to external APIs

### 2. Architecture Decisions

```
┌───────────────────────────────────────────────────────┐
│                  ON-PREMISE DEPLOYMENT                 │
├───────────────────────────────────────────────────────┤
│                                                       │
│   Documents ──► OCR ──► Parser ──► Chunker ──► FAISS │
│   (Scanned)    if     (Contract-    (Clause-  (Local) │
│                needed   aware)       based)           │
│                                                       │
│   Query ──► Local Embeddings ──► FAISS ──► Local LLM │
│            (E5-large)                    (Llama 2)    │
│                                                       │
└───────────────────────────────────────────────────────┘
```

**Key Decisions:**
- **Local everything:** No data leaves premises
- **Clause-aware chunking:** Keep contract sections intact
- **Human-in-the-loop:** Always show sources, require confirmation

**Chunking for Contracts:**
```python
def chunk_contract(text: str):
    # Split on section headers
    sections = re.split(r'\n(?=\d+\.\s+[A-Z])', text)
    
    chunks = []
    for section in sections:
        if len(section) > 2000:
            # Sub-chunk large sections
            chunks.extend(recursive_split(section))
        else:
            chunks.append(section)
    
    return chunks
```

### 3. Trade-offs

| Decision | Benefit | Cost |
|----------|---------|------|
| Local LLM | Data privacy | Lower quality than GPT-4 |
| Clause-based chunking | Better retrieval | Complex parsing |
| Human review required | Accuracy | Slower workflow |

---

## Scenario 4: Multi-Modal RAG (Bonus)

**Prompt:** "Design a RAG system that handles text, tables, and images in documents."

### Key Components

**Document Processing:**
```python
def process_document(doc_path: str):
    results = []
    
    # Text extraction
    text_chunks = extract_text_chunks(doc_path)
    results.extend(text_chunks)
    
    # Table extraction
    tables = extract_tables(doc_path)  # As markdown
    for table in tables:
        results.append({
            "content": table.to_markdown(),
            "type": "table",
            "page": table.page
        })
    
    # Image extraction + captioning
    images = extract_images(doc_path)
    for img in images:
        caption = vision_model.describe(img)
        results.append({
            "content": caption,
            "type": "image_caption",
            "image_path": img.path
        })
    
    return results
```

**Query Handling:**
```python
def query(question: str):
    # Retrieve all types
    results = retriever.search(question)
    
    # Format context appropriately
    context = ""
    for r in results:
        if r.type == "table":
            context += f"[TABLE]\n{r.content}\n"
        elif r.type == "image_caption":
            context += f"[IMAGE: {r.content}]\n"
        else:
            context += r.content + "\n"
    
    return llm.generate(context, question)
```

---

## Common Follow-Up Questions

**"How would you handle updates?"**
- Batch re-index nightly for bulk changes
- Incremental updates for new documents
- Version tracking to know what's current

**"What if latency is too high?"**
- Profile each component
- Cache aggressively
- Use faster models
- Pre-compute common queries

**"How do you ensure accuracy?"**
- Require source citations
- Confidence thresholds
- Human review for critical queries
- A/B testing against baselines

**"What about cost?"**
- Embed once, query many times
- Use smaller models where possible
- Cache cache cache
- Batch operations

---

## System Design Checklist

- [ ] Clarified requirements (scale, latency, security)
- [ ] Drew high-level architecture
- [ ] Explained component choices
- [ ] Addressed data flow (ingestion + query)
- [ ] Discussed scale considerations
- [ ] Mentioned monitoring/observability
- [ ] Covered failure modes
- [ ] Stated trade-offs made

---

## Next Up

Section 3: Code Explanation Practice — walking through your implementation.
