# 1.1 What is RAG and Why It Matters

## The Problem RAG Solves

Imagine you're a new employee at a bank. On day one, you're smart, well-educated, and capable of reasoning through problems. But you don't know anything about *this specific bank* — its policies, its products, its internal processes.

Now someone asks you: "What's our policy on loan deferrals for customers affected by natural disasters?"

You could try to reason through it based on general knowledge. But you'd probably be wrong. What you *need* is access to the bank's policy documents. You need to **retrieve** the relevant information before you **generate** a response.

**This is exactly the problem Large Language Models (LLMs) face.**

LLMs like GPT-4 or Claude are trained on vast amounts of public data. They're incredibly capable reasoners. But they don't know about:
- Your company's internal documents
- Recent events (after their training cutoff)
- Proprietary data you haven't shared with them

**RAG (Retrieval-Augmented Generation)** solves this by giving the LLM access to external knowledge at query time.

---

## The RAG Pipeline: A Mental Model

Think of RAG as a two-step process:

```
User Question → [RETRIEVE relevant documents] → [GENERATE answer using documents + LLM]
```

Let's break this down:

### Step 1: Retrieval
When a user asks a question, we first search a knowledge base to find documents (or chunks of documents) that are relevant to the question.

This isn't keyword search like Google circa 2005. It's **semantic search** — finding documents that are conceptually similar, even if they don't share exact words.

Example:
- Query: "What happens if a customer can't pay their mortgage?"
- Relevant doc: "Loan deferral policy for hardship cases..."

The words are different, but the *meaning* overlaps. Semantic search catches this.

### Step 2: Generation
Once we have relevant documents, we pass them to the LLM along with the user's question. The prompt looks something like:

```
Use the following context to answer the question.

Context:
[Retrieved Document 1]
[Retrieved Document 2]
[Retrieved Document 3]

Question: What happens if a customer can't pay their mortgage?

Answer:
```

The LLM now has the specific knowledge it needs. It can generate an accurate, grounded response.

---

## Why Not Just Put Everything in the Prompt?

Good question. You might think: "Why not just paste all my documents into the prompt?"

Three reasons:

### 1. Context Window Limits
LLMs have a maximum number of tokens they can process at once. GPT-4-turbo handles ~128K tokens. That sounds like a lot, but:
- A single 50-page policy document might be 20K tokens
- An enterprise might have thousands of documents
- You'd blow past limits instantly

### 2. Cost
LLM APIs charge by the token. Sending 100K tokens with every query gets expensive fast. RAG lets you send only the *relevant* 2-3K tokens.

### 3. Relevance
More context isn't always better. LLMs can get confused or distracted by irrelevant information. Retrieving only relevant chunks keeps the signal-to-noise ratio high.

---

## The RAG Architecture (Visual)

```
┌─────────────────────────────────────────────────────────────────┐
│                         RAG SYSTEM                               │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌──────────────┐                                               │
│  │   Documents  │                                               │
│  │  (PDF, DOCX, │                                               │
│  │   Markdown)  │                                               │
│  └──────┬───────┘                                               │
│         │                                                        │
│         ▼                                                        │
│  ┌──────────────┐     ┌──────────────┐     ┌──────────────┐    │
│  │   Chunking   │────▶│  Embedding   │────▶│ Vector Store │    │
│  │  (Split into │     │   Model      │     │  (ChromaDB)  │    │
│  │   pieces)    │     │              │     │              │    │
│  └──────────────┘     └──────────────┘     └──────┬───────┘    │
│                                                    │            │
│         ┌──────────────────────────────────────────┘            │
│         │                                                        │
│         │  INGESTION (happens once, or when docs change)        │
│  ───────┼────────────────────────────────────────────────────   │
│         │  QUERY TIME (happens on every user question)          │
│         │                                                        │
│         ▼                                                        │
│  ┌──────────────┐     ┌──────────────┐     ┌──────────────┐    │
│  │    User      │────▶│   Embed      │────▶│   Retrieve   │    │
│  │   Question   │     │   Query      │     │   Top K      │    │
│  └──────────────┘     └──────────────┘     └──────┬───────┘    │
│                                                    │            │
│                                                    ▼            │
│                                            ┌──────────────┐    │
│                                            │   Context +  │    │
│                                            │   Question   │    │
│                                            └──────┬───────┘    │
│                                                    │            │
│                                                    ▼            │
│                                            ┌──────────────┐    │
│                                            │     LLM      │    │
│                                            │  (GPT-4)     │    │
│                                            └──────┬───────┘    │
│                                                    │            │
│                                                    ▼            │
│                                            ┌──────────────┐    │
│                                            │   Answer     │    │
│                                            └──────────────┘    │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

---

## Key Components We'll Deep Dive

| Component | What It Does | We'll Cover In |
|-----------|--------------|----------------|
| **Document Loader** | Reads files (PDF, DOCX, etc.) | Part 2.2 |
| **Text Splitter** | Chunks documents into pieces | Part 2.3 |
| **Embedding Model** | Converts text to vectors | Part 1.2, 2.4 |
| **Vector Store** | Stores and searches vectors | Part 1.3, 2.5 |
| **Retriever** | Finds relevant chunks | Part 2.6 |
| **LLM** | Generates final answer | Part 1.4 |

---

## Why RAG for Financial Services?

RAG is particularly powerful in regulated industries like finance:

### 1. Compliance & Accuracy
Financial advice needs to be grounded in actual policy. RAG ensures answers come from authoritative sources, not LLM hallucinations.

### 2. Auditability
With RAG, you can show *which documents* informed an answer. This matters for compliance and legal review.

### 3. Fresh Information
Policies change. Regulations update. RAG lets you update your knowledge base without retraining models.

### 4. Data Privacy
Sensitive documents stay in your vector store — they're never sent to OpenAI for training. You control your data.

---

## The Terminology Cheat Sheet

| Term | Plain English |
|------|---------------|
| **RAG** | Retrieval-Augmented Generation — look stuff up, then answer |
| **Embedding** | Converting text to a list of numbers that capture meaning |
| **Vector** | That list of numbers (e.g., 1536 floats for OpenAI's model) |
| **Vector Store** | A database optimized for searching vectors |
| **Chunk** | A piece of a document (usually a few paragraphs) |
| **Context Window** | Maximum tokens an LLM can handle at once |
| **Semantic Search** | Finding similar meaning, not just matching keywords |
| **Top-K** | Retrieve the K most relevant chunks (usually 3-5) |

---

## What's Next?

Now you understand the *what* and *why* of RAG. Next, we need to understand the magic behind semantic search: **embeddings**.

How do we convert text into numbers that capture meaning? How does the math work? Why do similar concepts end up as nearby vectors?

→ [1.2 Embeddings Explained](./02-embeddings-explained.md)

---

## 💡 Interview Tip

> **Q: "Explain RAG in simple terms."**
>
> **A:** "RAG stands for Retrieval-Augmented Generation. Instead of relying only on what an LLM learned during training, we first search a knowledge base to find relevant documents, then pass those documents to the LLM along with the user's question. This grounds the LLM's response in actual data, reduces hallucinations, and lets us use private or recent information the model was never trained on."

