# 2.3 Chunking Strategies

## Why Chunking Is Critical

Chunking is one of the most impactful decisions in RAG system design. Get it wrong, and:

- **Chunks too small**: Miss context, retrieval finds fragments that don't make sense alone
- **Chunks too large**: Retrieval is imprecise, irrelevant info dilutes the good stuff
- **Bad split points**: Cut sentences mid-thought, break apart related information

There's no universal "best" chunk size. It depends on your documents, your queries, and your LLM's context window.

---

## The Core Parameters

### Chunk Size
How big each piece is (in characters or tokens).

- **Typical range**: 500-2000 characters (roughly 100-500 tokens)
- **Smaller chunks**: More precise retrieval, but less context
- **Larger chunks**: More context, but may include irrelevant info

### Chunk Overlap
How much adjacent chunks share.

```
Document: [AAAA][BBBB][CCCC][DDDD]

No overlap:     [AAAA] [BBBB] [CCCC] [DDDD]
200 overlap:    [AAAABB] [BBCCCC] [CCCCDD] [DDDD]
```

Overlap ensures that if important info is at a chunk boundary, it appears in at least one complete chunk.

- **Typical range**: 10-20% of chunk size
- **Example**: chunk_size=1000, chunk_overlap=200

---

## LangChain Text Splitters

### RecursiveCharacterTextSplitter (Most Common)

Tries to split on natural boundaries, falling back to smaller separators:

```python
from langchain.text_splitter import RecursiveCharacterTextSplitter

splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    separators=["\n\n", "\n", ". ", " ", ""]  # Try these in order
)

chunks = splitter.split_documents(documents)
```

**How it works:**
1. Try to split on `\n\n` (paragraphs)
2. If chunks are still too big, split on `\n` (lines)
3. If still too big, split on `. ` (sentences)
4. Last resort: split on spaces or characters

This preserves semantic units as much as possible.

### CharacterTextSplitter

Simple fixed-size splitting:

```python
from langchain.text_splitter import CharacterTextSplitter

splitter = CharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    separator="\n"  # Split only on this
)
```

Less intelligent, but predictable.

### TokenTextSplitter

Splits by token count (important for staying within LLM limits):

```python
from langchain.text_splitter import TokenTextSplitter

splitter = TokenTextSplitter(
    chunk_size=500,  # 500 tokens
    chunk_overlap=50
)
```

Use this when precise token counting matters.

### MarkdownTextSplitter

Respects Markdown structure:

```python
from langchain.text_splitter import MarkdownTextSplitter

splitter = MarkdownTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)
```

Splits on headers, code blocks, and other Markdown elements.

---

## Choosing Chunk Size: A Framework

### Start Here

```python
# Good default for most use cases
splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,    # ~250 words, ~200 tokens
    chunk_overlap=200   # 20% overlap
)
```

### Adjust Based On

| Factor | Smaller Chunks | Larger Chunks |
|--------|----------------|---------------|
| **Query type** | Specific facts | Broad concepts |
| **Document type** | Dense technical docs | Narrative text |
| **LLM context** | Limited (8K) | Large (128K) |
| **Retrieval k** | Higher k (5-10) | Lower k (2-4) |

### The Testing Approach

1. Start with defaults (1000 chars, 200 overlap)
2. Test with representative queries
3. Check: Do retrieved chunks contain the answer?
4. If chunks are too fragmented, increase size
5. If retrieval is imprecise, decrease size

---

## Advanced: Semantic Chunking

Instead of fixed sizes, split at topic boundaries.

### Using Sentence Embeddings

```python
from langchain_experimental.text_splitter import SemanticChunker
from langchain_openai import OpenAIEmbeddings

splitter = SemanticChunker(
    embeddings=OpenAIEmbeddings(),
    breakpoint_threshold_type="percentile",
    breakpoint_threshold_amount=95
)

chunks = splitter.split_documents(documents)
```

**How it works:**
1. Embed each sentence
2. Compare adjacent sentences' similarity
3. When similarity drops significantly, start a new chunk

**Pros:** More semantically coherent chunks
**Cons:** Slower (requires embedding), chunk sizes vary

---

## Advanced: Hierarchical Chunking

Create chunks at multiple levels:

```python
# Parent chunks (larger, for context)
parent_splitter = RecursiveCharacterTextSplitter(
    chunk_size=2000,
    chunk_overlap=0
)

# Child chunks (smaller, for precise retrieval)
child_splitter = RecursiveCharacterTextSplitter(
    chunk_size=400,
    chunk_overlap=0
)

# Split parents
parent_chunks = parent_splitter.split_documents(documents)

# Split each parent into children, tracking relationship
all_chunks = []
for parent in parent_chunks:
    children = child_splitter.split_documents([parent])
    for child in children:
        child.metadata["parent_id"] = id(parent)
    all_chunks.extend(children)
```

**Use case:** Retrieve on children (precise), but pass parent (more context) to LLM.

---

## Practical Examples

### Example 1: Dense Financial Policies

Financial documents are dense with defined terms and precise language.

```python
splitter = RecursiveCharacterTextSplitter(
    chunk_size=800,      # Smaller chunks
    chunk_overlap=200,   # Good overlap
    separators=["\n\n", "\n", ". ", "; ", ", ", " "]
)
```

Why:
- Policies have many specific clauses
- Queries often ask about specific rules
- Smaller chunks = more precise retrieval

### Example 2: Narrative Reports

Annual reports with flowing prose:

```python
splitter = RecursiveCharacterTextSplitter(
    chunk_size=1500,     # Larger chunks
    chunk_overlap=300,   # More overlap for continuity
    separators=["\n\n", "\n", ". "]
)
```

Why:
- Ideas span multiple paragraphs
- Context matters for understanding
- Larger chunks preserve narrative flow

### Example 3: FAQs

Question-answer pairs:

```python
splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,      # Each Q&A is small
    chunk_overlap=0,     # No overlap needed
    separators=["\n\n"]  # Split on blank lines between Q&As
)
```

Why:
- Each Q&A is self-contained
- Don't want to mix Q&As
- No benefit from overlap

---

## Preserving Context with Metadata

Add contextual information to chunks:

```python
def add_context_to_chunks(chunks, document_title: str):
    """Add document context to each chunk."""
    for i, chunk in enumerate(chunks):
        # Add positional info
        chunk.metadata["chunk_index"] = i
        chunk.metadata["total_chunks"] = len(chunks)
        chunk.metadata["position"] = "beginning" if i < 3 else "end" if i > len(chunks) - 3 else "middle"
        
        # Prepend document title for context
        chunk.page_content = f"[From: {document_title}]\n\n{chunk.page_content}"
    
    return chunks
```

Now even out-of-context chunks know where they came from.

---

## Debugging Chunking Issues

### Symptom: Relevant content not retrieved

**Check:** Is the answer split across chunks?

```python
# Search for a specific phrase to find which chunk(s) contain it
phrase = "loan deferral"
for i, chunk in enumerate(chunks):
    if phrase.lower() in chunk.page_content.lower():
        print(f"Chunk {i}: ...{chunk.page_content[:200]}...")
```

**Fix:** Increase chunk size or overlap

### Symptom: Retrieved chunks are vague

**Check:** Are chunks too large and generic?

```python
# Check average chunk size
avg_size = sum(len(c.page_content) for c in chunks) / len(chunks)
print(f"Average chunk size: {avg_size:.0f} characters")
```

**Fix:** Decrease chunk size

### Symptom: Missing sentence starts/ends

**Check:** Are sentences being cut mid-thought?

```python
# Check for incomplete sentences
for chunk in chunks[:5]:
    first_50 = chunk.page_content[:50]
    last_50 = chunk.page_content[-50:]
    print(f"Start: {first_50}")
    print(f"End: {last_50}")
    print("---")
```

**Fix:** Ensure `. ` is in your separators and overlap is sufficient

---

## Chunking in FinanceRAG

Our approach for financial documents:

```python
# In document_processor.py
from langchain.text_splitter import RecursiveCharacterTextSplitter

def chunk_documents(documents):
    """Chunk documents optimized for financial content."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        separators=[
            "\n## ",    # Markdown headers
            "\n### ",
            "\n\n",     # Paragraphs
            "\n",       # Lines
            ". ",       # Sentences
            "; ",       # Clauses (common in legal/policy text)
            ", ",       # Phrases
            " ",        # Words
            ""          # Characters
        ],
        length_function=len,
    )
    
    chunks = splitter.split_documents(documents)
    
    # Add chunk metadata
    for i, chunk in enumerate(chunks):
        chunk.metadata["chunk_id"] = i
    
    return chunks
```

---

## What's Next?

You've loaded documents and chunked them intelligently. Now it's time to convert those chunks into searchable vectors.

Which embedding model should you use? OpenAI's paid API? A free local model? How do they compare?

→ [2.4 Embedding Models](./04-embedding-models.md)

---

## 💡 Interview Tip

> **Q: "How do you decide on chunk size for a RAG system?"**
>
> **A:** "I start with a default of around 1000 characters with 200 overlap, then adjust based on testing. Key factors are: (1) Document type — dense technical docs benefit from smaller chunks, narrative text needs larger chunks to preserve context; (2) Query type — specific fact-finding queries work better with smaller chunks, broad questions need more context; (3) Retrieval k — if I'm retrieving more documents, I can use smaller chunks. I always test with representative queries and check whether the retrieved chunks actually contain the answer. If important information is getting split across chunks, I increase overlap or size."

> **Q: "What's chunk overlap and why does it matter?"**
>
> **A:** "Chunk overlap is how much text adjacent chunks share. If you have chunks of 1000 characters with 200 overlap, the last 200 characters of chunk 1 are the first 200 of chunk 2. This matters because important information might fall at a chunk boundary — without overlap, it could be split and neither chunk would have the complete thought. Overlap ensures continuity and prevents information loss at boundaries. I typically use 10-20% overlap."

