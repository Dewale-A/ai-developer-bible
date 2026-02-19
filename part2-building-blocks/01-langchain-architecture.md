# 2.1 LangChain Architecture

## What Is LangChain?

LangChain is a framework for building applications with LLMs. Think of it as the "plumbing" that connects:
- Document loaders
- Text splitters
- Embedding models
- Vector stores
- LLMs
- Chains and agents

Instead of writing custom code to connect OpenAI to ChromaDB to your PDFs, LangChain provides standardized interfaces.

---

## Why Use LangChain?

### The Problem Without It

```python
# Without LangChain: lots of custom glue code

# Load PDF (need a PDF library)
import pypdf
pdf = pypdf.PdfReader("doc.pdf")
text = "".join(page.extract_text() for page in pdf.pages)

# Chunk (write your own logic)
chunks = []
for i in range(0, len(text), 1000):
    chunks.append(text[i:i+1000])

# Embed (call OpenAI directly)
import openai
embeddings = []
for chunk in chunks:
    response = openai.embeddings.create(model="text-embedding-ada-002", input=chunk)
    embeddings.append(response.data[0].embedding)

# Store (manage ChromaDB directly)
import chromadb
client = chromadb.Client()
collection = client.create_collection("docs")
for i, (chunk, emb) in enumerate(zip(chunks, embeddings)):
    collection.add(ids=[str(i)], embeddings=[emb], documents=[chunk])

# Query (more manual work)
query_emb = openai.embeddings.create(model="text-embedding-ada-002", input="my question").data[0].embedding
results = collection.query(query_embeddings=[query_emb], n_results=3)

# Generate (build prompt manually)
context = "\n".join(results['documents'][0])
prompt = f"Context: {context}\n\nQuestion: my question\n\nAnswer:"
response = openai.chat.completions.create(model="gpt-4", messages=[{"role": "user", "content": prompt}])
```

That's a lot of code, and we haven't handled errors, different file types, or any advanced features.

### With LangChain

```python
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA

# Load & chunk
docs = PyPDFLoader("doc.pdf").load()
chunks = RecursiveCharacterTextSplitter(chunk_size=1000).split_documents(docs)

# Embed & store
vectorstore = Chroma.from_documents(chunks, OpenAIEmbeddings())

# Query
chain = RetrievalQA.from_chain_type(ChatOpenAI(), retriever=vectorstore.as_retriever())
answer = chain.invoke({"query": "my question"})
```

Same functionality, fraction of the code, and it handles edge cases.

---

## Core Abstractions

LangChain is built around these key concepts:

### 1. Document

A piece of text with metadata.

```python
from langchain.schema import Document

doc = Document(
    page_content="The loan policy states that...",
    metadata={"source": "loan_policy.pdf", "page": 1}
)
```

### 2. Document Loaders

Convert files into Documents.

```python
from langchain_community.document_loaders import (
    PyPDFLoader,      # PDFs
    Docx2txtLoader,   # Word docs
    TextLoader,       # Plain text
    CSVLoader,        # CSV files
    UnstructuredMarkdownLoader  # Markdown
)

# Each returns a list of Documents
docs = PyPDFLoader("file.pdf").load()
```

### 3. Text Splitters

Break Documents into smaller chunks.

```python
from langchain.text_splitter import (
    RecursiveCharacterTextSplitter,  # Most common
    CharacterTextSplitter,            # Simple fixed-size
    TokenTextSplitter,                # By token count
    MarkdownTextSplitter,             # Markdown-aware
)

splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
chunks = splitter.split_documents(docs)
```

### 4. Embeddings

Convert text to vectors.

```python
from langchain_openai import OpenAIEmbeddings
from langchain_community.embeddings import HuggingFaceEmbeddings

# OpenAI (API-based)
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

# HuggingFace (local)
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# Use them
vector = embeddings.embed_query("some text")  # Single text
vectors = embeddings.embed_documents(["text1", "text2"])  # Multiple
```

### 5. Vector Stores

Store and search embeddings.

```python
from langchain_community.vectorstores import (
    Chroma,     # Easy, local
    Pinecone,   # Managed, scalable
    FAISS,      # Fast, local
)

# Create from documents (embeds automatically)
vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    persist_directory="./db"
)

# Search
results = vectorstore.similarity_search("query", k=4)
```

### 6. Retrievers

An abstraction over vector stores for retrieving documents.

```python
# Basic retriever
retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

# Get relevant documents
docs = retriever.invoke("my question")
```

Why use a retriever instead of calling the vector store directly? Retrievers can be:
- Configured with different search strategies
- Combined (ensemble retrieval)
- Extended with re-ranking

### 7. LLMs and Chat Models

Interface to language models.

```python
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic

# OpenAI
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# Anthropic
llm = ChatAnthropic(model="claude-3-sonnet")

# Use them
response = llm.invoke("What is 2+2?")
```

### 8. Prompts

Templates for constructing prompts.

```python
from langchain.prompts import PromptTemplate, ChatPromptTemplate

# Simple template
template = PromptTemplate.from_template(
    "Answer the question based on the context.\n\nContext: {context}\n\nQuestion: {question}"
)
prompt = template.format(context="...", question="...")

# Chat template (with system message)
chat_template = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant. Answer only based on the provided context."),
    ("human", "Context: {context}\n\nQuestion: {question}")
])
```

### 9. Chains

Connect components into a pipeline.

```python
from langchain.chains import RetrievalQA, LLMChain

# RetrievalQA: retriever + LLM
chain = RetrievalQA.from_chain_type(
    llm=ChatOpenAI(),
    retriever=vectorstore.as_retriever()
)

# Or build custom chains with LCEL (LangChain Expression Language)
from langchain_core.runnables import RunnablePassthrough

chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt
    | llm
)
```

---

## LangChain Expression Language (LCEL)

LCEL is the modern way to build chains in LangChain. It uses the `|` pipe operator.

### Basic LCEL Chain

```python
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# Components
retriever = vectorstore.as_retriever()
prompt = ChatPromptTemplate.from_template("""
Answer based on the context.

Context: {context}

Question: {question}
""")
llm = ChatOpenAI()

# Chain them with |
chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

# Use it
answer = chain.invoke("What is the loan policy?")
```

### How LCEL Works

The `|` operator connects runnables. Each runnable:
1. Receives input from the previous step
2. Processes it
3. Passes output to the next step

```python
# This chain:
chain = retriever | prompt | llm | parser

# Is equivalent to:
def chain(input):
    step1 = retriever.invoke(input)
    step2 = prompt.invoke(step1)
    step3 = llm.invoke(step2)
    step4 = parser.invoke(step3)
    return step4
```

### Why LCEL?

- **Composable**: Easy to swap components
- **Streaming**: Built-in support for streaming responses
- **Async**: Works with async/await
- **Tracing**: Automatic integration with LangSmith for debugging

---

## LangChain Package Structure

LangChain split into multiple packages (this confuses people):

| Package | What It Contains |
|---------|-----------------|
| `langchain-core` | Base abstractions (Document, Runnable, etc.) |
| `langchain` | Chains, agents, high-level components |
| `langchain-community` | Third-party integrations (loaders, vector stores) |
| `langchain-openai` | OpenAI-specific components |
| `langchain-anthropic` | Anthropic-specific components |

### Import Patterns

```python
# Core types
from langchain.schema import Document
from langchain_core.runnables import RunnablePassthrough

# OpenAI (dedicated package)
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

# Community integrations
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma

# High-level chains
from langchain.chains import RetrievalQA
```

---

## How FinanceRAG Uses LangChain

Our project structure maps to LangChain components:

```
FinanceRAG/
├── src/
│   ├── tools/
│   │   ├── document_processor.py  
│   │   │   └── Uses: PyPDFLoader, MarkdownLoader, RecursiveCharacterTextSplitter
│   │   │
│   │   ├── vector_store.py        
│   │   │   └── Uses: Chroma, OpenAIEmbeddings
│   │   │
│   │   └── rag_chain.py           
│   │       └── Uses: ChatOpenAI, RetrievalQA or LCEL chain
```

We've organized it this way because:
1. **Separation of concerns**: Each file handles one responsibility
2. **Testability**: Can test document processing without the full chain
3. **Flexibility**: Easy to swap components (different embeddings, different LLM)

---

## Common LangChain Patterns

### Pattern 1: Logging and Debugging

```python
import langchain
langchain.debug = True  # Print all chain steps

# Or use callbacks
from langchain.callbacks import StdOutCallbackHandler
chain.invoke(query, config={"callbacks": [StdOutCallbackHandler()]})
```

### Pattern 2: Streaming Responses

```python
# Stream tokens as they're generated
for chunk in chain.stream("my question"):
    print(chunk, end="", flush=True)
```

### Pattern 3: Batch Processing

```python
# Process multiple queries at once
questions = ["Q1?", "Q2?", "Q3?"]
answers = chain.batch(questions)
```

### Pattern 4: Fallbacks

```python
# If GPT-4 fails, fall back to GPT-3.5
from langchain_openai import ChatOpenAI

primary = ChatOpenAI(model="gpt-4")
fallback = ChatOpenAI(model="gpt-3.5-turbo")

chain = primary.with_fallbacks([fallback])
```

---

## What's Next?

Now you understand how LangChain organizes all the pieces. Next, we'll dive into the first step of the ingestion pipeline: **loading documents**.

How do you read PDFs, Word docs, and Markdown files? What metadata should you preserve?

→ [2.2 Document Loading & Processing](./02-document-loading.md)

---

## 💡 Interview Tip

> **Q: "Why use LangChain instead of calling APIs directly?"**
>
> **A:** "LangChain provides standardized abstractions that make it easy to swap components — for example, switching from OpenAI embeddings to an open-source model, or from ChromaDB to Pinecone, often requires changing just one line. It also handles common patterns like chunking, retrieval, and prompt construction out of the box, which significantly reduces development time. The tradeoff is an additional dependency and some abstraction overhead, but for most RAG applications, the productivity gain is worth it."

> **Q: "Explain LangChain Expression Language (LCEL)."**
>
> **A:** "LCEL is LangChain's declarative way of composing chains using the pipe operator. You connect components like `retriever | prompt | llm | parser`, and LCEL handles the data flow between them. The benefits are: it's readable, it supports streaming and async out of the box, and it integrates with LangSmith for debugging. Under the hood, each component is a 'Runnable' that receives input, processes it, and passes output to the next step."

