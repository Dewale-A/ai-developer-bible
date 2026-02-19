# Part 3, Section 3: Document Processor

## What You'll Learn
- How to load different file types
- How chunking actually works
- Adding metadata to chunks

---

## The Document Processor Flow

```
File(s)               Load              Chunk            Ready for
  │                    │                  │              Embedding
  ▼                    ▼                  ▼                  ▼
[PDF/MD/TXT] → [LangChain Loader] → [Text Splitter] → [Document Objects]
```

---

## File: `src/tools/document_processor.py`

### Part 1: Imports

```python
import os
from pathlib import Path
from typing import List, Optional
import structlog

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_community.document_loaders import (
    TextLoader,
    PyPDFLoader,
    Docx2txtLoader,
    UnstructuredMarkdownLoader,
)

from src.config.settings import get_settings

logger = structlog.get_logger()
```

**Key imports explained:**

| Import | Purpose |
|--------|---------|
| `RecursiveCharacterTextSplitter` | Splits text into chunks intelligently |
| `Document` | LangChain's document container (text + metadata) |
| `TextLoader`, `PyPDFLoader`, etc. | File-type specific loaders |
| `structlog` | Structured logging (better than print!) |

---

### Part 2: Class Initialization

```python
class DocumentProcessor:
    """Process and chunk documents for RAG pipeline."""
    
    def __init__(self):
        self.settings = get_settings()
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.settings.chunk_size,       # 1000 chars
            chunk_overlap=self.settings.chunk_overlap,  # 200 chars
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )
```

**The `separators` list is key:**

```
Priority order for splitting:
1. "\n\n"  →  Try paragraph breaks first
2. "\n"    →  Then line breaks
3. " "     →  Then spaces
4. ""      →  Last resort: character by character
```

**Why this order?** Keeps related content together. A paragraph is better than a random cut mid-sentence.

---

```python
        # Supported file extensions and their loaders
        self.loaders = {
            ".txt": TextLoader,
            ".pdf": PyPDFLoader,
            ".docx": Docx2txtLoader,
            ".md": UnstructuredMarkdownLoader,
        }
```

**Loader mapping:** File extension → appropriate loader class.

Adding a new format? Just add to this dictionary:
```python
".csv": CSVLoader,
".html": BSHTMLLoader,
```

---

### Part 3: Loading a Single Document

```python
def load_document(self, file_path: str) -> List[Document]:
    """Load a single document from file path."""
    path = Path(file_path)
    
    # Check file exists
    if not path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")
    
    # Get the right loader
    extension = path.suffix.lower()  # ".pdf" → ".pdf"
    
    if extension not in self.loaders:
        raise ValueError(f"Unsupported file type: {extension}")
    
    loader_class = self.loaders[extension]
```

**So far:** Validate the file, pick the right loader.

```python
    try:
        loader = loader_class(str(path))
        documents = loader.load()
        
        # Add metadata to each document
        for doc in documents:
            doc.metadata["source"] = str(path)
            doc.metadata["filename"] = path.name
            doc.metadata["file_type"] = extension
        
        logger.info("document_loaded", file=path.name, pages=len(documents))
        return documents
        
    except Exception as e:
        logger.error("document_load_error", file=path.name, error=str(e))
        raise
```

**What `loader.load()` returns:**
```python
[
    Document(
        page_content="The actual text content...",
        metadata={"source": "file.pdf", "page": 1}
    ),
    Document(...),  # page 2
    ...
]
```

**We add extra metadata:** filename, file_type for later filtering.

---

### Part 4: Loading a Directory

```python
def load_directory(self, directory_path: str) -> List[Document]:
    """Load all supported documents from a directory."""
    path = Path(directory_path)
    
    if not path.exists():
        raise FileNotFoundError(f"Directory not found: {directory_path}")
    
    if not path.is_dir():
        raise ValueError(f"Path is not a directory: {directory_path}")
    
    all_documents = []
    
    # Loop through each supported file type
    for extension in self.loaders.keys():
        for file_path in path.glob(f"*{extension}"):
            try:
                docs = self.load_document(str(file_path))
                all_documents.extend(docs)
            except Exception as e:
                logger.warning("skip_file", file=file_path.name, error=str(e))
                continue  # Skip bad files, don't crash
    
    return all_documents
```

**Key behavior:** Bad files get logged and skipped, not crash the whole process.

---

### Part 5: Chunking Documents

```python
def chunk_documents(self, documents: List[Document]) -> List[Document]:
    """Split documents into chunks."""
    chunks = self.text_splitter.split_documents(documents)
    
    # Add chunk metadata
    for i, chunk in enumerate(chunks):
        chunk.metadata["chunk_index"] = i
        chunk.metadata["chunk_size"] = len(chunk.page_content)
    
    logger.info("documents_chunked", 
               input_docs=len(documents), 
               output_chunks=len(chunks))
    
    return chunks
```

**What `split_documents()` does:**

```
Input:  1 document with 5000 characters
        chunk_size=1000, overlap=200

Output: 6 chunks
        Chunk 0: chars 0-1000
        Chunk 1: chars 800-1800    ← 200 char overlap!
        Chunk 2: chars 1600-2600
        ...
```

**Metadata added:**
- `chunk_index` — which chunk is this (0, 1, 2...)
- `chunk_size` — actual size (might be less than max)

---

### Part 6: Main Entry Point

```python
def process_documents(self, source_path: str) -> List[Document]:
    """Load and chunk documents from file or directory."""
    path = Path(source_path)
    
    if path.is_file():
        documents = self.load_document(source_path)
    elif path.is_dir():
        documents = self.load_directory(source_path)
    else:
        raise ValueError(f"Invalid path: {source_path}")
    
    chunks = self.chunk_documents(documents)
    
    return chunks
```

**Simple logic:**
1. Is it a file? Load that file.
2. Is it a directory? Load all files in it.
3. Either way → chunk the results.

---

## Visual Example

```
Input: loan_policy.md (3000 chars)

After Loading:
┌────────────────────────────────────────────┐
│ Document                                    │
│   page_content: "Loan Policy... (3000 ch)" │
│   metadata: {filename: "loan_policy.md"}   │
└────────────────────────────────────────────┘

After Chunking (chunk_size=1000, overlap=200):
┌──────────────────┐
│ Chunk 0          │
│ chars 0-1000     │
│ metadata: {      │
│   filename: ..., │
│   chunk_index: 0 │
│ }                │
└──────────────────┘
         ↓ overlap
┌──────────────────┐
│ Chunk 1          │
│ chars 800-1800   │
│ metadata: {      │
│   chunk_index: 1 │
│ }                │
└──────────────────┘
         ↓ overlap
┌──────────────────┐
│ Chunk 2          │
│ chars 1600-2600  │
│ ...              │
└──────────────────┘
```

---

## Interview Questions

**Q: Why use RecursiveCharacterTextSplitter instead of just splitting every N characters?**

A: Recursive splitter tries to keep semantic units together:
- First tries to split on paragraphs (`\n\n`)
- Falls back to sentences, then words
- Result: chunks that make sense, not cut mid-word

**Q: Why add chunk overlap?**

A: Context preservation. If a sentence spans two chunks, both chunks get it. Prevents losing information at boundaries.

**Q: What metadata would you add for a production system?**

A: Beyond what we have:
- `created_at` — when document was added
- `document_id` — unique identifier
- `page_number` — for PDFs
- `section_title` — if parseable
- `author`, `version` — if available

**Q: How would you handle a 500MB PDF?**

A: Stream it! Load page by page instead of all at once:
```python
# Instead of loader.load()
for page in loader.lazy_load():
    process_page(page)
```

---

## Next Up

Section 4: `vector_store.py` — storing and searching embeddings with ChromaDB.
