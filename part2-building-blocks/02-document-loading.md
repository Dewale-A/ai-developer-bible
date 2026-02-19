# 2.2 Document Loading & Processing

## The First Step: Getting Text Out of Files

Before you can embed and search documents, you need to extract the text from them. This sounds simple, but:

- PDFs can be scanned images, multi-column layouts, or have weird encodings
- Word docs have headers, footers, tables, and formatting
- Markdown has code blocks, links, and hierarchical structure

Document loading is often where RAG projects hit their first snag.

---

## LangChain Document Loaders

LangChain provides loaders for most file types:

### PDF Files

```python
from langchain_community.document_loaders import PyPDFLoader

loader = PyPDFLoader("policy.pdf")
documents = loader.load()

# Each page becomes a Document
for doc in documents:
    print(f"Page {doc.metadata['page']}: {doc.page_content[:100]}...")
```

**Output:**
```
Page 0: LOAN POLICY DOCUMENT\n\nSection 1: Introduction\n\nThis document outlines...
Page 1: Section 2: Eligibility Requirements\n\nApplicants must meet the following...
```

**What you get:**
- `page_content`: The text from that page
- `metadata`: `{"source": "policy.pdf", "page": 0}`

### Alternative PDF Loaders

Different loaders handle PDFs differently:

```python
# PyPDFLoader - Pure Python, fast, basic extraction
from langchain_community.document_loaders import PyPDFLoader

# PDFMiner - Better at complex layouts
from langchain_community.document_loaders import PDFMinerLoader

# Unstructured - Most robust, handles images/tables, requires more setup
from langchain_community.document_loaders import UnstructuredPDFLoader
```

**When to use which:**

| Loader | Speed | Quality | Setup |
|--------|-------|---------|-------|
| PyPDFLoader | Fast | Basic | Easy |
| PDFMinerLoader | Medium | Better layouts | Easy |
| UnstructuredPDFLoader | Slow | Best | Requires dependencies |

### Word Documents

```python
from langchain_community.document_loaders import Docx2txtLoader

loader = Docx2txtLoader("report.docx")
documents = loader.load()

# Returns one Document with all text
print(documents[0].page_content[:200])
```

### Markdown Files

```python
from langchain_community.document_loaders import UnstructuredMarkdownLoader

loader = UnstructuredMarkdownLoader("README.md")
documents = loader.load()
```

### Plain Text

```python
from langchain_community.document_loaders import TextLoader

loader = TextLoader("notes.txt", encoding="utf-8")
documents = loader.load()
```

### CSV Files

```python
from langchain_community.document_loaders import CSVLoader

loader = CSVLoader("data.csv")
documents = loader.load()  # Each row becomes a Document
```

---

## Loading Multiple Files

### Directory Loader

Load all files from a directory:

```python
from langchain_community.document_loaders import DirectoryLoader

# Load all PDFs from a folder
loader = DirectoryLoader(
    path="./documents",
    glob="**/*.pdf",        # Pattern to match
    loader_cls=PyPDFLoader  # Loader to use for each file
)
documents = loader.load()
print(f"Loaded {len(documents)} documents")
```

### Custom Multi-Format Loader

```python
from pathlib import Path
from langchain_community.document_loaders import (
    PyPDFLoader, 
    Docx2txtLoader, 
    UnstructuredMarkdownLoader
)

def load_document(file_path: str):
    """Load a document based on its file extension."""
    path = Path(file_path)
    
    loaders = {
        ".pdf": PyPDFLoader,
        ".docx": Docx2txtLoader,
        ".md": UnstructuredMarkdownLoader,
    }
    
    loader_class = loaders.get(path.suffix.lower())
    if loader_class is None:
        raise ValueError(f"Unsupported file type: {path.suffix}")
    
    loader = loader_class(str(path))
    return loader.load()

# Use it
docs = load_document("policy.pdf")
```

---

## Metadata: Why It Matters

Metadata travels with your documents through the entire pipeline. It's crucial for:

### 1. Source Attribution
Show users where answers come from:
```
"Based on loan_policy.pdf, page 5..."
```

### 2. Filtering
Search only specific sources:
```python
results = vectorstore.similarity_search(
    "loan requirements",
    filter={"source": "loan_policy.pdf"}
)
```

### 3. Debugging
When something's wrong, metadata tells you which document caused the issue.

### Adding Custom Metadata

```python
from langchain.schema import Document

# Load documents
docs = PyPDFLoader("policy.pdf").load()

# Enrich metadata
for doc in docs:
    doc.metadata["document_type"] = "policy"
    doc.metadata["department"] = "lending"
    doc.metadata["last_updated"] = "2024-01-15"
    doc.metadata["confidentiality"] = "internal"

# Now you can filter by any of these fields
```

---

## Handling Real-World Documents

### Problem 1: Scanned PDFs (Images)

If your PDF is a scanned image, text extraction returns nothing.

**Solution: OCR**

```python
# Using Unstructured with OCR
from langchain_community.document_loaders import UnstructuredPDFLoader

loader = UnstructuredPDFLoader(
    "scanned_doc.pdf",
    mode="elements",
    strategy="ocr_only"  # Force OCR
)
documents = loader.load()
```

### Problem 2: Tables

Tables in PDFs often become garbled text.

**Solution: Structured extraction**

```python
# Unstructured can preserve table structure
loader = UnstructuredPDFLoader(
    "report_with_tables.pdf",
    mode="elements"  # Separates tables from text
)
documents = loader.load()

# Tables become separate elements you can handle specially
for doc in documents:
    if doc.metadata.get("category") == "Table":
        # Handle table differently
        pass
```

### Problem 3: Headers and Footers

Repeated headers/footers add noise to every page.

**Solution: Post-processing**

```python
def clean_document(doc: Document) -> Document:
    """Remove common headers/footers."""
    text = doc.page_content
    
    # Remove common header patterns
    lines = text.split('\n')
    cleaned_lines = [
        line for line in lines
        if not line.strip().startswith("Page ")  # Page numbers
        and not line.strip() == "CONFIDENTIAL"    # Repeated headers
        and len(line.strip()) > 0                 # Empty lines
    ]
    
    doc.page_content = '\n'.join(cleaned_lines)
    return doc

# Apply to all documents
documents = [clean_document(doc) for doc in documents]
```

### Problem 4: Encoding Issues

Some files have weird encodings that cause errors.

**Solution: Specify encoding**

```python
from langchain_community.document_loaders import TextLoader

# Try common encodings
for encoding in ["utf-8", "latin-1", "cp1252"]:
    try:
        loader = TextLoader("file.txt", encoding=encoding)
        documents = loader.load()
        break
    except UnicodeDecodeError:
        continue
```

---

## Document Processing Pipeline

Here's a robust document processing function:

```python
from pathlib import Path
from typing import List
from langchain.schema import Document
from langchain_community.document_loaders import (
    PyPDFLoader,
    Docx2txtLoader,
    UnstructuredMarkdownLoader,
    TextLoader
)
import logging

logger = logging.getLogger(__name__)

LOADER_MAPPING = {
    ".pdf": PyPDFLoader,
    ".docx": Docx2txtLoader,
    ".doc": Docx2txtLoader,
    ".md": UnstructuredMarkdownLoader,
    ".txt": TextLoader,
}

def load_documents(
    file_paths: List[str],
    add_metadata: dict = None
) -> List[Document]:
    """
    Load documents from multiple file paths.
    
    Args:
        file_paths: List of paths to documents
        add_metadata: Additional metadata to add to all documents
    
    Returns:
        List of Document objects
    """
    all_documents = []
    
    for file_path in file_paths:
        path = Path(file_path)
        
        if not path.exists():
            logger.warning(f"File not found: {file_path}")
            continue
        
        # Get appropriate loader
        loader_class = LOADER_MAPPING.get(path.suffix.lower())
        if loader_class is None:
            logger.warning(f"Unsupported file type: {path.suffix}")
            continue
        
        try:
            # Load document
            loader = loader_class(str(path))
            documents = loader.load()
            
            # Enrich metadata
            for doc in documents:
                doc.metadata["source"] = str(path)
                doc.metadata["filename"] = path.name
                doc.metadata["file_type"] = path.suffix.lower()
                
                if add_metadata:
                    doc.metadata.update(add_metadata)
            
            all_documents.extend(documents)
            logger.info(f"Loaded {len(documents)} documents from {path.name}")
            
        except Exception as e:
            logger.error(f"Error loading {file_path}: {e}")
            continue
    
    return all_documents


# Usage
docs = load_documents(
    file_paths=["policy.pdf", "guidelines.md", "faq.docx"],
    add_metadata={"department": "lending", "version": "2024-01"}
)
```

---

## Best Practices

### 1. Preserve Structure Where Possible
- Keep page numbers in metadata
- Note section headings
- Identify document types

### 2. Clean Early
- Remove noise (headers, footers, page numbers)
- Normalize whitespace
- Handle encoding issues at load time

### 3. Log Everything
- Which files loaded successfully
- Which failed and why
- Document counts at each stage

### 4. Test with Real Data
- Your production documents will have issues your samples don't
- Test with messy, real-world files early

---

## What's Next?

You've loaded your documents. Now comes a critical step that's often underestimated: **chunking**.

How do you split documents into pieces? Too small and you lose context. Too large and retrieval becomes imprecise.

→ [2.3 Chunking Strategies](./03-chunking-strategies.md)

---

## 💡 Interview Tip

> **Q: "What challenges have you encountered loading documents for RAG?"**
>
> **A:** "The main challenges are: (1) Scanned PDFs that require OCR instead of text extraction; (2) Tables that become garbled when extracted as plain text; (3) Repeated headers and footers that add noise to every page; (4) Encoding issues with older documents. I address these by using the right loader for each document type — Unstructured for complex layouts with tables and OCR needs, and post-processing steps to remove noise like repeated headers. Metadata preservation is also important — tracking source file and page number for attribution."

