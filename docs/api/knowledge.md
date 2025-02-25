---
title: Knowledge API
---

# Knowledge API

The Knowledge module provides capabilities for working with documents, vector databases, web search, and retrieval-augmented generation (RAG).

## RAG (Retrieval-Augmented Generation)

`RAG` provides interfaces for vector database retrieval and context augmentation.

```python
from ToolAgents.knowledge.vector_database.rag import RAG

rag = RAG(vector_database_provider=vector_db_provider)
```

### Constructor Parameters

- `vector_database_provider` (VectorDatabaseProvider): Provider for vector database operations
- `embedding_provider` (EmbeddingProvider, optional): Provider for generating embeddings
- `reranking_provider` (RerankingProvider, optional): Provider for reranking results

### Methods

#### `add_document(document, metadata=None)`

Adds a document to the vector database.

**Parameters:**
- `document` (str): Document text
- `metadata` (dict, optional): Additional metadata

#### `add_documents(documents, metadata=None)`

Adds multiple documents to the vector database.

**Parameters:**
- `documents` (List[str]): Document texts
- `metadata` (List[dict], optional): Additional metadata for each document

#### `retrieve_documents(query, k=5, filter=None)`

Retrieves relevant documents for a query.

**Parameters:**
- `query` (str): Query text
- `k` (int): Number of documents to retrieve
- `filter` (dict, optional): Filter criteria

**Returns:**
- `List[dict]`: Retrieved documents with metadata and relevance scores

#### `retrieve_and_rerank_documents(query, k=5, rerank_k=None, filter=None)`

Retrieves and reranks documents for a query.

**Parameters:**
- `query` (str): Query text
- `k` (int): Number of documents to retrieve
- `rerank_k` (int, optional): Number of documents after reranking
- `filter` (dict, optional): Filter criteria

**Returns:**
- `List[dict]`: Retrieved and reranked documents

#### `get_relevant_context(query, max_tokens=3000, k=5, rerank=False, rerank_k=None)`

Gets relevant context for a query, limiting by token count.

**Parameters:**
- `query` (str): Query text
- `max_tokens` (int): Maximum tokens to include
- `k` (int): Number of documents to retrieve
- `rerank` (bool): Whether to rerank results
- `rerank_k` (int, optional): Number of documents after reranking

**Returns:**
- `str`: Combined relevant context

## Vector Database

### VectorDatabaseProvider

Abstract interface for vector database operations.

```python
from ToolAgents.knowledge.vector_database.vector_database_provider import VectorDatabaseProvider
```

#### Methods

##### `add_texts(texts, metadata=None, ids=None)`

Adds texts to the database.

**Parameters:**
- `texts` (List[str]): Texts to add
- `metadata` (List[dict], optional): Metadata for each text
- `ids` (List[str], optional): IDs for each text

**Returns:**
- `List[str]`: IDs of added texts

##### `query(query_text, k=5, filter=None)`

Queries for similar texts.

**Parameters:**
- `query_text` (str): Query text
- `k` (int): Number of results
- `filter` (dict, optional): Filter criteria

**Returns:**
- `List[dict]`: Query results

### ChromaDB

Implementation of VectorDatabaseProvider using Chroma.

```python
from ToolAgents.knowledge.vector_database.implementations.chroma_db import ChromaDB

vector_db = ChromaDB(
    collection_name="your_collection",
    persist_directory="./chroma_db"
)
```

### EmbeddingProvider

Abstract interface for generating text embeddings.

```python
from ToolAgents.knowledge.vector_database.embedding_provider import EmbeddingProvider
```

#### Methods

##### `get_embeddings(texts)`

Generates embeddings for texts.

**Parameters:**
- `texts` (List[str]): Texts to embed

**Returns:**
- `List[List[float]]`: Embeddings

### SentenceTransformerEmbeddings

Implementation of EmbeddingProvider using sentence-transformers.

```python
from ToolAgents.knowledge.vector_database.implementations.sentence_transformer_embeddings import SentenceTransformerEmbeddings

embeddings = SentenceTransformerEmbeddings(
    model_name="all-MiniLM-L6-v2"
)
```

### RerankingProvider

Abstract interface for reranking search results.

```python
from ToolAgents.knowledge.vector_database.reranking_provider import RerankingProvider
```

#### Methods

##### `rerank(query, documents, scores=None, k=None)`

Reranks documents based on relevance to the query.

**Parameters:**
- `query` (str): Query text
- `documents` (List[str]): Documents to rerank
- `scores` (List[float], optional): Initial scores
- `k` (int, optional): Number of results to return

**Returns:**
- `List[Tuple[str, float]]`: Reranked documents with scores

### MBXAIReranking

Implementation of RerankingProvider using MBXAI's reranking.

```python
from ToolAgents.knowledge.vector_database.implementations.mbxai_reranking import MBXAIReranking

reranker = MBXAIReranking(
    api_key="your-mbxai-key",
    model_name="rerank-multilingual-v2.0"
)
```

## Document Processing

### Document

Class representing a document.

```python
from ToolAgents.knowledge.document.document import Document

document = Document(
    content="Document content",
    metadata={"source": "file.pdf"}
)
```

### DocumentProvider

Abstract interface for loading documents.

```python
from ToolAgents.knowledge.document.document_provider import DocumentProvider
```

#### Methods

##### `load_document(file_path)`

Loads a document from a file.

**Parameters:**
- `file_path` (str): Path to the document file

**Returns:**
- `Document`: Loaded document

### PDFDocumentProvider

Implementation of DocumentProvider for PDF files.

```python
from ToolAgents.knowledge.document.implementations.pypdf2_pdf import PyPDF2DocumentProvider

pdf_provider = PyPDF2DocumentProvider()
document = pdf_provider.load_document("document.pdf")
```

## Text Processing

### TextSplitter

Splits text into manageable chunks.

```python
from ToolAgents.knowledge.text_processing.text_splitter import TextSplitter

splitter = TextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)

chunks = splitter.split_text(long_text)
```

### Summarizer

Generates summaries using LLMs.

```python
from ToolAgents.knowledge.text_processing.summarizer import Summarizer

summarizer = Summarizer(
    chat_api=chat_api_provider
)

summary = summarizer.summarize(long_text)
```

### TextTransformer

Transforms text between different formats.

```python
from ToolAgents.knowledge.text_processing.text_transformer import TextTransformer

transformer = TextTransformer()
markdown = transformer.html_to_markdown(html_content)
```

## Web Search and Crawling

### WebSearch

Abstract interface for web search.

```python
from ToolAgents.knowledge.web_search.web_search import WebSearch
```

#### Methods

##### `search(query, num_results=5)`

Searches the web for a query.

**Parameters:**
- `query` (str): Search query
- `num_results` (int): Number of results to return

**Returns:**
- `List[dict]`: Search results with URLs and snippets

### GoogleSearch

Implementation of WebSearch using Google.

```python
from ToolAgents.knowledge.web_search.implementations.googlesearch import GoogleSearch

search = GoogleSearch()
results = search.search("ToolAgents framework")
```

### DuckDuckGoSearch

Implementation of WebSearch using DuckDuckGo.

```python
from ToolAgents.knowledge.web_search.implementations.duck_duck_go import DuckDuckGoSearch

search = DuckDuckGoSearch()
results = search.search("ToolAgents framework")
```

### WebCrawler

Abstract interface for web crawling.

```python
from ToolAgents.knowledge.web_crawler.web_crawler import WebCrawler
```

#### Methods

##### `extract_content(url)`

Extracts content from a webpage.

**Parameters:**
- `url` (str): URL to crawl

**Returns:**
- `str`: Extracted content

### TrafilaturaCrawler

Implementation of WebCrawler using Trafilatura.

```python
from ToolAgents.knowledge.web_crawler.implementations.trafilatura import TrafilaturaCrawler

crawler = TrafilaturaCrawler()
content = crawler.extract_content("https://example.com")
```

### CamoufoxCrawler

Implementation of WebCrawler using Camoufox.

```python
from ToolAgents.knowledge.web_crawler.implementations.camoufox_crawler import CamoufoxCrawler

crawler = CamoufoxCrawler()
content = crawler.extract_content("https://example.com")
```