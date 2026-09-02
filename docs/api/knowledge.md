---
title: Knowledge API
---

# Knowledge API

Retrieval, document ingestion, text splitting, web search and crawling.

!!! note "Most of this needs an optional extra"

    Only the base classes, BM25 retrieval and the text splitters are pure
    Python. The rest pull in third-party packages:

    | Area | Extra |
    | --- | --- |
    | Chroma, sentence-transformers embeddings, cross-encoder reranking, and the `numpy` types the base classes use | `memory` |
    | OCR PDF ingestion | `ocr` |
    | Web search and crawling | `search` |

    Two imports are declared in no extra at all: `PyPDF2` (used by
    `PDFProvider`) and `ddgs` (used by `DDGWebSearchProvider`). Install them
    directly if you need those two.

## RAG

::: ToolAgents.knowledge.vector_database.rag.RAG

## Vector databases

A `VectorDatabaseProvider` stores chunks and returns the nearest matches for a
query. `BM25VectorDatabaseProvider` is a pure-Python keyword retriever with no
dependencies; `EnsembleVectorDatabaseProvider` fuses the scores of two
providers, which is the usual way to combine keyword and semantic retrieval.

::: ToolAgents.knowledge.vector_database.vector_database_provider.VectorDatabaseProvider

::: ToolAgents.knowledge.vector_database.vector_database_provider.VectorSearchResult

::: ToolAgents.knowledge.vector_database.vector_database_provider.VectorCollection

::: ToolAgents.knowledge.vector_database.implementations.bm25_database.BM25VectorDatabaseProvider

::: ToolAgents.knowledge.vector_database.implementations.ensemble_vector_database.EnsembleVectorDatabaseProvider

::: ToolAgents.knowledge.vector_database.implementations.chroma_db_vector_database.ChromaDbVectorDatabaseProvider

### Embeddings

::: ToolAgents.knowledge.vector_database.embedding_provider.EmbeddingProvider

::: ToolAgents.knowledge.vector_database.embedding_provider.EmbeddingTask

::: ToolAgents.knowledge.vector_database.embedding_provider.EmbeddingResult

::: ToolAgents.knowledge.vector_database.embedding_provider.EmbeddingPrefixConfig

::: ToolAgents.knowledge.vector_database.implementations.open_ai_embeddings.OpenAIEmbeddingProvider

::: ToolAgents.knowledge.vector_database.implementations.sentence_transformer_embeddings.SentenceTransformerEmbeddingProvider

### Reranking

::: ToolAgents.knowledge.vector_database.reranking_provider.RerankingProvider

::: ToolAgents.knowledge.vector_database.reranking_provider.RerankingResult

::: ToolAgents.knowledge.vector_database.reranking_provider.RerankedDocument

::: ToolAgents.knowledge.vector_database.implementations.cross_encoder_reranking.CrossEncoderRerankingProvider

## Documents

A `Document` is a list of `DocumentChunk`s. `DocumentGenerator` turns raw text
into one using a [text splitter](#text-splitting); a `DocumentProvider` loads
documents from a source such as a PDF.

::: ToolAgents.knowledge.document.document.Document

::: ToolAgents.knowledge.document.document.DocumentChunk

::: ToolAgents.knowledge.document.document.DocumentGenerator

::: ToolAgents.knowledge.document.document_provider.DocumentProvider

::: ToolAgents.knowledge.document.implementations.pypdf2_pdf.PDFProvider

::: ToolAgents.knowledge.document.implementations.pytesseract_pdf.PDFOCRProvider

## Text splitting

::: ToolAgents.knowledge.text_processing.text_splitter.TextSplitter

::: ToolAgents.knowledge.text_processing.text_splitter.RecursiveCharacterTextSplitter

::: ToolAgents.knowledge.text_processing.text_splitter.SimpleTextSplitter

::: ToolAgents.knowledge.text_processing.text_splitter.NonTextSplitter

## Text processing

`TextTransformer` runs a prompt template over a document through an agent.
`summarize_list_of_strings` summarizes several documents at once, and
`SummarizingFunctionToolPostProcessor` is a
[post-processor](tools.md#pre-and-post-processors) that shrinks long tool
results before they return to the model.

::: ToolAgents.knowledge.text_processing.text_transformer.TextTransformer

::: ToolAgents.knowledge.text_processing.summarizer.summarize_list_of_strings

::: ToolAgents.knowledge.text_processing.summarizer.SummarizingFunctionToolPostProcessor

## Web search

Each provider returns a list of URLs for a query. Note that the abstract method
is declared as `search_web(query, number_of_results)` while the implementations
use `search_web(search_query, num_results)`.

::: ToolAgents.knowledge.web_search.web_search.WebSearchProvider

::: ToolAgents.knowledge.web_search.implementations.duck_duck_go.DDGWebSearchProvider

::: ToolAgents.knowledge.web_search.implementations.googlesearch.GoogleWebSearchProvider

::: ToolAgents.knowledge.web_search.implementations.hacker_news.HackernewsWebSearchProvider

## Web crawling

::: ToolAgents.knowledge.web_crawler.web_crawler.WebCrawler

::: ToolAgents.knowledge.web_crawler.implementations.trafilatura.TrafilaturaWebCrawler

::: ToolAgents.knowledge.web_crawler.implementations.camoufox_crawler.CamoufoxWebCrawler

::: ToolAgents.knowledge.web_crawler.html2markdown.HTML2Markdown

::: ToolAgents.knowledge.web_crawler.html2markdown.convert_file
