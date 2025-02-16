import json
import os
from copy import copy
from typing import Any, List

import httpx
from duckduckgo_search import DDGS
from googlesearch import search
import chromadb
from chromadb.api.types import IncludeEnum
from sentence_transformers import CrossEncoder, SentenceTransformer
from joblib import Parallel, delayed
from pdf2image import convert_from_path
import pytesseract
from typing import List
import PyPDF2
import os
from .document import Document, DocumentGenerator, DocumentProvider
from .embedding_provider import EmbeddingProvider, EmbeddingResult
from .reranking_provider import RerankingProvider, RerankingResult, RerankedDocument
from .vector_database_provider import VectorDatabaseProvider, VectorSearchResult
from .web_search import WebCrawler, WebSearchProvider
from .. import FunctionTool
from ..utilities.text_splitter import TextSplitter
from trafilatura import fetch_url, extract

class ChromaDbVectorDatabaseProvider(VectorDatabaseProvider):

    def __init__(self, embedding_provider: EmbeddingProvider, reranking_provider: RerankingProvider = None,
                 persistent_db_path="./retrieval_memory", default_collection_name="default_collection",
                 persistent: bool = False):
        super().__init__(embedding_provider, reranking_provider)
        if persistent:
            self.client = chromadb.PersistentClient(path=persistent_db_path)
        else:
            self.client = chromadb.EphemeralClient()

        self.collection = self.client.get_or_create_collection(
            name=default_collection_name
        )

    def add_documents(self, documents: list[Document]) -> None:
        texts = []
        metadata = []
        ids = []

        for document in documents:
            for chunk in document.document_chunks:
                meta = copy(document.metadata)
                if meta is None:
                    meta = {}
                meta["parent_doc_id"] = chunk.parent_doc_id
                meta["chunk_index"] = chunk.chunk_index
                ids.append(chunk.id)
                texts.append(chunk.content)
                metadata.append(meta)
        embeddings = self.embedding_provider.get_embedding(texts=texts)
        embeddings = embeddings.embeddings
        mem = texts
        self.collection.add(documents=mem, embeddings=embeddings, metadatas=metadata, ids=ids)

    def add_texts(self, texts: list[str], metadata: list[dict]) -> None:
        mem = texts
        ids = [str(self.generate_unique_id()) for _ in range(len(texts))]
        self.collection.add(documents=mem, metadatas=metadata, ids=ids)

    def query(self, query: str, query_filter: Any = None, k: int = 3, **kwargs) -> VectorSearchResult:
        query_embedding = self.embedding_provider.get_embedding([query])
        query_result = self.collection.query(
            query_embedding.embeddings[0],
            n_results=min(k *4, self.collection.count()),
            include=[IncludeEnum.metadatas, IncludeEnum.documents, IncludeEnum.distances],
            where=query_filter
        )
        documents = []
        for doc in query_result["documents"][0]:
            documents.append(doc)
        if self.reranking_provider is not None:
            results = self.reranking_provider.rerank_texts(query, documents, k=k, return_documents=True)
            # Putting everything together in a vector search result object.
            result = VectorSearchResult(query_result["ids"][0], [r.content for r in results.reranked_documents], [r.additional_data["score"] for r in results.reranked_documents])
            return result
        else:
            result = VectorSearchResult(query_result["ids"][0], documents,
                               [r for r in query_result["distances"][0]])
            return result

    def create_or_set_current_collection(self, collection_name: str) -> None:
        self.collection = self.client.get_or_create_collection(
            name=collection_name
        )

class SentenceTransformerEmbeddingProvider(EmbeddingProvider):

    def __init__(self, sentence_transformer_model_path: str = "all-MiniLM-L6-v2", trust_remote_code: bool = False, device: str = "cpu"):
        self.encoder = SentenceTransformer(
            sentence_transformer_model_path,
            trust_remote_code=trust_remote_code,
            device=device
        )
    def get_embedding(self, texts: list[str]) -> EmbeddingResult:
        return EmbeddingResult(self.encoder.encode(texts))


class MXBAIRerankingProvider(RerankingProvider):

    def __init__(self, rerank_model: str ="mixedbread-ai/mxbai-rerank-xsmall-v1", trust_remote_code: bool = False, device: str = "cpu"):
        super().__init__()
        self.cross_encoder = CrossEncoder(rerank_model, trust_remote_code=trust_remote_code, device=device)

    def rerank_texts(self, query: str, texts: list, k: int, **kwargs) -> RerankingResult:
        results = self.cross_encoder.rank(query, texts, return_documents=True, top_k=k)
        documents = []
        for doc in results:
            text = doc.pop("text", None)
            if text is not None:
                documents.append(RerankedDocument(text, doc))
        return RerankingResult(reranked_documents=documents)


class RAG:
    """
    Represents a chromadb vector database with a Colbert reranker.
    """

    def __init__(
        self,
        vector_database_provider: VectorDatabaseProvider
    ):
        self.vector_database_provider = vector_database_provider

    def add_document(self, document: str, metadata: dict = None):
        self.vector_database_provider.add_texts([document], [metadata])

    def add_documents(self, documents: list[str], metadata: list[dict] = None):
        self.vector_database_provider.add_texts(documents, metadata)

    def retrieve_documents(self, query: str, k):
        return self.vector_database_provider.query(query, k=k)


class PDFOCRProvider(DocumentProvider):

    def __init__(self, folder: str, text_splitter: TextSplitter):
        self.folder = folder
        self.document_generator = DocumentGenerator(text_splitter=text_splitter)


    def get_documents(self, **kwargs) -> list[Document]:
        documents = []
        for text in self.process_pdf(self.folder):
            documents.append(self.document_generator.generate_document(text))

        return documents

    @staticmethod
    def process_page(page):
        # Convert the page to grayscale
        page = page.convert('L')

        # Apply OCR using the preloaded pytesseract
        page_text = pytesseract.image_to_string(page)

        return page_text


    @staticmethod
    def find_files_by_extension(folder_path, extension):
        import os
        """
        Find all files with a specific extension in a given folder.

        Args:
            folder_path (str): Path to the folder to search in
            extension (str): File extension to search for (e.g., '.pdf')

        Returns:
            list: List of full file paths for files with the specified extension
        """
        # Make sure extension starts with a dot
        if not extension.startswith('.'):
            extension = '.' + extension

        matching_files = []

        # Walk through the directory
        try:
            for root, _, files in os.walk(folder_path):
                for file in files:
                    if file.lower().endswith(extension.lower()):
                        # Create full file path and add to list
                        full_path = os.path.join(root, file)
                        matching_files.append(full_path)

            return matching_files

        except Exception as e:
            print(f"Error accessing directory: {e}")
            return []

    @staticmethod
    def process_pdf(path):
        results = []
        for file in PDFOCRProvider.find_files_by_extension(path, "pdf"):
            pages = convert_from_path(str(file), dpi=300, fmt='PNG')
            page_texts = Parallel(n_jobs=-1)(delayed(PDFOCRProvider.process_page)(page) for page in pages)
            results.append('\n'.join(page_texts))
        return results


class PDFProvider(DocumentProvider):
    """
    A document provider that extracts text directly from PDF files without using OCR.
    """

    def __init__(self, folder: str, text_splitter: TextSplitter):
        """
        Initialize the PDF provider.

        Args:
            folder (str): Path to the folder containing PDF files
            text_splitter (TextSplitter): Text splitter instance for chunking documents
        """
        self.folder = folder
        self.document_generator = DocumentGenerator(text_splitter=text_splitter)

    def get_documents(self, **kwargs) -> List[Document]:
        """
        Get documents from PDF files in the specified folder.

        Returns:
            List[Document]: List of processed documents
        """
        documents = []
        for text in self._process_pdf_files(self.folder):
            documents.append(self.document_generator.generate_document(text))
        return documents

    @staticmethod
    def _find_pdf_files(folder_path: str) -> List[str]:
        """
        Find all PDF files in the given folder path.

        Args:
            folder_path (str): Path to search for PDF files

        Returns:
            List[str]: List of PDF file paths
        """
        pdf_files = []
        for root, _, files in os.walk(folder_path):
            for file in files:
                if file.lower().endswith('.pdf'):
                    pdf_files.append(os.path.join(root, file))
        return pdf_files

    def _process_pdf_files(self, folder_path: str) -> List[str]:
        """
        Process all PDF files in the given folder and extract their text content.

        Args:
            folder_path (str): Path to the folder containing PDF files

        Returns:
            List[str]: List of extracted text content from PDF files
        """
        results = []
        pdf_files = self._find_pdf_files(folder_path)

        for pdf_path in pdf_files:
            try:
                with open(pdf_path, 'rb') as file:
                    # Create PDF reader object
                    pdf_reader = PyPDF2.PdfReader(file)

                    # Extract text from all pages
                    text_content = []
                    for page in pdf_reader.pages:
                        text_content.append(page.extract_text())

                    # Join all pages with newlines
                    full_text = '\n'.join(text_content)
                    results.append(full_text)

            except Exception as e:
                print(f"Error processing PDF file {pdf_path}: {str(e)}")
                continue

        return results

class TrafilaturaWebCrawler(WebCrawler):
    def get_website_content_from_url(self, url: str) -> str:
        """
        Get website content from a URL using Selenium and BeautifulSoup for improved content extraction and filtering.

        Args:
            url (str): URL to get website content from.

        Returns:
            str: Extracted content including title, main text, and tables.
        """

        try:
            downloaded = fetch_url(url)

            result = extract(downloaded, include_formatting=True, include_links=True, output_format='json', url=url)

            if result:
                result = json.loads(result)
                return f'=========== Website Title: {result["title"]} ===========\n\n=========== Website URL: {url} ===========\n\n=========== Website Content ===========\n\n{result["raw_text"]}\n\n=========== Website Content End ===========\n\n'
            else:
                return ""
        except Exception as e:
            return f"An error occurred: {str(e)}"
    def get_tool(self):
        return FunctionTool(self.get_website_content_from_url)

class DDGWebSearchProvider(WebSearchProvider):

    def search_web(self, search_query: str, num_results: int):
        """
        Search the web, like a google search query. Returns a list of result URLs
        Args:
            search_query (str): Search query.
            num_results (int): Number of results to return.
        Returns:
            List[str]: List of URLs with search results.
        """
        results = DDGS().text(search_query, region='wt-wt', safesearch='off', max_results=num_results)
        return [res["href"] for res in results]

    def get_tool(self):
        return FunctionTool(self.search_web)

class HackernewsWebSearchProvider(WebSearchProvider):

    def search_web(self, search_query: str, num_results: int):
        """
        Search the web, like a google search query. Returns a list of result URLs
        Args:
            search_query (str): Search query.
            num_results (int): Number of results to return.
        Returns:
            List[str]: List of URLs with search results.
        """
        # Fetch top story IDs
        response = httpx.get("https://hacker-news.firebaseio.com/v0/topstories.json")
        response.raise_for_status()
        story_ids = response.json()

        # Fetch story details
        stories = []
        for story_id in story_ids[:num_results]:
            story_response = httpx.get(f"https://hacker-news.firebaseio.com/v0/item/{story_id}.json")
            story_response.raise_for_status()
            story = story_response.json()
            story["username"] = story["by"]
            stories.append(story["url"])

        return stories

    def get_tool(self):
        return FunctionTool(self.search_web)

class GoogleWebSearchProvider(WebSearchProvider):
    def search_web(self, search_query: str, num_results: int):
        """
        Search the web, like a google search query. Returns a list of result URLs
        Args:
            search_query (str): Search query.
            num_results (int): Number of results to return.
        Returns:
            List[str]: List of URLs with search results.
        """
        try:
            # Only return the top 5 results for simplicity
            return list(search(search_query, num_results=num_results))
        except Exception as e:
            return f"An error occurred during Google search: {str(e)}"
    def get_tool(self):
        return FunctionTool(self.search_web)