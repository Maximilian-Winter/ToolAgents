from copy import copy
from typing import Any

import chromadb
from chromadb.api.types import IncludeEnum
from sentence_transformers import CrossEncoder, SentenceTransformer
from joblib import Parallel, delayed
from pdf2image import convert_from_path
import pytesseract

from .document import Document, DocumentGenerator, DocumentProvider
from .embedding_provider import EmbeddingProvider, EmbeddingResult
from .reranking_provider import RerankingProvider, RerankingResult, RerankedDocument
from .vector_database_provider import VectorDatabaseProvider, VectorSearchResult
from ..utilities.text_splitter import TextSplitter


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
            n_results=k,
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

    def __init__(self, rerank_model: str ="mixedbread-ai/mxbai-rerank-xsmall-v1"):
        super().__init__()
        self.cross_encoder = CrossEncoder(rerank_model)

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

    @staticmethod
    def set_pytesseract_cmd(path: str = r'C:\Program Files\Tesseract-OCR\tesseract.exe'):
        pytesseract.pytesseract.tesseract_cmd = path

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
    def process_pdf(path):
        pages = convert_from_path(path, dpi=300, fmt='PNG')
        page_texts = Parallel(n_jobs=-1)(delayed(PDFOCRProvider.process_page)(page) for page in pages)

        return page_texts


class ArxivTool:

    def __init__(self, folder: str, text_splitter: TextSplitter):
        self.folder = folder
        self.document_generator = DocumentGenerator(text_splitter=text_splitter)
        self.pdf_ocr_provider = PDFOCRProvider(folder, text_splitter)

    def get_documents(self, query) -> list[Document]:
        pass