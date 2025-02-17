from copy import copy
from typing import Any

import chromadb
from chromadb.api.types import IncludeEnum

from ToolAgents.knowledge import Document
from ToolAgents.knowledge.vector_database import VectorDatabaseProvider, EmbeddingProvider, RerankingProvider, \
    VectorSearchResult


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