from copy import copy
from typing import Any, Optional

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct, Filter

from ToolAgents.knowledge import Document
from ToolAgents.knowledge.vector_database import (
    VectorDatabaseProvider,
    EmbeddingProvider,
    RerankingProvider,
    VectorSearchResult,
    VectorCollectionSnapshot,
)


class QdrantVectorDatabaseProvider(VectorDatabaseProvider):

    def __init__(
            self,
            embedding_provider: EmbeddingProvider,
            reranking_provider: RerankingProvider = None,
            url: str = "localhost",
            port: int = 6333,
            api_key: Optional[str] = None,
            collection_name: str = "default_collection",
            vector_size: int = 1536,
    ):
        super().__init__(embedding_provider, reranking_provider)

        self.client = QdrantClient(
            url=url,
            port=port,
            api_key=api_key
        )

        self.collection_name = collection_name
        self.vector_size = vector_size
        self._ensure_collection_exists()

    def _ensure_collection_exists(self):
        collections = self.client.get_collections().collections
        collection_names = [col.name for col in collections]

        if self.collection_name not in collection_names:
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(size=self.vector_size, distance=Distance.COSINE)
            )

    def add_documents(self, documents: list[Document]) -> None:
        points = []
        texts = []

        for document in documents:
            for chunk in document.document_chunks:
                meta = copy(document.metadata) or {}
                meta.update({
                    "parent_doc_id": chunk.parent_doc_id,
                    "chunk_index": chunk.chunk_index,
                    "content": chunk.content
                })
                texts.append(chunk.content)
                points.append({
                    "id": chunk.id,
                    "payload": meta
                })

        embeddings = self.embedding_provider.get_embedding(texts=texts)

        for i, point in enumerate(points):
            point["vector"] = embeddings.embeddings[i]

        self.client.upsert(
            collection_name=self.collection_name,
            points=[PointStruct(**point) for point in points]
        )

    def add_texts(self, texts: list[str], metadata: list[dict]) -> None:
        points = []
        embeddings = self.embedding_provider.get_embedding(texts=texts)

        for i, text in enumerate(texts):
            meta = copy(metadata[i]) if i < len(metadata) else {}
            meta["content"] = text

            points.append(PointStruct(
                id=str(self.generate_unique_id()),
                vector=embeddings.embeddings[i],
                payload=meta
            ))

        self.client.upsert(
            collection_name=self.collection_name,
            points=points
        )

    def query(
            self, query: str, query_filter: Any = None, k: int = 3, **kwargs
    ) -> VectorSearchResult:
        query_embedding = self.embedding_provider.get_embedding([query])

        search_result = self.client.search(
            collection_name=self.collection_name,
            query_vector=query_embedding.embeddings[0],
            limit=min(k * 4, 1000),
            query_filter=query_filter
        )

        documents = []
        ids = []
        scores = []
        metadata = []

        for point in search_result:
            ids.append(str(point.id))
            documents.append(point.payload["content"])
            scores.append(point.score)
            metadata.append(point.payload)

        if self.reranking_provider is not None:
            results = self.reranking_provider.rerank_texts(
                query, documents, k=k, return_documents=True
            )

            reranked_ids = []
            doc_to_id = {doc: id for doc, id in zip(documents, ids)}

            for r in results.reranked_documents:
                reranked_ids.append(doc_to_id[r.content])

            return VectorSearchResult(
                reranked_ids,
                [r.content for r in results.reranked_documents],
                [r.additional_data["score"] for r in results.reranked_documents],
            )
        else:
            return VectorSearchResult(ids[:k], documents[:k], scores[:k], metadata[:k])

    def create_or_set_current_collection(self, collection_name: str) -> None:
        self.collection_name = collection_name
        self._ensure_collection_exists()

    def remove_by_ids(self, ids: list[str]) -> None:
        self.client.delete(
            collection_name=self.collection_name,
            points_selector=ids
        )

    def get_all_entries(self) -> VectorCollectionSnapshot:
        scroll_result = self.client.scroll(
            collection_name=self.collection_name,
            limit=10000
        )

        points = scroll_result[0]
        ids = []
        chunks = []
        metadata = []

        for point in points:
            ids.append(str(point.id))
            chunks.append(point.payload["content"])
            metadata.append(point.payload)

        return VectorCollectionSnapshot(
            ids=ids,
            chunks=chunks,
            metadata=metadata
        )

    def delete_collection(self, collection_name: str) -> None:
        self.client.delete_collection(collection_name)