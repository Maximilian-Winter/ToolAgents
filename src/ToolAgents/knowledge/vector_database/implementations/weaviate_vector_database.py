from copy import copy
from typing import Any, Optional
import json

import weaviate
from weaviate.classes.config import Configure

from ToolAgents.knowledge import Document
from ToolAgents.knowledge.vector_database import (
    VectorDatabaseProvider,
    EmbeddingProvider,
    RerankingProvider,
    VectorSearchResult,
    VectorCollectionSnapshot,
)


class WeaviateVectorDatabaseProvider(VectorDatabaseProvider):

    def __init__(
            self,
            embedding_provider: EmbeddingProvider,
            reranking_provider: RerankingProvider = None,
            url: str = "http://localhost:8080",
            api_key: Optional[str] = None,
            collection_name: str = "DefaultCollection",
    ):
        super().__init__(embedding_provider, reranking_provider)

        auth_config = None
        if api_key:
            auth_config = weaviate.AuthApiKey(api_key=api_key)

        self.client = weaviate.connect_to_custom(
            http_host=url.replace("http://", "").replace("https://", ""),
            http_port=8080,
            http_secure=False,
            auth_credentials=auth_config
        )

        self.collection_name = collection_name
        self._ensure_collection_exists()

    def _ensure_collection_exists(self):
        if not self.client.collections.exists(self.collection_name):
            self.client.collections.create(
                name=self.collection_name,
                vectorizer_config=Configure.Vectorizer.none()
            )
        self.collection = self.client.collections.get(self.collection_name)

    def add_documents(self, documents: list[Document]) -> None:
        objects = []
        texts = []

        for document in documents:
            for chunk in document.document_chunks:
                meta = copy(document.metadata) or {}
                meta.update({
                    "parent_doc_id": chunk.parent_doc_id,
                    "chunk_index": chunk.chunk_index,
                })

                texts.append(chunk.content)
                objects.append({
                    "uuid": chunk.id,
                    "properties": {
                        "content": chunk.content,
                        "metadata": json.dumps(meta)
                    }
                })

        embeddings = self.embedding_provider.get_embedding(texts=texts)

        for i, obj in enumerate(objects):
            obj["vector"] = embeddings.embeddings[i]

        self.collection.data.insert_many(objects)

    def add_texts(self, texts: list[str], metadata: list[dict]) -> None:
        objects = []
        embeddings = self.embedding_provider.get_embedding(texts=texts)

        for i, text in enumerate(texts):
            meta = copy(metadata[i]) if i < len(metadata) else {}
            objects.append({
                "uuid": str(self.generate_unique_id()),
                "properties": {
                    "content": text,
                    "metadata": json.dumps(meta)
                },
                "vector": embeddings.embeddings[i]
            })

        self.collection.data.insert_many(objects)

    def query(
            self, query: str, query_filter: Any = None, k: int = 3, **kwargs
    ) -> VectorSearchResult:
        query_embedding = self.embedding_provider.get_embedding([query])

        response = self.collection.query.near_vector(
            near_vector=query_embedding.embeddings[0],
            limit=min(k * 4, 1000),
            where=query_filter
        )

        documents = []
        ids = []
        scores = []
        metadata = []

        for obj in response.objects:
            ids.append(str(obj.uuid))
            documents.append(obj.properties["content"])
            scores.append(obj.metadata.distance)

            try:
                meta = json.loads(obj.properties["metadata"])
            except:
                meta = {}
            metadata.append(meta)

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
        for id in ids:
            self.collection.data.delete_by_id(id)

    def get_all_entries(self) -> VectorCollectionSnapshot:
        response = self.collection.iterator()

        ids = []
        chunks = []
        metadata = []

        for obj in response:
            ids.append(str(obj.uuid))
            chunks.append(obj.properties["content"])

            try:
                meta = json.loads(obj.properties["metadata"])
            except:
                meta = {}
            metadata.append(meta)

        return VectorCollectionSnapshot(
            ids=ids,
            chunks=chunks,
            metadata=metadata
        )

    def delete_collection(self, collection_name: str) -> None:
        self.client.collections.delete(collection_name)