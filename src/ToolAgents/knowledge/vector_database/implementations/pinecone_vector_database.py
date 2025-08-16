from copy import copy
from typing import Any, Optional
import json

import pinecone
from pinecone import Pinecone

from ToolAgents.knowledge import Document
from ToolAgents.knowledge.vector_database import (
    VectorDatabaseProvider,
    EmbeddingProvider,
    RerankingProvider,
    VectorSearchResult,
    VectorCollectionSnapshot,
)


class PineconeVectorDatabaseProvider(VectorDatabaseProvider):

    def __init__(
            self,
            embedding_provider: EmbeddingProvider,
            reranking_provider: RerankingProvider = None,
            api_key: str = None,
            index_name: str = "default-index",
            environment: str = None,
            dimension: int = 1536,
    ):
        super().__init__(embedding_provider, reranking_provider)
        self.pc = Pinecone(api_key=api_key)
        self.index_name = index_name
        self.dimension = dimension

        if index_name not in self.pc.list_indexes().names():
            self.pc.create_index(
                name=index_name,
                dimension=dimension,
                metric="cosine"
            )

        self.index = self.pc.Index(index_name)

    def add_documents(self, documents: list[Document]) -> None:
        vectors = []
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
                vectors.append({
                    "id": chunk.id,
                    "metadata": meta
                })

        embeddings = self.embedding_provider.get_embedding(texts=texts)

        for i, vector in enumerate(vectors):
            vector["values"] = embeddings.embeddings[i]

        self.index.upsert(vectors=vectors)

    def add_texts(self, texts: list[str], metadata: list[dict]) -> None:
        vectors = []
        embeddings = self.embedding_provider.get_embedding(texts=texts)

        for i, text in enumerate(texts):
            meta = copy(metadata[i]) if i < len(metadata) else {}
            meta["content"] = text
            vectors.append({
                "id": str(self.generate_unique_id()),
                "values": embeddings.embeddings[i],
                "metadata": meta
            })

        self.index.upsert(vectors=vectors)

    def query(
            self, query: str, query_filter: Any = None, k: int = 3, **kwargs
    ) -> VectorSearchResult:
        query_embedding = self.embedding_provider.get_embedding([query])

        query_result = self.index.query(
            vector=query_embedding.embeddings[0],
            top_k=min(k * 4, 10000),
            include_metadata=True,
            filter=query_filter
        )

        documents = []
        ids = []
        scores = []
        metadata = []

        for match in query_result["matches"]:
            ids.append(match["id"])
            documents.append(match["metadata"]["content"])
            scores.append(match["score"])
            metadata.append(match["metadata"])

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
        if collection_name not in self.pc.list_indexes().names():
            self.pc.create_index(
                name=collection_name,
                dimension=self.dimension,
                metric="cosine"
            )
        self.index = self.pc.Index(collection_name)
        self.index_name = collection_name

    def remove_by_ids(self, ids: list[str]) -> None:
        self.index.delete(ids=ids)

    def get_all_entries(self) -> VectorCollectionSnapshot:
        stats = self.index.describe_index_stats()
        total_count = stats["total_vector_count"]

        all_ids = []
        all_chunks = []
        all_metadata = []

        for ids_batch in self.index.list(limit=total_count):
            if not ids_batch:
                break

            fetch_result = self.index.fetch(ids=ids_batch)
            for id, vector_data in fetch_result["vectors"].items():
                all_ids.append(id)
                all_chunks.append(vector_data["metadata"]["content"])
                all_metadata.append(vector_data["metadata"])

        return VectorCollectionSnapshot(
            ids=all_ids,
            chunks=all_chunks,
            metadata=all_metadata
        )

    def delete_collection(self, collection_name: str) -> None:
        self.pc.delete_index(collection_name)