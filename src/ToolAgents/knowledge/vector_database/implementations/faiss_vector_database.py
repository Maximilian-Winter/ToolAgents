from copy import copy
from typing import Any, Optional
import pickle
import numpy as np
import faiss

from ToolAgents.knowledge import Document
from ToolAgents.knowledge.vector_database import (
    VectorDatabaseProvider,
    EmbeddingProvider,
    RerankingProvider,
    VectorSearchResult,
    VectorCollectionSnapshot,
)


class FaissVectorDatabaseProvider(VectorDatabaseProvider):

    def __init__(
            self,
            embedding_provider: EmbeddingProvider,
            reranking_provider: RerankingProvider = None,
            dimension: int = 1536,
            index_type: str = "flat",
            persist_path: Optional[str] = None,
    ):
        super().__init__(embedding_provider, reranking_provider)

        self.dimension = dimension
        self.persist_path = persist_path

        if index_type == "flat":
            self.index = faiss.IndexFlatIP(dimension)
        elif index_type == "ivf":
            quantizer = faiss.IndexFlatIP(dimension)
            self.index = faiss.IndexIVFFlat(quantizer, dimension, 100)
        else:
            raise ValueError(f"Unsupported index type: {index_type}")

        self.id_to_index = {}
        self.index_to_id = {}
        self.metadata_store = {}
        self.content_store = {}
        self.next_index = 0

        if persist_path:
            self._load_index()

    def _load_index(self):
        try:
            self.index = faiss.read_index(f"{self.persist_path}.index")
            with open(f"{self.persist_path}.metadata", "rb") as f:
                data = pickle.load(f)
                self.id_to_index = data["id_to_index"]
                self.index_to_id = data["index_to_id"]
                self.metadata_store = data["metadata_store"]
                self.content_store = data["content_store"]
                self.next_index = data["next_index"]
        except FileNotFoundError:
            pass

    def _save_index(self):
        if self.persist_path:
            faiss.write_index(self.index, f"{self.persist_path}.index")
            with open(f"{self.persist_path}.metadata", "wb") as f:
                pickle.dump({
                    "id_to_index": self.id_to_index,
                    "index_to_id": self.index_to_id,
                    "metadata_store": self.metadata_store,
                    "content_store": self.content_store,
                    "next_index": self.next_index
                }, f)

    def add_documents(self, documents: list[Document]) -> None:
        vectors = []
        texts = []
        ids = []
        metadata_list = []

        for document in documents:
            for chunk in document.document_chunks:
                meta = copy(document.metadata) or {}
                meta.update({
                    "parent_doc_id": chunk.parent_doc_id,
                    "chunk_index": chunk.chunk_index,
                })

                texts.append(chunk.content)
                ids.append(chunk.id)
                metadata_list.append(meta)

        embeddings = self.embedding_provider.get_embedding(texts=texts)
        vectors = np.array(embeddings.embeddings).astype('float32')

        faiss.normalize_L2(vectors)

        start_index = self.next_index
        self.index.add(vectors)

        for i, (id, text, meta) in enumerate(zip(ids, texts, metadata_list)):
            index = start_index + i
            self.id_to_index[id] = index
            self.index_to_id[index] = id
            self.content_store[id] = text
            self.metadata_store[id] = meta

        self.next_index += len(vectors)
        self._save_index()

    def add_texts(self, texts: list[str], metadata: list[dict]) -> None:
        vectors = []
        ids = []
        embeddings = self.embedding_provider.get_embedding(texts=texts)
        vectors = np.array(embeddings.embeddings).astype('float32')

        faiss.normalize_L2(vectors)

        start_index = self.next_index
        self.index.add(vectors)

        for i, (text, meta) in enumerate(zip(texts, metadata)):
            id = str(self.generate_unique_id())
            index = start_index + i
            self.id_to_index[id] = index
            self.index_to_id[index] = id
            self.content_store[id] = text
            self.metadata_store[id] = meta

        self.next_index += len(vectors)
        self._save_index()

    def query(
            self, query: str, query_filter: Any = None, k: int = 3, **kwargs
    ) -> VectorSearchResult:
        query_embedding = self.embedding_provider.get_embedding([query])
        query_vector = np.array(query_embedding.embeddings).astype('float32')
        faiss.normalize_L2(query_vector)

        search_k = min(k * 4, self.index.ntotal)
        if search_k == 0:
            return VectorSearchResult([], [], [])

        scores, indices = self.index.search(query_vector, search_k)

        documents = []
        ids = []
        valid_scores = []
        metadata = []

        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:
                continue
            id = self.index_to_id.get(idx)
            if id:
                ids.append(id)
                documents.append(self.content_store[id])
                valid_scores.append(float(score))
                metadata.append(self.metadata_store[id])

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
            return VectorSearchResult(ids[:k], documents[:k], valid_scores[:k], metadata[:k])

    def create_or_set_current_collection(self, collection_name: str) -> None:
        if self.persist_path:
            self.persist_path = f"{collection_name}"
            self._load_index()

    def remove_by_ids(self, ids: list[str]) -> None:
        for id in ids:
            if id in self.id_to_index:
                index = self.id_to_index[id]
                del self.id_to_index[id]
                del self.index_to_id[index]
                del self.content_store[id]
                del self.metadata_store[id]

        self._save_index()

    def get_all_entries(self) -> VectorCollectionSnapshot:
        ids = list(self.content_store.keys())
        chunks = [self.content_store[id] for id in ids]
        metadata = [self.metadata_store[id] for id in ids]

        return VectorCollectionSnapshot(
            ids=ids,
            chunks=chunks,
            metadata=metadata
        )

    def delete_collection(self, collection_name: str) -> None:
        import os
        if self.persist_path:
            try:
                os.remove(f"{collection_name}.index")
                os.remove(f"{collection_name}.metadata")
            except FileNotFoundError:
                pass