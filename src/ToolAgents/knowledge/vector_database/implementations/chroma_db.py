from copy import copy
from typing import Any, List, Dict, Optional
from datetime import datetime

import chromadb

from ToolAgents.knowledge import Document
from ToolAgents.knowledge.vector_database import (
    VectorDatabaseProvider,
    EmbeddingProvider,
    RerankingProvider,
    VectorSearchResult,
    DocumentEntry,
    CollectionInfo,
)


class ChromaDbVectorDatabaseProvider(VectorDatabaseProvider):

    def __init__(
            self,
            embedding_provider: EmbeddingProvider,
            reranking_provider: RerankingProvider = None,
            persistent_db_path="./retrieval_memory",
            default_collection_name="default_collection",
            persistent: bool = False,
    ):
        super().__init__(embedding_provider, reranking_provider)
        if persistent:
            self.client = chromadb.PersistentClient(path=persistent_db_path)
        else:
            self.client = chromadb.EphemeralClient()

        self.collection = self.client.get_or_create_collection(
            name=default_collection_name
        )
        self.current_collection_name = default_collection_name

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
        self.collection.add(
            documents=mem, embeddings=embeddings, metadatas=metadata, ids=ids
        )

    def add_texts(self, texts: list[str], metadata: list[dict]) -> None:
        mem = texts
        ids = [str(self.generate_unique_id()) for _ in range(len(texts))]
        self.collection.add(documents=mem, metadatas=metadata, ids=ids)

    def query(
            self, query: str, query_filter: Any = None, k: int = 3, **kwargs
    ) -> VectorSearchResult:
        query_embedding = self.embedding_provider.get_embedding([query])
        query_result = self.collection.query(
            query_embedding.embeddings[0],
            n_results=min(k * 4, self.collection.count()),
            include=[
                "documents",
                "metadatas",
                "distances"
            ],
            where=query_filter,
        )

        document_text_to_ids = {}
        documents = []
        for id, doc in zip(query_result["ids"][0], query_result["documents"][0]):
            documents.append(doc)
            document_text_to_ids[doc] = id

        if self.reranking_provider is not None:
            results = self.reranking_provider.rerank_texts(
                query, documents, k=k, return_documents=True
            )
            doc_ids = []
            for r in results.reranked_documents:
                doc_ids.append(document_text_to_ids[r.content])

            result = VectorSearchResult(
                doc_ids,
                [r.content for r in results.reranked_documents],
                [r.additional_data["score"] for r in results.reranked_documents],
            )
            return result
        else:
            result = VectorSearchResult(
                query_result["ids"][0],
                documents,
                [r for r in query_result["distances"][0]],
            )
            return result

    def create_or_set_current_collection(self, collection_name: str) -> None:
        self.collection = self.client.get_or_create_collection(name=collection_name)
        self.current_collection_name = collection_name

    def remove_documents(self, document_ids: List[str]) -> int:
        """Remove documents by their IDs."""
        try:
            # ChromaDB doesn't return count, so we check before deletion
            count_before = len(document_ids)
            self.collection.delete(ids=document_ids)
            return count_before
        except Exception:
            return 0

    def remove_documents_by_filter(self, filter_dict: Dict[str, Any]) -> int:
        """Remove documents matching the specified filter criteria."""
        try:
            # Get matching documents first to count them
            result = self.collection.get(where=filter_dict)
            count = len(result["ids"])

            if count > 0:
                self.collection.delete(where=filter_dict)

            return count
        except Exception:
            return 0

    def get_all_documents(
            self,
            limit: Optional[int] = None,
            offset: int = 0,
            include_embeddings: bool = False
    ) -> List[DocumentEntry]:
        """Retrieve all documents from the current collection."""
        include_fields = ["documents", "metadatas"]
        if include_embeddings:
            include_fields.append("embeddings")

        # ChromaDB doesn't have native offset, so we get all and slice
        total_count = self.collection.count()

        if total_count == 0:
            return []

        # Determine how many to fetch
        fetch_limit = total_count if limit is None else min(offset + limit, total_count)

        result = self.collection.get(
            limit=fetch_limit,
            include=include_fields
        )

        # Apply offset
        documents = []
        start_idx = min(offset, len(result["ids"]))
        end_idx = len(result["ids"]) if limit is None else min(start_idx + limit, len(result["ids"]))

        for i in range(start_idx, end_idx):
            doc_entry = DocumentEntry(
                id=result["ids"][i],
                content=result["documents"][i],
                metadata=result["metadatas"][i] if result["metadatas"] else None,
                embedding=result.get("embeddings", [None])[i] if include_embeddings else None
            )
            documents.append(doc_entry)

        return documents

    def get_documents_by_ids(
            self,
            document_ids: List[str],
            include_embeddings: bool = False
    ) -> List[DocumentEntry]:
        """Retrieve specific documents by their IDs."""
        include_fields = ["documents", "metadatas"]
        if include_embeddings:
            include_fields.append("embeddings")

        result = self.collection.get(
            ids=document_ids,
            include=include_fields
        )

        documents = []
        for i in range(len(result["ids"])):
            doc_entry = DocumentEntry(
                id=result["ids"][i],
                content=result["documents"][i],
                metadata=result["metadatas"][i] if result["metadatas"] else None,
                embedding=result.get("embeddings", [None])[i] if include_embeddings else None
            )
            documents.append(doc_entry)

        return documents

    def get_documents_by_filter(
            self,
            filter_dict: Dict[str, Any],
            limit: Optional[int] = None,
            offset: int = 0,
            include_embeddings: bool = False
    ) -> List[DocumentEntry]:
        """Retrieve documents matching the specified filter criteria."""
        include_fields = ["documents", "metadatas"]
        if include_embeddings:
            include_fields.append("embeddings")

        # Get all matching documents
        result = self.collection.get(
            where=filter_dict,
            include=include_fields
        )

        # Apply offset and limit
        documents = []
        start_idx = min(offset, len(result["ids"]))
        end_idx = len(result["ids"]) if limit is None else min(start_idx + limit, len(result["ids"]))

        for i in range(start_idx, end_idx):
            doc_entry = DocumentEntry(
                id=result["ids"][i],
                content=result["documents"][i],
                metadata=result["metadatas"][i] if result["metadatas"] else None,
                embedding=result.get("embeddings", [None])[i] if include_embeddings else None
            )
            documents.append(doc_entry)

        return documents

    def update_document_metadata(
            self,
            document_id: str,
            metadata: Dict[str, Any],
            merge: bool = True
    ) -> bool:
        """Update metadata for a specific document."""
        try:
            if merge:
                # Get existing metadata
                existing = self.collection.get(ids=[document_id], include=["metadatas"])
                if existing["ids"]:
                    existing_metadata = existing["metadatas"][0] or {}
                    # Merge with new metadata
                    existing_metadata.update(metadata)
                    metadata = existing_metadata

            # Update the document
            self.collection.update(
                ids=[document_id],
                metadatas=[metadata]
            )
            return True
        except Exception:
            return False

    def list_collections(self) -> List[CollectionInfo]:
        """List all available collections in the database."""
        collections = self.client.list_collections()
        collection_infos = []

        for col in collections:
            try:
                # Get collection to access its data
                temp_collection = self.client.get_collection(col.name)
                count = temp_collection.count()

                collection_infos.append(
                    CollectionInfo(
                        name=col.name,
                        document_count=count,
                        metadata=col.metadata if hasattr(col, 'metadata') else None,
                        created_at=None,  # ChromaDB doesn't track this
                        updated_at=None  # ChromaDB doesn't track this
                    )
                )
            except Exception:
                # If we can't access the collection, add minimal info
                collection_infos.append(
                    CollectionInfo(
                        name=col.name,
                        document_count=0,
                        metadata=None
                    )
                )

        return collection_infos

    def delete_collection(self, collection_name: str) -> bool:
        """Delete a collection from the database."""
        try:
            self.client.delete_collection(collection_name)

            # If we deleted the current collection, switch to default
            if collection_name == self.current_collection_name:
                self.create_or_set_current_collection("default_collection")

            return True
        except Exception:
            return False

    def get_collection_info(self, collection_name: Optional[str] = None) -> CollectionInfo:
        """Get information about a specific collection or the current collection."""
        name = collection_name or self.current_collection_name

        try:
            col = self.client.get_collection(name) if collection_name else self.collection

            return CollectionInfo(
                name=name,
                document_count=col.count(),
                metadata=col.metadata if hasattr(col, 'metadata') else None,
                created_at=None,  # ChromaDB doesn't track this
                updated_at=None  # ChromaDB doesn't track this
            )
        except Exception:
            return CollectionInfo(name=name, document_count=0)

    def clear_collection(self, collection_name: Optional[str] = None) -> int:
        """Remove all documents from a collection without deleting the collection itself."""
        name = collection_name or self.current_collection_name

        try:
            col = self.client.get_collection(name) if collection_name else self.collection

            # Get all document IDs
            result = col.get()
            doc_count = len(result["ids"])

            if doc_count > 0:
                # Delete all documents
                col.delete(ids=result["ids"])

            return doc_count
        except Exception:
            return 0

    def get_document_count(self, filter_dict: Optional[Dict[str, Any]] = None) -> int:
        """Get the count of documents in the current collection."""
        if filter_dict:
            result = self.collection.get(where=filter_dict, include=[])
            return len(result["ids"])
        else:
            return self.collection.count()

    def exists(self, document_id: str) -> bool:
        """Check if a document exists in the current collection."""
        try:
            result = self.collection.get(ids=[document_id], include=[])
            return len(result["ids"]) > 0
        except Exception:
            return False

    def batch_update_metadata(
            self,
            updates: Dict[str, Dict[str, Any]],
            merge: bool = True
    ) -> Dict[str, bool]:
        """Update metadata for multiple documents in a single operation."""
        results = {}

        ids_list = list(updates.keys())
        metadatas_list = []

        if merge:
            # Get existing metadata for all documents
            existing = self.collection.get(ids=ids_list, include=["metadatas"])
            existing_map = {
                id: meta for id, meta in zip(existing["ids"], existing["metadatas"])
            }

            for doc_id in ids_list:
                existing_metadata = existing_map.get(doc_id, {}) or {}
                existing_metadata.update(updates[doc_id])
                metadatas_list.append(existing_metadata)
        else:
            metadatas_list = [updates[doc_id] for doc_id in ids_list]

        try:
            self.collection.update(
                ids=ids_list,
                metadatas=metadatas_list
            )
            # Assume all successful if no exception
            for doc_id in ids_list:
                results[doc_id] = True
        except Exception as e:
            # On failure, mark all as failed
            for doc_id in ids_list:
                results[doc_id] = False

        return results

    def export_collection(
            self,
            collection_name: Optional[str] = None,
            include_embeddings: bool = False
    ) -> Dict[str, Any]:
        """Export all data from a collection for backup or migration."""
        name = collection_name or self.current_collection_name

        try:
            col = self.client.get_collection(name) if collection_name else self.collection

            include_fields = ["documents", "metadatas"]
            if include_embeddings:
                include_fields.append("embeddings")

            result = col.get(include=include_fields)

            export_data = {
                "collection_name": name,
                "export_timestamp": datetime.now().isoformat(),
                "document_count": len(result["ids"]),
                "ids": result["ids"],
                "documents": result["documents"],
                "metadatas": result["metadatas"],
            }

            if include_embeddings and "embeddings" in result:
                export_data["embeddings"] = result["embeddings"]

            return export_data
        except Exception as e:
            return {
                "collection_name": name,
                "error": str(e),
                "document_count": 0
            }

    def import_collection(
            self,
            data: Dict[str, Any],
            collection_name: str,
            overwrite: bool = False
    ) -> bool:
        """Import data into a collection from an export."""
        try:
            # Create or get the collection
            if overwrite:
                try:
                    self.client.delete_collection(collection_name)
                except Exception:
                    pass  # Collection might not exist

            col = self.client.get_or_create_collection(collection_name)

            # Prepare the data for import
            ids = data.get("ids", [])
            documents = data.get("documents", [])
            metadatas = data.get("metadatas", [])

            if not ids or not documents:
                return False

            # Add documents in batches (ChromaDB has limits)
            batch_size = 1000
            for i in range(0, len(ids), batch_size):
                batch_ids = ids[i:i + batch_size]
                batch_docs = documents[i:i + batch_size]
                batch_meta = metadatas[i:i + batch_size] if metadatas else None

                if "embeddings" in data and data["embeddings"]:
                    batch_embeddings = data["embeddings"][i:i + batch_size]
                    col.add(
                        ids=batch_ids,
                        documents=batch_docs,
                        metadatas=batch_meta,
                        embeddings=batch_embeddings
                    )
                else:
                    # Generate embeddings
                    embeddings = self.embedding_provider.get_embedding(batch_docs)
                    col.add(
                        ids=batch_ids,
                        documents=batch_docs,
                        metadatas=batch_meta,
                        embeddings=embeddings.embeddings
                    )

            return True
        except Exception as e:
            print(f"Import failed: {e}")
            return False