from typing import List, Dict, Any, Optional

from ToolAgents.knowledge.vector_database.vector_database_provider import (
    VectorDatabaseProvider,
    DocumentEntry,
    CollectionInfo,
)


class RAG:
    """
    RAG (Retrieval Augmented Generation) system with
    database management capabilities.
    """

    def __init__(self, vector_database_provider: VectorDatabaseProvider):
        self.vector_database_provider = vector_database_provider

    # Original methods

    def add_document(self, document: str, metadata: dict = None):
        """Add a single document to the database."""
        self.vector_database_provider.add_texts([document], [metadata])

    def add_documents(self, documents: list[str], metadata: list[dict] = None):
        """Add multiple documents to the database."""
        self.vector_database_provider.add_texts(documents, metadata)

    def retrieve_documents(self, query: str, k: int = 3):
        """Retrieve the k most relevant documents for a query."""
        return self.vector_database_provider.query(query, k=k)

    # Enhanced document management methods

    def remove_document(self, document_id: str) -> bool:
        """
        Remove a single document by its ID.

        Args:
            document_id: ID of the document to remove

        Returns:
            True if the document was removed, False otherwise
        """
        count = self.vector_database_provider.remove_documents([document_id])
        return count > 0

    def remove_documents(self, document_ids: List[str]) -> int:
        """
        Remove multiple documents by their IDs.

        Args:
            document_ids: List of document IDs to remove

        Returns:
            Number of documents removed
        """
        return self.vector_database_provider.remove_documents(document_ids)

    def remove_documents_by_metadata(self, metadata_filter: Dict[str, Any]) -> int:
        """
        Remove all documents matching specific metadata criteria.

        Args:
            metadata_filter: Dictionary of metadata fields and values to match

        Returns:
            Number of documents removed
        """
        return self.vector_database_provider.remove_documents_by_filter(metadata_filter)

    def get_document(self, document_id: str, include_embedding: bool = False) -> Optional[DocumentEntry]:
        """
        Retrieve a single document by its ID.

        Args:
            document_id: ID of the document to retrieve
            include_embedding: Whether to include the embedding vector

        Returns:
            DocumentEntry object or None if not found
        """
        docs = self.vector_database_provider.get_documents_by_ids(
            [document_id], include_embeddings=include_embedding
        )
        return docs[0] if docs else None

    def get_all_documents(
            self,
            limit: Optional[int] = None,
            offset: int = 0,
            include_embeddings: bool = False
    ) -> List[DocumentEntry]:
        """
        Retrieve all documents from the current collection.

        Args:
            limit: Maximum number of documents to retrieve
            offset: Number of documents to skip
            include_embeddings: Whether to include embedding vectors

        Returns:
            List of DocumentEntry objects
        """
        return self.vector_database_provider.get_all_documents(
            limit=limit, offset=offset, include_embeddings=include_embeddings
        )

    def search_by_metadata(
            self,
            metadata_filter: Dict[str, Any],
            limit: Optional[int] = None,
            offset: int = 0
    ) -> List[DocumentEntry]:
        """
        Search for documents by metadata criteria.

        Args:
            metadata_filter: Dictionary of metadata fields and values to match
            limit: Maximum number of documents to retrieve
            offset: Number of documents to skip

        Returns:
            List of matching DocumentEntry objects
        """
        return self.vector_database_provider.get_documents_by_filter(
            metadata_filter, limit=limit, offset=offset
        )

    def update_metadata(
            self,
            document_id: str,
            metadata: Dict[str, Any],
            merge: bool = True
    ) -> bool:
        """
        Update metadata for a specific document.

        Args:
            document_id: ID of the document to update
            metadata: New metadata dictionary
            merge: If True, merge with existing metadata; if False, replace

        Returns:
            True if successful, False otherwise
        """
        return self.vector_database_provider.update_document_metadata(
            document_id, metadata, merge
        )

    def batch_update_metadata(
            self,
            updates: Dict[str, Dict[str, Any]],
            merge: bool = True
    ) -> Dict[str, bool]:
        """
        Update metadata for multiple documents at once.

        Args:
            updates: Dictionary mapping document IDs to their new metadata
            merge: If True, merge with existing metadata; if False, replace

        Returns:
            Dictionary mapping document IDs to success status
        """
        return self.vector_database_provider.batch_update_metadata(updates, merge)

    def document_exists(self, document_id: str) -> bool:
        """
        Check if a document exists in the database.

        Args:
            document_id: ID of the document to check

        Returns:
            True if document exists, False otherwise
        """
        return self.vector_database_provider.exists(document_id)

    def get_document_count(self, metadata_filter: Optional[Dict[str, Any]] = None) -> int:
        """
        Get the count of documents in the current collection.

        Args:
            metadata_filter: Optional filter to count only matching documents

        Returns:
            Number of documents
        """
        return self.vector_database_provider.get_document_count(metadata_filter)

    # Collection management methods

    def create_collection(self, collection_name: str):
        """
        Create a new collection or switch to it if it exists.

        Args:
            collection_name: Name of the collection
        """
        self.vector_database_provider.create_or_set_current_collection(collection_name)

    def switch_collection(self, collection_name: str):
        """
        Switch to a different collection.

        Args:
            collection_name: Name of the collection to switch to
        """
        self.vector_database_provider.create_or_set_current_collection(collection_name)

    def list_collections(self) -> List[CollectionInfo]:
        """
        List all available collections.

        Returns:
            List of CollectionInfo objects
        """
        return self.vector_database_provider.list_collections()

    def delete_collection(self, collection_name: str) -> bool:
        """
        Delete a collection permanently.

        Args:
            collection_name: Name of the collection to delete

        Returns:
            True if successful, False otherwise
        """
        return self.vector_database_provider.delete_collection(collection_name)

    def clear_collection(self, collection_name: Optional[str] = None) -> int:
        """
        Remove all documents from a collection without deleting it.

        Args:
            collection_name: Name of collection to clear (None for current)

        Returns:
            Number of documents removed
        """
        return self.vector_database_provider.clear_collection(collection_name)

    def get_collection_info(self, collection_name: Optional[str] = None) -> CollectionInfo:
        """
        Get information about a collection.

        Args:
            collection_name: Name of collection (None for current)

        Returns:
            CollectionInfo object
        """
        return self.vector_database_provider.get_collection_info(collection_name)

    def copy_collection(
            self,
            source: str,
            target: str,
            overwrite: bool = False
    ) -> bool:
        """
        Copy all documents from one collection to another.

        Args:
            source: Name of the source collection
            target: Name of the target collection
            overwrite: If True, overwrite target if it exists

        Returns:
            True if successful
        """
        return self.vector_database_provider.copy_collection(source, target, overwrite)

    def rename_collection(self, old_name: str, new_name: str) -> bool:
        """
        Rename a collection.

        Args:
            old_name: Current name of the collection
            new_name: New name for the collection

        Returns:
            True if successful
        """
        return self.vector_database_provider.rename_collection(old_name, new_name)

    # Import/Export methods

    def export_collection(
            self,
            collection_name: Optional[str] = None,
            include_embeddings: bool = False
    ) -> Dict[str, Any]:
        """
        Export collection data for backup or migration.

        Args:
            collection_name: Name of collection to export (None for current)
            include_embeddings: Whether to include embedding vectors

        Returns:
            Dictionary containing collection data
        """
        return self.vector_database_provider.export_collection(
            collection_name, include_embeddings
        )

    def import_collection(
            self,
            data: Dict[str, Any],
            collection_name: str,
            overwrite: bool = False
    ) -> bool:
        """
        Import data into a collection.

        Args:
            data: Previously exported collection data
            collection_name: Name of the collection to import into
            overwrite: If True, overwrite existing collection

        Returns:
            True if successful
        """
        return self.vector_database_provider.import_collection(
            data, collection_name, overwrite
        )

    def backup_to_file(
            self,
            filepath: str,
            collection_name: Optional[str] = None,
            include_embeddings: bool = False
    ) -> bool:
        """
        Backup a collection to a JSON file.

        Args:
            filepath: Path to the backup file
            collection_name: Name of collection to backup (None for current)
            include_embeddings: Whether to include embedding vectors

        Returns:
            True if successful
        """
        import json

        try:
            data = self.export_collection(collection_name, include_embeddings)
            with open(filepath, 'w') as f:
                json.dump(data, f, indent=2)
            return True
        except Exception as e:
            print(f"Backup failed: {e}")
            return False

    def restore_from_file(
            self,
            filepath: str,
            collection_name: Optional[str] = None,
            overwrite: bool = False
    ) -> bool:
        """
        Restore a collection from a JSON backup file.

        Args:
            filepath: Path to the backup file
            collection_name: Name for the restored collection (uses name from backup if None)
            overwrite: If True, overwrite existing collection

        Returns:
            True if successful
        """
        import json

        try:
            with open(filepath, 'r') as f:
                data = json.load(f)

            target_name = collection_name or data.get("collection_name", "restored_collection")
            return self.import_collection(data, target_name, overwrite)
        except Exception as e:
            print(f"Restore failed: {e}")
            return False

    # Statistics and analytics methods

    def get_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about the current collection.

        Returns:
            Dictionary containing various statistics
        """
        stats = self.vector_database_provider.get_statistics()

        # Add additional statistics if needed
        all_docs = self.get_all_documents(limit=100)  # Sample for stats
        if all_docs:
            avg_length = sum(len(doc.content) for doc in all_docs) / len(all_docs)
            stats["average_document_length"] = avg_length

            # Analyze metadata fields
            metadata_fields = set()
            for doc in all_docs:
                if doc.metadata:
                    metadata_fields.update(doc.metadata.keys())
            stats["metadata_fields"] = list(metadata_fields)

        return stats

    def find_duplicates(self, similarity_threshold: float = 0.95) -> List[List[str]]:
        """
        Find duplicate or near-duplicate documents based on similarity.

        Args:
            similarity_threshold: Similarity score threshold (0-1)

        Returns:
            List of document ID groups that are duplicates
        """
        # This is a simplified implementation
        # In practice, you'd want to use more sophisticated deduplication
        duplicates = []
        all_docs = self.get_all_documents()

        for i, doc in enumerate(all_docs):
            # Query for similar documents
            results = self.retrieve_documents(doc.content, k=5)

            duplicate_group = [doc.id]
            for j, score in enumerate(results.scores):
                if score > similarity_threshold and results.ids[j] != doc.id:
                    duplicate_group.append(results.ids[j])

            if len(duplicate_group) > 1:
                # Sort to ensure consistent ordering
                duplicate_group.sort()
                if duplicate_group not in duplicates:
                    duplicates.append(duplicate_group)

        return duplicates

    def optimize_collection(self) -> Dict[str, Any]:
        """
        Optimize the current collection by removing duplicates and reorganizing.

        Returns:
            Dictionary with optimization results
        """
        results = {
            "duplicates_removed": 0,
            "original_count": self.get_document_count(),
        }

        # Find and remove duplicates
        duplicates = self.find_duplicates()
        for dup_group in duplicates:
            # Keep the first, remove the rest
            to_remove = dup_group[1:]
            removed = self.remove_documents(to_remove)
            results["duplicates_removed"] += removed

        results["final_count"] = self.get_document_count()
        results["space_saved"] = results["original_count"] - results["final_count"]

        return results