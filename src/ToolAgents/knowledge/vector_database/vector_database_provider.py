import abc
import dataclasses
import uuid
from typing import Optional, Any, List, Dict

from ToolAgents.knowledge import Document
from ToolAgents.knowledge.vector_database.reranking_provider import RerankingProvider
from ToolAgents.knowledge.vector_database.embedding_provider import EmbeddingProvider


@dataclasses.dataclass
class VectorSearchResult:
    ids: list[str]
    chunks: list[str]
    scores: list[float]
    metadata: Optional[list[dict[str, Any]]] = None


@dataclasses.dataclass
class CollectionInfo:
    """Information about a collection in the vector database."""
    name: str
    document_count: int
    metadata: Optional[Dict[str, Any]] = None
    created_at: Optional[str] = None
    updated_at: Optional[str] = None


@dataclasses.dataclass
class DocumentEntry:
    """Represents a document entry in the vector database."""
    id: str
    content: str
    metadata: Optional[Dict[str, Any]] = None
    embedding: Optional[List[float]] = None


class VectorDatabaseProvider(abc.ABC):

    def __init__(
            self,
            embedding_provider: EmbeddingProvider,
            reranking_provider: RerankingProvider,
    ):
        self.embedding_provider = embedding_provider
        self.reranking_provider = reranking_provider

    @abc.abstractmethod
    def add_documents(self, documents: list[Document]) -> None:
        """Add multiple structured documents to the database."""
        pass

    @abc.abstractmethod
    def add_texts(self, texts: list[str], metadata: list[dict]) -> None:
        """Add raw texts with metadata to the database."""
        pass

    @abc.abstractmethod
    def query(
            self, query: str, query_filter: Any = None, k: int = 3, **kwargs
    ) -> VectorSearchResult:
        """Query the database for similar documents."""
        pass

    @abc.abstractmethod
    def create_or_set_current_collection(self, collection_name: str) -> None:
        """Create a new collection or switch to an existing one."""
        pass

    # New abstract methods for enhanced management

    @abc.abstractmethod
    def remove_documents(self, document_ids: List[str]) -> int:
        """
        Remove documents by their IDs.

        Args:
            document_ids: List of document IDs to remove

        Returns:
            Number of documents successfully removed
        """
        pass

    @abc.abstractmethod
    def remove_documents_by_filter(self, filter_dict: Dict[str, Any]) -> int:
        """
        Remove documents matching the specified filter criteria.

        Args:
            filter_dict: Dictionary of metadata fields and values to filter by

        Returns:
            Number of documents removed
        """
        pass

    @abc.abstractmethod
    def get_all_documents(
            self,
            limit: Optional[int] = None,
            offset: int = 0,
            include_embeddings: bool = False
    ) -> List[DocumentEntry]:
        """
        Retrieve all documents from the current collection.

        Args:
            limit: Maximum number of documents to retrieve (None for all)
            offset: Number of documents to skip
            include_embeddings: Whether to include embedding vectors

        Returns:
            List of DocumentEntry objects
        """
        pass

    @abc.abstractmethod
    def get_documents_by_ids(
            self,
            document_ids: List[str],
            include_embeddings: bool = False
    ) -> List[DocumentEntry]:
        """
        Retrieve specific documents by their IDs.

        Args:
            document_ids: List of document IDs to retrieve
            include_embeddings: Whether to include embedding vectors

        Returns:
            List of DocumentEntry objects
        """
        pass

    @abc.abstractmethod
    def get_documents_by_filter(
            self,
            filter_dict: Dict[str, Any],
            limit: Optional[int] = None,
            offset: int = 0,
            include_embeddings: bool = False
    ) -> List[DocumentEntry]:
        """
        Retrieve documents matching the specified filter criteria.

        Args:
            filter_dict: Dictionary of metadata fields and values to filter by
            limit: Maximum number of documents to retrieve
            offset: Number of documents to skip
            include_embeddings: Whether to include embedding vectors

        Returns:
            List of DocumentEntry objects
        """
        pass

    @abc.abstractmethod
    def update_document_metadata(
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
            merge: If True, merge with existing metadata; if False, replace entirely

        Returns:
            True if successful, False otherwise
        """
        pass

    @abc.abstractmethod
    def list_collections(self) -> List[CollectionInfo]:
        """
        List all available collections in the database.

        Returns:
            List of CollectionInfo objects
        """
        pass

    @abc.abstractmethod
    def delete_collection(self, collection_name: str) -> bool:
        """
        Delete a collection from the database.

        Args:
            collection_name: Name of the collection to delete

        Returns:
            True if successful, False if collection doesn't exist
        """
        pass

    @abc.abstractmethod
    def get_collection_info(self, collection_name: Optional[str] = None) -> CollectionInfo:
        """
        Get information about a specific collection or the current collection.

        Args:
            collection_name: Name of the collection (None for current collection)

        Returns:
            CollectionInfo object
        """
        pass

    @abc.abstractmethod
    def clear_collection(self, collection_name: Optional[str] = None) -> int:
        """
        Remove all documents from a collection without deleting the collection itself.

        Args:
            collection_name: Name of the collection to clear (None for current collection)

        Returns:
            Number of documents removed
        """
        pass

    @abc.abstractmethod
    def get_document_count(self, filter_dict: Optional[Dict[str, Any]] = None) -> int:
        """
        Get the count of documents in the current collection.

        Args:
            filter_dict: Optional filter to count only matching documents

        Returns:
            Number of documents
        """
        pass

    @abc.abstractmethod
    def exists(self, document_id: str) -> bool:
        """
        Check if a document exists in the current collection.

        Args:
            document_id: ID of the document to check

        Returns:
            True if document exists, False otherwise
        """
        pass

    @abc.abstractmethod
    def batch_update_metadata(
            self,
            updates: Dict[str, Dict[str, Any]],
            merge: bool = True
    ) -> Dict[str, bool]:
        """
        Update metadata for multiple documents in a single operation.

        Args:
            updates: Dictionary mapping document IDs to their new metadata
            merge: If True, merge with existing metadata; if False, replace entirely

        Returns:
            Dictionary mapping document IDs to success status
        """
        pass

    @abc.abstractmethod
    def export_collection(
            self,
            collection_name: Optional[str] = None,
            include_embeddings: bool = False
    ) -> Dict[str, Any]:
        """
        Export all data from a collection for backup or migration.

        Args:
            collection_name: Name of collection to export (None for current)
            include_embeddings: Whether to include embedding vectors

        Returns:
            Dictionary containing collection data and metadata
        """
        pass

    @abc.abstractmethod
    def import_collection(
            self,
            data: Dict[str, Any],
            collection_name: str,
            overwrite: bool = False
    ) -> bool:
        """
        Import data into a collection from an export.

        Args:
            data: Previously exported collection data
            collection_name: Name of the collection to import into
            overwrite: If True, overwrite existing collection; if False, merge

        Returns:
            True if successful
        """
        pass

    @staticmethod
    def generate_unique_id():
        """Generate a unique identifier for documents."""
        unique_id = str(uuid.uuid4())
        return unique_id

    # Helper methods that can be implemented in the base class

    def get_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about the current collection.

        Returns:
            Dictionary containing statistics like document count,
            average document length, metadata field distribution, etc.
        """
        info = self.get_collection_info()
        doc_count = self.get_document_count()

        return {
            "collection_name": info.name,
            "document_count": doc_count,
            "created_at": info.created_at,
            "updated_at": info.updated_at,
            "metadata": info.metadata
        }

    def copy_collection(
            self,
            source_collection: str,
            target_collection: str,
            overwrite: bool = False
    ) -> bool:
        """
        Copy all documents from one collection to another.

        Args:
            source_collection: Name of the source collection
            target_collection: Name of the target collection
            overwrite: If True, overwrite target if it exists

        Returns:
            True if successful
        """
        # Export from source
        data = self.export_collection(source_collection)

        # Import to target
        return self.import_collection(data, target_collection, overwrite)

    def rename_collection(
            self,
            old_name: str,
            new_name: str
    ) -> bool:
        """
        Rename a collection.

        Args:
            old_name: Current name of the collection
            new_name: New name for the collection

        Returns:
            True if successful
        """
        # Default implementation: copy and delete
        if self.copy_collection(old_name, new_name):
            return self.delete_collection(old_name)
        return False