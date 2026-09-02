from ToolAgents.knowledge.vector_database.vector_database_provider import (
    VectorDatabaseProvider,
)


class RAG:
    """Retrieval over any vector database provider.

    A thin facade: documents go in through ``add_document`` /
    ``add_documents`` and come back out through ``retrieve_documents``. The
    storage, embedding and reranking strategies are whatever providers you
    hand it, not a fixed choice.
    """

    def __init__(self, vector_database_provider: VectorDatabaseProvider):
        self.vector_database_provider = vector_database_provider

    def add_document(self, document: str, metadata: dict = None):
        self.vector_database_provider.add_texts([document], [metadata])

    def add_documents(self, documents: list[str], metadata: list[dict] = None):
        self.vector_database_provider.add_texts(documents, metadata)

    def retrieve_documents(self, query: str, k):
        return self.vector_database_provider.query(query, k=k)
