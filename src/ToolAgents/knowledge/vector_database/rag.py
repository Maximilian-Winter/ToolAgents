from ToolAgents.knowledge.vector_database.vector_database_provider import VectorDatabaseProvider


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