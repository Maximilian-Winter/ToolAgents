import abc
import dataclasses
import uuid
from typing import Optional, Any

from ToolAgents.knowledge import Document
from ToolAgents.knowledge.vector_database.reranking_provider import RerankingProvider
from ToolAgents.knowledge.vector_database.embedding_provider import EmbeddingProvider


@dataclasses.dataclass
class VectorSearchResult:
    ids: list[str]
    chunks: list[str]
    scores: list[float]
    metadata: Optional[list[dict[str, Any]]] = None

class VectorDatabaseProvider(abc.ABC):

    def __init__(self, embedding_provider: EmbeddingProvider, reranking_provider: RerankingProvider):
        self.embedding_provider = embedding_provider
        self.reranking_provider = reranking_provider

    @abc.abstractmethod
    def add_documents(self, documents: list[Document]) -> None:
        pass

    @abc.abstractmethod
    def add_texts(self, texts: list[str], metadata: list[dict]) -> None:
        pass

    @abc.abstractmethod
    def query(self, query: str, query_filter: Any = None, k: int = 3, **kwargs) -> VectorSearchResult:
        pass

    @abc.abstractmethod
    def create_or_set_current_collection(self, collection_name: str) -> None:
        pass

    @staticmethod
    def generate_unique_id():
        unique_id = str(uuid.uuid4())
        return unique_id

