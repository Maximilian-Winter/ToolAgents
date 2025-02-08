import abc
import dataclasses
from typing import Optional, Any

from ToolAgents.knowledge import Document

@dataclasses.dataclass
class VectorSearchResult:
    ids: list[str]
    chunks: list[str]
    scores: list[float]
    metadata: Optional[list[dict[str, Any]]] = None

class VectorDatabaseProvider(abc.ABC):
    @abc.abstractmethod
    def add_document(self, document: Document) -> None:
        pass

    @abc.abstractmethod
    def add_documents(self, document: list[Document]) -> None:
        pass

    @abc.abstractmethod
    def add_text(self, document: str, metadata: dict) -> None:
        pass

    @abc.abstractmethod
    def add_texts(self, document: list[str], metadata: list[dict]) -> None:
        pass