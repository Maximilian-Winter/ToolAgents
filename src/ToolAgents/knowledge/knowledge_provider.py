import abc
from typing import Any

from ToolAgents.knowledge import Document


class KnowledgeProvider(abc.ABC):

    @abc.abstractmethod
    def query_knowledge(self, query: str, query_filter: Any = None, k: int = 3, **kwargs) -> list[str]:
        pass

    @abc.abstractmethod
    def add_knowledge(self, knowledge: list[str], **kwargs) -> None:
        pass

    @abc.abstractmethod
    def add_documents(self, document: list[Document]) -> None:
        pass