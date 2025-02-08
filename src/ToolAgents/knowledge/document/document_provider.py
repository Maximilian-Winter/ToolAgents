from abc import ABC, abstractmethod

from ToolAgents.knowledge import Document


class DocumentProvider(ABC):
    @abstractmethod
    def get_documents(self, **kwargs) -> list[Document]:
        pass