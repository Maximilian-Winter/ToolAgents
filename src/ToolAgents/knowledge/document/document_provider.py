from abc import ABC, abstractmethod

from .document import Document


class DocumentProvider(ABC):
    @abstractmethod
    def get_documents(self, **kwargs) -> list[Document]:
        pass

