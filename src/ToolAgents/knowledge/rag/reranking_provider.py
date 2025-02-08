import abc
import dataclasses
from typing import Any


@dataclasses.dataclass
class RerankedDocument:
    content: str
    additional_data: dict[str, Any]

@dataclasses.dataclass
class RerankingResult:
    reranked_documents: list[RerankedDocument]

class RerankingProvider(abc.ABC):
    @abc.abstractmethod
    def rerank(self, query: str, documents: list, k: int, **kwargs) -> RerankingResult:
        pass
