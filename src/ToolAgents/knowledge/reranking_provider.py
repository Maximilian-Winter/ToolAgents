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
    def rerank_texts(self, query: str, texts: list, k: int, **kwargs) -> RerankingResult:
        pass