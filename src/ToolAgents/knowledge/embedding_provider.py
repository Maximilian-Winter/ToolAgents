import abc
import dataclasses
from typing import Union

from numpy import ndarray


@dataclasses.dataclass
class EmbeddingResult:
    embeddings: ndarray

class EmbeddingProvider(abc.ABC):

    @abc.abstractmethod
    def get_embedding(self, texts: list[str]) -> EmbeddingResult:
        pass
