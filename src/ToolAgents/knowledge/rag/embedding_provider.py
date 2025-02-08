import abc
import dataclasses


@dataclasses.dataclass
class EmbeddingResult:
    embeddings: list

class EmbeddingProvider(abc.ABC):

    @abc.abstractmethod
    def get_embedding(self, texts: list[str]) -> EmbeddingResult:
        pass
