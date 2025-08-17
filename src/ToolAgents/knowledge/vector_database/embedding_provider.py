import abc
import dataclasses
import enum
from typing import Optional

from numpy import ndarray

class EmbeddingTask(enum.Enum):
    QUERY = "query"
    STORE = "store"
    CLUSTER = "cluster"
    CUSTOM = "custom"

@dataclasses.dataclass
class EmbeddingPrefixConfig:
    query: str = "query: "
    store: str = "store: "
    cluster: str = "cluster: "
    custom: str = "custom: "

@dataclasses.dataclass
class EmbeddingResult:
    embeddings: ndarray


class EmbeddingProvider(abc.ABC):

    def __init__(self, prefix_config: EmbeddingPrefixConfig):
        self.prefix_config = prefix_config

    @abc.abstractmethod
    def get_embedding(self, texts: list[str], embedding_task: EmbeddingTask) -> EmbeddingResult:
        pass