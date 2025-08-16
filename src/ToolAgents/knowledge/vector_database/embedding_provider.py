import abc
import dataclasses
from typing import Union, Optional, List

from numpy import ndarray


@dataclasses.dataclass
class EmbeddingResult:
    embeddings: ndarray


class EmbeddingProvider(abc.ABC):
    """
    Abstract base class for embedding providers.
    """

    @abc.abstractmethod
    def get_embedding(
            self,
            texts: List[str],
            prefix: Optional[str] = None,
            **kwargs
    ) -> EmbeddingResult:
        """
        Generate embeddings for the given texts.

        Args:
            texts: List of text strings to embed
            prefix: Optional prefix to prepend to each text before embedding
            **kwargs: Additional provider-specific parameters

        Returns:
            EmbeddingResult containing the embeddings
        """
        pass

    @abc.abstractmethod
    def get_embedding_with_prefixes(
            self,
            texts: List[str],
            prefixes: Union[str, List[str], None] = None,
            **kwargs
    ) -> EmbeddingResult:
        """
        Generate embeddings with different prefixes for each text.

        Args:
            texts: List of text strings to embed
            prefixes: Single prefix for all texts, or list of prefixes (one per text)
            **kwargs: Additional provider-specific parameters

        Returns:
            EmbeddingResult containing the embeddings
        """
        pass


class PrefixConfig:
    """Configuration for embedding prefixes used in different contexts."""

    def __init__(
            self,
            store_prefix: Optional[str] = None,
            recall_prefix: Optional[str] = None,
            cluster_prefix: Optional[str] = None,
            default_prefix: Optional[str] = None
    ):
        """
        Initialize prefix configuration.

        Args:
            store_prefix: Prefix used when storing documents
            recall_prefix: Prefix used when querying/recalling
            cluster_prefix: Prefix used when clustering
            default_prefix: Default prefix when none specified
        """
        self.store_prefix = store_prefix
        self.recall_prefix = recall_prefix
        self.cluster_prefix = cluster_prefix
        self.default_prefix = default_prefix

    def get_prefix(self, context: str = "default") -> Optional[str]:
        """Get the appropriate prefix for the given context."""
        prefix_map = {
            "store": self.store_prefix,
            "recall": self.recall_prefix,
            "query": self.recall_prefix,  # Alias for recall
            "cluster": self.cluster_prefix,
            "default": self.default_prefix
        }
        return prefix_map.get(context, self.default_prefix)