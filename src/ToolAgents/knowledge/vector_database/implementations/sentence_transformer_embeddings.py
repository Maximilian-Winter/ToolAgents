from typing import List, Optional, Union
from sentence_transformers import SentenceTransformer

from ToolAgents.knowledge.vector_database import (
    EmbeddingProvider,
    EmbeddingResult,
    PrefixConfig
)


class SentenceTransformerEmbeddingProvider(EmbeddingProvider):
    """
    Sentence Transformer embedding provider with prefix support.
    """

    def __init__(
            self,
            sentence_transformer_model_path: str = "all-MiniLM-L6-v2",
            trust_remote_code: bool = False,
            device: str = "cpu",
            prefix_config: Optional[PrefixConfig] = None,
            **encoder_kwargs
    ):
        """
        Initialize the Sentence Transformer embedding provider.

        Args:
            sentence_transformer_model_path: Model name or path
            trust_remote_code: Whether to trust remote code
            device: Device to use ('cpu', 'cuda', etc.)
            prefix_config: Optional prefix configuration
            **encoder_kwargs: Additional arguments for the encoder
        """
        self.encoder = SentenceTransformer(
            sentence_transformer_model_path,
            trust_remote_code=trust_remote_code,
            device=device
        )
        self.prefix_config = prefix_config or PrefixConfig()
        self.encoder_kwargs = encoder_kwargs

    def get_embedding(
            self,
            texts: List[str],
            prefix: Optional[str] = None,
            **kwargs
    ) -> EmbeddingResult:
        """
        Generate embeddings with an optional prefix.

        Args:
            texts: List of texts to embed
            prefix: Optional prefix to prepend to each text
            **kwargs: Additional encoding parameters

        Returns:
            EmbeddingResult with embeddings
        """
        # Merge kwargs with default encoder kwargs
        encoding_kwargs = {**self.encoder_kwargs, **kwargs}

        if prefix:
            texts_with_prefix = [prefix + text for text in texts]
            embeddings = self.encoder.encode(texts_with_prefix, **encoding_kwargs)
        else:
            embeddings = self.encoder.encode(texts, **encoding_kwargs)

        return EmbeddingResult(embeddings)

    def get_embedding_with_prefixes(
            self,
            texts: List[str],
            prefixes: Union[str, List[str], None] = None,
            **kwargs
    ) -> EmbeddingResult:
        """
        Generate embeddings with different prefixes for each text.

        Args:
            texts: List of texts to embed
            prefixes: Single prefix for all texts, or list of prefixes
            **kwargs: Additional encoding parameters

        Returns:
            EmbeddingResult with embeddings
        """
        encoding_kwargs = {**self.encoder_kwargs, **kwargs}

        if prefixes is None:
            return self.get_embedding(texts, **kwargs)

        if isinstance(prefixes, str):
            # Single prefix for all texts
            return self.get_embedding(texts, prefix=prefixes, **kwargs)

        # List of prefixes (one per text)
        if len(prefixes) != len(texts):
            raise ValueError(f"Number of prefixes ({len(prefixes)}) must match number of texts ({len(texts)})")

        texts_with_prefixes = [
            (prefix + text if prefix else text)
            for prefix, text in zip(prefixes, texts)
        ]
        embeddings = self.encoder.encode(texts_with_prefixes, **encoding_kwargs)

        return EmbeddingResult(embeddings)

    def get_embedding_for_context(
            self,
            texts: List[str],
            context: str = "default",
            **kwargs
    ) -> EmbeddingResult:
        """
        Generate embeddings using the appropriate prefix for the context.

        Args:
            texts: List of texts to embed
            context: Context name ('store', 'recall', 'cluster', etc.)
            **kwargs: Additional encoding parameters

        Returns:
            EmbeddingResult with embeddings
        """
        prefix = self.prefix_config.get_prefix(context)
        return self.get_embedding(texts, prefix=prefix, **kwargs)