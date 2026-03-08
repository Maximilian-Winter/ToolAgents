from typing import Optional

from numpy import array
from openai import OpenAI

from ToolAgents.knowledge.vector_database import (
    EmbeddingProvider,
    EmbeddingResult,
    EmbeddingPrefixConfig,
    EmbeddingTask,
)


class OpenAIEmbeddingProvider(EmbeddingProvider):
    def __init__(
            self,
            model: str = "text-embedding-3-small",
            api_key: Optional[str] = None,
            base_url: Optional[str] = None,
            prefix_config: EmbeddingPrefixConfig = EmbeddingPrefixConfig(),
    ):
        """
        OpenAI-based EmbeddingProvider.

        - api_key: if None, uses OPENAI_API_KEY env var.
        - base_url: override the OpenAI API base URL (e.g. for proxies / self-hosted gateways).
        """
        super().__init__(prefix_config)

        client_kwargs: dict = {}
        if api_key is not None:
            client_kwargs["api_key"] = api_key
        if base_url is not None:
            client_kwargs["base_url"] = base_url

        self.client = OpenAI(**client_kwargs)
        self.model = model

    def _apply_prefixes(self, texts: list[str], embedding_task: EmbeddingTask) -> list[str]:
        if embedding_task == EmbeddingTask.QUERY:
            return [self.prefix_config.query + text for text in texts]
        elif embedding_task == EmbeddingTask.STORE:
            return [self.prefix_config.store + text for text in texts]
        elif embedding_task == EmbeddingTask.CLUSTER:
            return [self.prefix_config.cluster + text for text in texts]
        elif embedding_task == EmbeddingTask.CUSTOM:
            return [self.prefix_config.custom + text for text in texts]
        return texts

    def get_embedding(
            self,
            texts: list[str],
            embedding_task: EmbeddingTask = EmbeddingTask.STORE,
    ) -> EmbeddingResult:
        texts = self._apply_prefixes(texts, embedding_task)

        response = self.client.embeddings.create(
            model=self.model,
            input=texts,

        )

        embeddings = array([item.embedding for item in response.data])
        return EmbeddingResult(embeddings)