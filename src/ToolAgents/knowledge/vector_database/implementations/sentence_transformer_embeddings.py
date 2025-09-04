from typing import Optional
from sentence_transformers import SentenceTransformer

from ToolAgents.knowledge.vector_database import EmbeddingProvider, EmbeddingResult, EmbeddingPrefixConfig, EmbeddingTask


class SentenceTransformerEmbeddingProvider(EmbeddingProvider):

    def __init__(self, sentence_transformer_model_path: str = "all-MiniLM-L6-v2",
                 trust_remote_code: bool = False, device: str = "cpu", prefix_config: EmbeddingPrefixConfig = EmbeddingPrefixConfig()):
        super().__init__(prefix_config)
        self.encoder = SentenceTransformer(
            sentence_transformer_model_path,
            trust_remote_code=trust_remote_code,
            device=device,
        )

    def get_embedding(self, texts: list[str], embedding_task: EmbeddingTask) -> EmbeddingResult:
        if embedding_task == EmbeddingTask.QUERY:
            texts = [self.prefix_config.query + text for text in texts]
        elif embedding_task == EmbeddingTask.STORE:
            texts = [self.prefix_config.store + text for text in texts]
        elif embedding_task == EmbeddingTask.CLUSTER:
            texts = [self.prefix_config.cluster + text for text in texts]
        elif embedding_task == EmbeddingTask.CUSTOM:
            texts = [self.prefix_config.custom + text for text in texts]

        embeddings = self.encoder.encode(texts)
        return EmbeddingResult(embeddings)