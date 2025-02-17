from sentence_transformers import SentenceTransformer

from ToolAgents.knowledge.vector_database import EmbeddingProvider, EmbeddingResult


class SentenceTransformerEmbeddingProvider(EmbeddingProvider):

    def __init__(self, sentence_transformer_model_path: str = "all-MiniLM-L6-v2", trust_remote_code: bool = False, device: str = "cpu"):
        self.encoder = SentenceTransformer(
            sentence_transformer_model_path,
            trust_remote_code=trust_remote_code,
            device=device
        )
    def get_embedding(self, texts: list[str]) -> EmbeddingResult:
        return EmbeddingResult(self.encoder.encode(texts))
