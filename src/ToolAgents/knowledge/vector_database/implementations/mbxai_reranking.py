from sentence_transformers import CrossEncoder

from ToolAgents.knowledge.vector_database import RerankingProvider, RerankingResult, RerankedDocument


class MXBAIRerankingProvider(RerankingProvider):

    def __init__(self, rerank_model: str ="mixedbread-ai/mxbai-rerank-xsmall-v1", trust_remote_code: bool = False, device: str = "cpu"):
        super().__init__()
        self.cross_encoder = CrossEncoder(rerank_model, trust_remote_code=trust_remote_code, device=device)

    def rerank_texts(self, query: str, texts: list, k: int, **kwargs) -> RerankingResult:
        results = self.cross_encoder.rank(query, texts, return_documents=True, top_k=k)
        documents = []
        for doc in results:
            text = doc.pop("text", None)
            if text is not None:
                documents.append(RerankedDocument(text, doc))
        return RerankingResult(reranked_documents=documents)