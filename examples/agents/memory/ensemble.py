from ToolAgents.knowledge.vector_database.sentence_transformer_embeddings import SentenceTransformerEmbeddingProvider
from ToolAgents.knowledge.vector_database.cross_encoder_reranking import CrossEncoderRerankingProvider
from ToolAgents.knowledge.vector_database.chroma_db_vector_database import ChromaDbVectorDatabaseProvider
from bm25_vector_database import BM25VectorDatabaseProvider
from ensemble_vector_database import EnsembleVectorDatabaseProvider
from ToolAgents.knowledge.rag import RAG

emb = SentenceTransformerEmbeddingProvider()
reranker = CrossEncoderRerankingProvider()

dense = ChromaDbVectorDatabaseProvider(embedding_provider=emb, reranking_provider=None, persistent=False)
sparse = BM25VectorDatabaseProvider(embedding_provider=emb, reranking_provider=None)

hybrid = EnsembleVectorDatabaseProvider(
    embedding_provider=emb,
    dense_provider=dense,
    sparse_provider=sparse,
    reranking_provider=reranker,             # final stage rerank
    dense_weight=0.45,
    sparse_weight=0.55,
    dense_scores_are_similarities=False,     # Chroma returns distances
)

rag = RAG(vector_database_provider=hybrid)

