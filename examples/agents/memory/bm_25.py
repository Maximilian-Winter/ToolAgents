from ToolAgents.knowledge.vector_database.sentence_transformer_embeddings import SentenceTransformerEmbeddingProvider
from ToolAgents.knowledge.vector_database.cross_encoder_reranking import CrossEncoderRerankingProvider
from bm25_vector_database import BM25VectorDatabaseProvider
from ToolAgents.knowledge.rag import RAG

emb = SentenceTransformerEmbeddingProvider()  # kept for API compatibility
reranker = CrossEncoderRerankingProvider()
bm25 = BM25VectorDatabaseProvider(embedding_provider=emb, reranking_provider=reranker)

rag = RAG(vector_database_provider=bm25)
rag.add_document("Postgres supports full-text search with GIN indexes.", {"source": "notes"})
res = rag.retrieve_documents("How does Postgres FTS work?", k=3)
print(res)
