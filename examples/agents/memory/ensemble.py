"""Hybrid dense and sparse retrieval with the current vector database imports.

Install the optional memory dependencies before running:

    pip install "ToolAgents[memory]"
"""

from ToolAgents.knowledge.vector_database import RAG
from ToolAgents.knowledge.vector_database.implementations.bm25_database import (
    BM25VectorDatabaseProvider,
)
from ToolAgents.knowledge.vector_database.implementations.chroma_db_vector_database import (
    ChromaDbVectorDatabaseProvider,
)
from ToolAgents.knowledge.vector_database.implementations.cross_encoder_reranking import (
    CrossEncoderRerankingProvider,
)
from ToolAgents.knowledge.vector_database.implementations.ensemble_vector_database import (
    EnsembleVectorDatabaseProvider,
)
from ToolAgents.knowledge.vector_database.implementations.sentence_transformer_embeddings import (
    SentenceTransformerEmbeddingProvider,
)


if __name__ == "__main__":
    embeddings = SentenceTransformerEmbeddingProvider()
    reranker = CrossEncoderRerankingProvider()

    dense = ChromaDbVectorDatabaseProvider(
        embedding_provider=embeddings,
        reranking_provider=None,
        persistent=False,
    )
    sparse = BM25VectorDatabaseProvider(
        embedding_provider=embeddings,
        reranking_provider=None,
    )

    hybrid = EnsembleVectorDatabaseProvider(
        embedding_provider=embeddings,
        dense_provider=dense,
        sparse_provider=sparse,
        reranking_provider=reranker,
        dense_weight=0.45,
        sparse_weight=0.55,
        dense_scores_are_similarities=False,
    )

    rag = RAG(vector_database_provider=hybrid)
    rag.add_documents(
        [
            "The Thornqueen boss fight depends on readable mist phase cues.",
            "QA is tracking Thornqueen visibility bugs and controller remapping.",
            "VFX backlog includes mist silhouettes and root burst timing.",
        ]
    )

    print(rag.retrieve_documents("mist visibility blocker", k=2))

