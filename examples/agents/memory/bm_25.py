"""Dependency-light keyword retrieval with the BM25 vector-database provider."""

from ToolAgents.knowledge.vector_database import RAG
from ToolAgents.knowledge.vector_database.implementations.bm25_database import (
    BM25VectorDatabaseProvider,
)


if __name__ == "__main__":
    bm25 = BM25VectorDatabaseProvider(embedding_provider=None)
    rag = RAG(vector_database_provider=bm25)

    rag.add_documents(
        [
            "Postgres supports full-text search with GIN indexes.",
            "SQLite can run lightweight local applications without a server.",
            "BM25 is a sparse retrieval algorithm based on term frequency.",
        ],
        [{"source": "notes"}, {"source": "notes"}, {"source": "notes"}],
    )

    result = rag.retrieve_documents("How does Postgres search work?", k=2)
    for chunk, score in zip(result.chunks, result.scores):
        print(f"{score:.3f}: {chunk}")

