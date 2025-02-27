from ToolAgents.knowledge.vector_database import RAG
from ToolAgents.knowledge.vector_database.implementations.chroma_db import (
    ChromaDbVectorDatabaseProvider,
)
from ToolAgents.knowledge.vector_database.implementations.sentence_transformer_embeddings import (
    SentenceTransformerEmbeddingProvider,
)
from ToolAgents.knowledge.vector_database.implementations.mbxai_reranking import (
    MXBAIRerankingProvider,
)

if __name__ == "__main__":
    rag = RAG(
        ChromaDbVectorDatabaseProvider(
            SentenceTransformerEmbeddingProvider(), MXBAIRerankingProvider()
        )
    )

    texts = [
        "Hello World!",
        "Hello Mom!",
        "Hi World!",
        "Hello Earth!",
        "Hello John!",
        "Hello Frank!",
        "Hola Tierra!",
        "Hello Pablo!",
        "Hello Sam!",
    ]
    rag.add_documents(texts)
    print(rag.retrieve_documents("Hola mundo!", 4))
