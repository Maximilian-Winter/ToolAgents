from ToolAgents.knowledge.default_implementation import ChromaDbVectorDatabaseProvider, \
    SentenceTransformerEmbeddingProvider, MXBAIRerankingProvider, RAG

if __name__ == "__main__":
    rag = RAG(ChromaDbVectorDatabaseProvider(SentenceTransformerEmbeddingProvider(), MXBAIRerankingProvider()))

    texts =[
        "Hello World!", "Hello Mom!", "Hi World!", "Hello Earth!", "Hello John!", "Hello Frank!", "Hola Tierra!",
        "Hello Pablo!", "Hello Sam!"
    ]
    rag.add_documents(texts)
    print(rag.retrieve_documents("Hola mundo!", 4))