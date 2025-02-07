import uuid
import chromadb
from chromadb.api.types import IncludeEnum
from chromadb.utils import embedding_functions
from ragatouille import RAGPretrainedModel
from sentence_transformers import CrossEncoder


class RAGWithReranking:
    """
    Represents a chromadb vector database with a Colbert reranker.
    """

    def __init__(
        self,
        persistent_db_path="./retrieval_memory",
        embedding_model_name="BAAI/bge-small-en-v1.5",
        collection_name="retrieval_memory_collection",
        persistent: bool = True,
    ):
        self.cross_encoder = CrossEncoder("mixedbread-ai/mxbai-rerank-xsmall-v1")
        if persistent:
            self.client = chromadb.PersistentClient(path=persistent_db_path)
        else:
            self.client = chromadb.EphemeralClient()
        self.sentence_transformer_ef = (
            embedding_functions.SentenceTransformerEmbeddingFunction(
                model_name=embedding_model_name
            )
        )
        self.collection = self.client.get_or_create_collection(
            name=collection_name, embedding_function=self.sentence_transformer_ef
        )

    def add_document(self, document: str, metadata: dict = None):
        """Add a memory with a given description and importance to the memory stream."""
        mem = [document]
        ids = [str(self.generate_unique_id())]
        self.collection.add(documents=mem, metadatas=metadata, ids=ids)

    def add_documents(self, documents: list[str], metadata: dict = None):
        """Add a memory with a given description and importance to the memory stream."""
        mem = documents
        ids = [str(self.generate_unique_id()) for _ in range(len(documents))]
        self.collection.add(documents=mem, metadatas=metadata, ids=ids)

    def retrieve_documents(self, query: str, k):
        query_embedding = self.sentence_transformer_ef([query])
        query_result = self.collection.query(
            query_embedding,
            n_results=k,
            include=[IncludeEnum.metadatas, IncludeEnum.embeddings, IncludeEnum.documents, IncludeEnum.distances],
        )
        documents = []
        for doc in query_result["documents"][0]:
            documents.append(doc)
        results = self.cross_encoder.rank(query, documents, return_documents=True, top_k=k)
        return results

    @staticmethod
    def generate_unique_id():
        unique_id = str(uuid.uuid4())
        return unique_id
