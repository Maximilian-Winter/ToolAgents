#!/usr/bin/env python3
"""
A sophisticated Retrieval-Augmented Generation (RAG) system that:
  - Splits documents into chunks using a recursive text splitter
  - Computes embeddings using OpenAI’s API
  - Indexes chunks in ChromaDB
  - Retrieves relevant chunks for a query and generates an answer using a language model
"""

import re
import openai
import chromadb
from chromadb.config import Settings


# =============================================================================
# Text Splitting Classes
# =============================================================================

class SimpleTextSplitter:
    """
    Splits text into consecutive overlapping (or non-overlapping) chunks.
    """

    def __init__(self, text, chunk_size, overlap=0):
        if chunk_size <= 0:
            raise ValueError("chunk_size must be a positive integer.")
        if overlap < 0:
            raise ValueError("overlap cannot be negative.")
        if overlap >= chunk_size:
            raise ValueError("overlap must be strictly less than chunk_size.")
        self.text = text
        self.chunk_size = chunk_size
        self.overlap = overlap

    def get_chunks(self):
        chunks = []
        start = 0
        while start < len(self.text):
            end = start + self.chunk_size
            chunks.append(self.text[start:end])
            start = end - self.overlap
        return chunks


class RecursiveCharacterTextSplitter:
    """
    Recursively splits text using a hierarchy of separators, and if necessary,
    falls back to fixed-size splitting with overlap.
    """

    def __init__(self, separators, chunk_size, chunk_overlap, length_function=len, keep_separator=False):
        if chunk_size <= 0:
            raise ValueError("chunk_size must be a positive integer.")
        if chunk_overlap < 0:
            raise ValueError("chunk_overlap cannot be negative.")
        if chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap must be strictly less than chunk_size.")
        self.separators = separators
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.length_function = length_function
        self.keep_separator = keep_separator

    def split_text(self, text, depth=0):
        # Base case: if all separators are exhausted, use fixed-size splitting.
        if depth == len(self.separators):
            return self._split_into_fixed_size(text)

        current_separator = self.separators[depth]
        if current_separator == "":
            return list(text)

        if self.keep_separator:
            # Use regex to split while capturing the separator.
            pieces = re.split(f"({re.escape(current_separator)})", text)
            merged_pieces = []
            for i in range(0, len(pieces), 2):
                piece = pieces[i]
                if i + 1 < len(pieces) and pieces[i + 1] == current_separator:
                    piece += pieces[i + 1]
                merged_pieces.append(piece)
            pieces = merged_pieces
        else:
            pieces = text.split(current_separator)

        refined_pieces = []
        for piece in pieces:
            if self.length_function(piece) > self.chunk_size:
                refined_pieces.extend(self.split_text(piece, depth + 1))
            else:
                refined_pieces.append(piece)

        # At the top level, merge pieces into valid chunks.
        if depth == 0:
            return self._merge_pieces(refined_pieces)
        return refined_pieces

    def _split_into_fixed_size(self, text):
        step = self.chunk_size - self.chunk_overlap  # guaranteed > 0
        if not text:
            return []
        chunks = [text[i: i + self.chunk_size] for i in range(0, len(text), step)]
        if len(chunks) > 1 and len(chunks[-1]) < self.chunk_overlap:
            chunks[-2] += chunks[-1]
            chunks.pop()
        return chunks

    def _merge_pieces(self, pieces):
        if not pieces:
            return []
        merged = []
        current_chunk = pieces[0]
        for piece in pieces[1:]:
            if self.length_function(current_chunk + piece) <= self.chunk_size:
                current_chunk += piece
            else:
                merged.append(current_chunk)
                if len(current_chunk) == self.chunk_size:
                    # Maintain overlap if the current chunk is exactly full.
                    current_chunk = current_chunk[-self.chunk_overlap:] + piece
                else:
                    current_chunk = piece
        merged.append(current_chunk)
        return merged


# =============================================================================
# Embedding Function
# =============================================================================

def get_embedding(text: str) -> list:
    """
    Get the embedding vector for the provided text using OpenAI's API.
    Ensure your OpenAI API key is configured in your environment.
    """
    response = openai.Embedding.create(
        input=text,
        model="text-embedding-ada-002"
    )
    return response["data"][0]["embedding"]


# =============================================================================
# RAG System Class
# =============================================================================

class RAGSystem:
    """
    A cohesive Retrieval-Augmented Generation system that integrates:
      - Document splitting
      - Embedding computation
      - Vector storage with ChromaDB
      - Query handling with generation via a language model
    """

    def __init__(self, collection_name="document_chunks",
                 chunk_size=500, chunk_overlap=50,
                 separators=["\n\n", "\n", " "],
                 keep_separator=True,
                 embedding_function=get_embedding):
        self.embedding_function = embedding_function
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.separators = separators
        self.keep_separator = keep_separator

        # Initialize the recursive text splitter.
        self.text_splitter = RecursiveCharacterTextSplitter(
            separators=self.separators,
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            keep_separator=self.keep_separator
        )

        # Set up the ChromaDB client and collection.
        self.client = chromadb.Client(Settings(
            chroma_db_impl="duckdb+parquet",
            persist_directory="./chroma_db"
        ))
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            embedding_function=self.embedding_function
        )

    def index_documents(self, documents):
        """
        Index a list of documents. Each document should be a dictionary with keys "id" and "text".
        """
        for doc in documents:
            doc_id = doc["id"]
            text = doc["text"]
            chunks = self.text_splitter.split_text(text)
            for i, chunk in enumerate(chunks):
                chunk_id = f"{doc_id}_chunk_{i}"
                self.collection.add(
                    documents=[chunk],
                    ids=[chunk_id],
                    metadatas=[{"source": doc_id}],
                )
                print(f"Indexed chunk '{chunk_id}' from document '{doc_id}'.")

    def answer_query(self, query, top_k=5, model="gpt-3.5-turbo", temperature=0.7):
        """
        Retrieve the top-k relevant chunks for the query and generate an answer using the language model.
        """
        results = self.collection.query(
            query_texts=[query],
            n_results=top_k
        )
        retrieved_chunks = results["documents"][0]
        print("Retrieved the following chunks:")
        for idx, chunk in enumerate(retrieved_chunks):
            preview = (chunk[:100] + "...") if len(chunk) > 100 else chunk
            print(f"Chunk {idx + 1}: {preview}")

        # Concatenate retrieved chunks to form the context.
        context = "\n\n".join(retrieved_chunks)
        prompt = f"""You are a helpful assistant. Use the following context to answer the query.

Context:
{context}

Query:
{query}

Answer:"""
        response = openai.ChatCompletion.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are a knowledgeable assistant."},
                {"role": "user", "content": prompt}
            ],
            temperature=temperature
        )
        answer = response["choices"][0]["message"]["content"]
        return answer


# =============================================================================
# Main Execution Example
# =============================================================================

if __name__ == "__main__":
    # Sample documents to index (replace with your actual content)
    documents = [
        {"id": "doc1",
         "text": "Your document text 1. It can be long, contain multiple paragraphs, and even have sections."},
        {"id": "doc2",
         "text": "Your document text 2. Another engaging piece of writing that might need splitting into chunks."}
    ]

    # Instantiate the RAG system
    print("Initializing the RAG system...")
    rag_system = RAGSystem()

    # Index the provided documents
    print("\nIndexing documents...")
    rag_system.index_documents(documents)

    # Process a sample query
    user_query = "What are the key points discussed in document 1?"
    print("\nProcessing query...")
    final_answer = rag_system.answer_query(user_query)

    print("\nFinal Answer:")
    print(final_answer)
