import json

from ToolAgents.agent_memory.semantic_memory.memory_new import SemanticMemory, SemanticMemoryConfig, \
    SimpleExtractPatternStrategy, TimeBasedCleanupStrategy, SummarizationExtractPatternStrategy
from ToolAgents.knowledge.vector_database import (
    EmbeddingPrefixConfig
)

from ToolAgents.knowledge.vector_database.implementations.chroma_db_vector_database import ChromaDbVectorDatabaseProvider
from ToolAgents.knowledge.vector_database.implementations.sentence_transformer_embeddings import SentenceTransformerEmbeddingProvider

from ToolAgents.knowledge.vector_database.implementations.cross_encoder_reranking import CrossEncoderRerankingProvider

from ToolAgents.knowledge.vector_database import EmbeddingPrefixConfig

from ToolAgents.agent_memory.semantic_memory.hdbscan_cluster_embeddings_strategy import HDBSCANClusterEmbeddingsStrategy


def create_basic_memory_system():
    """Create a basic semantic memory system with default configuration"""

    # Configure embedding prefixes for different tasks
    prefix_config = EmbeddingPrefixConfig(
        query="search_query: ",
        store="search_document: ",
        cluster="cluster: ",
    )

    # Initialize embedding provider
    embedding_provider = SentenceTransformerEmbeddingProvider(
        sentence_transformer_model_path="all-MiniLM-L6-v2",
        trust_remote_code=False,
        device="cpu",
        prefix_config=prefix_config
    )

    # Initialize reranking provider (optional)
    reranking_provider = CrossEncoderRerankingProvider(
        rerank_model="mixedbread-ai/mxbai-rerank-xsmall-v1",
        device="cpu"
    )

    # Initialize vector database provider
    vector_db_provider = ChromaDbVectorDatabaseProvider(
        embedding_provider=embedding_provider,
        reranking_provider=reranking_provider,
        persistent_db_path="./semantic_memory",
        default_collection_name="working_memory",
        persistent=False
    )

    # Configure semantic memory
    config = SemanticMemoryConfig(
        cleanup_strategy=TimeBasedCleanupStrategy(
            working_memory_ttl_hours=24.0,
            long_term_memory_ttl_days=30.0,
            min_access_count=3
        ),
        enable_long_term_memory=True,
        minimum_cluster_size=4,
        minimum_cluster_similarity=0.875,
        debug_mode=False
    )

    # Create semantic memory instance
    memory = SemanticMemory(
        vector_db_provider=vector_db_provider,
        embedding_provider=embedding_provider,
        config=config
    )

    return memory


def create_advanced_memory_system(agent, summarizer_settings):
    """Create an advanced semantic memory system with LLM summarization and HDBSCAN clustering"""

    # Configure embedding prefixes for nomic embeddings
    prefix_config = EmbeddingPrefixConfig(
        query="search_query: ",
        store="search_document: ",
        cluster="cluster: ",
    )

    # Use more powerful embeddings
    embedding_provider = SentenceTransformerEmbeddingProvider(
        sentence_transformer_model_path="nomic-ai/nomic-embed-text-v1.5",
        trust_remote_code=True,
        device="cuda",
        prefix_config=prefix_config
    )

    # No reranking for this example
    vector_db_provider = ChromaDbVectorDatabaseProvider(
        embedding_provider=embedding_provider,
        reranking_provider=None,
        persistent_db_path="./semantic_memory_advanced",
        default_collection_name="working_memory",
        persistent=False
    )

    # Use advanced strategies
    pattern_strategy = SummarizationExtractPatternStrategy(
        agent=agent,
        summarizer_settings=summarizer_settings,
        user_name="User",
        assistant_name="Assistant",
        debug_mode=True
    )

    clustering_strategy = HDBSCANClusterEmbeddingsStrategy(
        min_cluster_size=3,
        min_samples=2
    )

    # Configure semantic memory with advanced strategies
    config = SemanticMemoryConfig(
        extract_pattern_strategy=pattern_strategy,
        cluster_embeddings_strategy=clustering_strategy,
        cleanup_strategy=TimeBasedCleanupStrategy(),
        enable_long_term_memory=True,
        minimum_cluster_size=3,
        minimum_cluster_similarity=0.75,
        debug_mode=True
    )

    # Create semantic memory instance
    memory = SemanticMemory(
        vector_db_provider=vector_db_provider,
        embedding_provider=embedding_provider,
        config=config
    )

    return memory


def example_usage():
    """Example of how to use the semantic memory system"""

    # Create a basic memory system
    memory = create_basic_memory_system()

    # Store some memories
    memory_id1 = memory.store(
        "The user likes Python programming and machine learning.",
        context={"conversation_id": "conv_001", "topic": "interests"}
    )

    memory_id2 = memory.store(
        "The user is working on a project about natural language processing.",
        context={"conversation_id": "conv_002", "topic": "work"}
    )

    memory_id3 = memory.store(
        "The user prefers clean code with meaningful variable names.",
        context={"conversation_id": "conv_003", "topic": "coding_style"}
    )

    # Store more similar memories to trigger pattern consolidation
    for i in range(5):
        memory.store(
            f"The user mentioned they enjoy working with Python for data science tasks.",
            context={"conversation_id": f"conv_{i + 4:03d}", "topic": "programming"}
        )

    # Recall memories related to programming
    recalled = memory.recall(
        query="What does the user like about programming?",
        n_results=3,
        context_filter=None,  # Could filter by topic, conversation_id, etc.
        alpha_recency=0.5,
        alpha_relevance=1.0,
        alpha_frequency=0.3
    )

    print("Recalled memories:")
    for mem in recalled:
        print(f"- Content: {mem['content'][:100]}...")
        print(f"  Similarity: {mem['similarity']:.3f}")
        print(f"  Rank Score: {mem['rank_score']:.3f}")
        print(f"  Type: {mem['memory_type']}")
        print()

    # Check memory statistics
    stats = memory.get_stats()
    print(f"Working memory count: {stats['working_count']}")
    print(f"Long-term memory count: {stats['long_term_count']}")


if __name__ == "__main__":
    example_usage()