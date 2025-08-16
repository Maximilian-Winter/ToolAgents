import abc
import dataclasses
import re
import uuid
import json
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor
from typing import List, Dict, Optional, Any, Mapping

import numpy as np
from torch import Tensor

from ToolAgents.data_models.messages import ChatMessage
from ToolAgents.utilities.message_template import MessageTemplate
from ToolAgents.provider.llm_provider import ProviderSettings
from ToolAgents.agents.base_llm_agent import BaseToolAgent

# Import the abstract interfaces
from ToolAgents.knowledge.vector_database import (
    VectorDatabaseProvider,
    EmbeddingProvider,
    RerankingProvider,
    PrefixConfig
)
from ToolAgents.knowledge.vector_database.implementations.chroma_db import ChromaDbVectorDatabaseProvider
from ToolAgents.knowledge.vector_database.implementations.sentence_transformer_embeddings import (
    SentenceTransformerEmbeddingProvider
)


# =============================================================================
# Extraction Pattern Strategies (unchanged)
# =============================================================================

class ExtractPatternStrategy(abc.ABC):
    """Abstract base class for extraction pattern strategies."""

    @abc.abstractmethod
    def extract_pattern(
            self,
            pattern_id,
            documents: List[str],
            metadatas: list[Mapping[str, str | int | float | bool]],
            timestamp=datetime.now().isoformat(),
    ) -> Dict:
        pass


class SimpleExtractPatternStrategy(ExtractPatternStrategy):
    """A simple extraction strategy that aggregates documents by concatenation."""

    def extract_pattern(
            self,
            pattern_id,
            documents: List[str],
            metadatas: List[Dict],
            timestamp=datetime.now().isoformat(),
    ) -> Dict:
        counters = [m["access_count"] for m in metadatas]
        pattern_metadata = {
            "type": "pattern",
            "timestamp": timestamp,
            "pattern_id": pattern_id,
            "last_access_timestamp": timestamp,
            "access_count": sum(counters),
            "source_count": len(documents),
            "source_timestamps": json.dumps([m["timestamp"] for m in metadatas]),
        }
        return {"content": "\n".join(documents), "metadata": pattern_metadata}


# Template for summarization
sum_chat_turns_template = MessageTemplate.from_string(
    """You will be analyzing a collection of chat turns to extract information about {USER_NAME} and their relationship with {ASSISTANT_NAME}. Here are the chat turns:

<chat_turns>
{CHAT_TURNS}
</chat_turns>

Your task is to carefully read through these chat turns and extract all relevant information about the user named {USER_NAME} and their relationship with {ASSISTANT_NAME}. This information may include personal details, preferences, interactions, or any other relevant data that can be inferred from the conversation.

Follow these steps:

1. Read through the chat turns thoroughly, paying close attention to any mentions of {USER_NAME} or interactions between {USER_NAME} and {ASSISTANT_NAME}.

2. Extract and note down any information about {USER_NAME}, such as:
   - Personal details (age, occupation, location, etc.)
   - Preferences or interests
   - Personality traits
   - Any other relevant information

3. Analyze the interactions between {USER_NAME} and {ASSISTANT_NAME} to understand their relationship. Look for:
   - Frequency of interactions
   - Tone of conversation
   - Types of requests or questions from {USER_NAME}
   - Any expressed feelings or attitudes towards {ASSISTANT_NAME}

4. Organize the extracted information into two main categories:
   a) Information about {USER_NAME}
   b) Relationship between {USER_NAME} and {ASSISTANT_NAME}

Present your findings in the following format:

<extracted_information>
<user_info>
[List all relevant information about {USER_NAME} here]
</user_info>

<relationship_info>
[Describe the relationship between {USER_NAME} and {ASSISTANT_NAME} based on the analyzed interactions]
</relationship_info>
</extracted_information>

Remember to base your analysis solely on the information provided in the chat turns. Do not make assumptions or include information that is not directly stated or strongly implied in the conversation."""
)


class SummarizationExtractPatternStrategy(ExtractPatternStrategy):
    """An extraction strategy that uses an LLM to summarize patterns."""

    def __init__(
            self,
            agent: BaseToolAgent,
            summarizer_settings: ProviderSettings,
            user_name: str,
            assistant_name: str,
            chat_turn_summary: MessageTemplate = sum_chat_turns_template,
            debug_mode: bool = False,
    ):
        self.agent = agent
        self.user_name = user_name
        self.assistant_name = assistant_name
        self.debug_mode = debug_mode
        self.summarizer_settings = summarizer_settings
        self.chat_turn_summary = chat_turn_summary

    def extract_pattern(
            self,
            pattern_id,
            documents: List[str],
            metadatas: List[Dict],
            timestamp=datetime.now().isoformat(),
    ) -> Dict:
        counters = [m["access_count"] for m in metadatas]
        pattern_metadata = {
            "type": "pattern",
            "timestamp": timestamp,
            "pattern_id": pattern_id,
            "last_access_timestamp": timestamp,
            "access_count": sum(counters),
            "source_count": len(documents),
            "source_timestamps": json.dumps([m["timestamp"] for m in metadatas]),
        }

        prompt = self.chat_turn_summary.generate_message_content(
            CHAT_TURNS="\n\n---\n\n".join(documents),
            USER_NAME=self.user_name,
            ASSISTANT_NAME=self.assistant_name,
        )

        if self.debug_mode:
            print(prompt)

        result = self.agent.get_response(
            messages=[ChatMessage.create_user_message(prompt)],
            settings=self.summarizer_settings,
        )

        match = re.findall(r"<summary>(.*?)</summary>", result.response, re.DOTALL)
        patterns = []
        for content in match:
            patterns.append(content.replace("**", ""))
            if self.debug_mode:
                print(content.replace("**", ""), flush=True)

        if len(patterns) == 0:
            return {
                "content": result.response.replace("**", ""),
                "metadata": pattern_metadata,
            }
        return {"content": "\n".join(patterns), "metadata": pattern_metadata}


# =============================================================================
# Embedding Clustering Strategies (unchanged)
# =============================================================================

class ClusterEmbeddingsStrategy(abc.ABC):
    """Abstract base class for clustering embeddings."""

    @abc.abstractmethod
    def cluster_embeddings(
            self, embeddings: List[np.ndarray], minimum_cluster_similarity: float = 0.75
    ) -> List[List[int]]:
        pass


class SimpleClusterEmbeddingsStrategy(ClusterEmbeddingsStrategy):
    """A simple clustering strategy using cosine similarity."""

    def cluster_embeddings(
            self,
            embeddings: List[np.ndarray | Tensor],
            minimum_cluster_similarity: float = 0.75,
    ) -> List[List[int]]:
        embeddings_array = np.stack(embeddings)
        norms = np.linalg.norm(embeddings_array, axis=1)
        norms[norms == 0] = 1
        normalized = embeddings_array / norms[:, np.newaxis]
        similarity_matrix = np.dot(normalized, normalized.T)

        clusters = []
        used_indices = set()

        for i in range(len(embeddings)):
            if i in used_indices:
                continue

            cluster = [i]
            used_indices.add(i)

            for j in range(i + 1, len(embeddings)):
                if (
                        j not in used_indices
                        and similarity_matrix[i, j] > minimum_cluster_similarity
                ):
                    cluster.append(j)
                    used_indices.add(j)

            clusters.append(cluster)

        return clusters


# =============================================================================
# Cleanup Strategies - Refactored for Vector Database Interface
# =============================================================================

class CleanupStrategy(abc.ABC):
    """Abstract base class for memory cleanup strategies."""

    @abc.abstractmethod
    def cleanup_working_memory(
            self, vector_db: VectorDatabaseProvider, current_date: datetime
    ) -> None:
        """Clean up the working memory based on time and access counts."""
        pass

    @abc.abstractmethod
    def cleanup_long_term_memory(
            self, vector_db: VectorDatabaseProvider, current_date: datetime
    ) -> None:
        """Clean up the long-term memory based on time since last access."""
        pass


class TimeBasedCleanupStrategy(CleanupStrategy):
    """A cleanup strategy that removes memories based on age and access frequency."""

    def __init__(
            self,
            working_memory_ttl_hours: float = 24.0,
            long_term_memory_ttl_days: float = 30.0,
            min_access_count: int = 3,
    ):
        self.working_memory_ttl = timedelta(hours=working_memory_ttl_hours)
        self.long_term_memory_ttl = timedelta(days=long_term_memory_ttl_days)
        self.min_access_count = min_access_count

    def cleanup_working_memory(
            self, vector_db: VectorDatabaseProvider, current_date: datetime
    ) -> None:
        """Remove old working memories with low access counts."""
        # Switch to working memory collection
        vector_db.create_or_set_current_collection("working_memory")

        # Get all documents with metadata
        memories = vector_db.get_all_documents(include_embeddings=False)

        to_delete = []
        for memory in memories:
            if memory.metadata:
                last_access = datetime.fromisoformat(
                    memory.metadata["last_access_timestamp"]
                )
                age = current_date - last_access

                if (
                        age > self.working_memory_ttl
                        and memory.metadata["access_count"] < self.min_access_count
                ):
                    to_delete.append(memory.id)

        if to_delete:
            vector_db.remove_documents(to_delete)

    def cleanup_long_term_memory(
            self, vector_db: VectorDatabaseProvider, current_date: datetime
    ) -> None:
        """Remove very old long-term memories."""
        # Switch to long-term memory collection
        vector_db.create_or_set_current_collection("long_term_memory")

        memories = vector_db.get_all_documents(include_embeddings=False)

        to_delete = []
        for memory in memories:
            if memory.metadata:
                last_access = datetime.fromisoformat(
                    memory.metadata["last_access_timestamp"]
                )
                age = current_date - last_access

                if age > self.long_term_memory_ttl:
                    to_delete.append(memory.id)

        if to_delete:
            vector_db.remove_documents(to_delete)


# =============================================================================
# Configuration Dataclasses - Updated
# =============================================================================

@dataclasses.dataclass
class EmbeddingsConfig:
    """Configuration for generating embeddings."""
    sentence_transformer_model_path: str = "all-MiniLM-L6-v2"
    trust_remote_code: bool = False
    device: str = "cpu"
    embedding_kwargs: Dict[str, Any] = dataclasses.field(default_factory=dict)
    # Prefix configuration
    prefix_config: Optional[PrefixConfig] = None


@dataclasses.dataclass
class SemanticMemoryConfig:
    """Configuration for the semantic memory system."""
    persist: bool = True
    persist_directory: Optional[str] = "./memory"
    embeddings_config: EmbeddingsConfig = dataclasses.field(
        default_factory=EmbeddingsConfig
    )
    # Optional custom providers
    vector_database_provider: Optional[VectorDatabaseProvider] = None
    embedding_provider: Optional[EmbeddingProvider] = None
    reranking_provider: Optional[RerankingProvider] = None
    # Strategies
    extract_pattern_strategy: ExtractPatternStrategy = SimpleExtractPatternStrategy()
    cluster_embeddings_strategy: ClusterEmbeddingsStrategy = (
        SimpleClusterEmbeddingsStrategy()
    )
    cleanup_strategy: CleanupStrategy = None
    # Memory settings
    enable_long_term_memory: bool = True
    cleanup_interval_hours: float = 1.0
    decay_factor: float = 0.98
    query_result_multiplier: int = 8
    minimum_cluster_size: int = 4
    minimum_cluster_similarity: float = 0.875
    minimum_similarity_threshold: float = 0.70
    debug_mode: bool = False


# Example configurations with prefixes
def create_nomic_config() -> SemanticMemoryConfig:
    """Create a configuration for Nomic embeddings with GPU support."""
    prefix_config = PrefixConfig(
        store_prefix="search_document: ",
        recall_prefix="search_query: ",
        cluster_prefix="cluster: "
    )

    embeddings_config = EmbeddingsConfig(
        sentence_transformer_model_path="nomic-ai/nomic-embed-text-v1.5",
        trust_remote_code=True,
        device="cuda",
        prefix_config=prefix_config
    )

    return SemanticMemoryConfig(embeddings_config=embeddings_config)


# =============================================================================
# SemanticMemory Class - Refactored
# =============================================================================

class SemanticMemory:
    """
    A semantic memory system using abstract vector database interfaces.
    """

    def __init__(self, config: SemanticMemoryConfig = SemanticMemoryConfig()):
        """Initialize the semantic memory system with the provided configuration."""

        # Initialize embedding provider
        if config.embedding_provider:
            self.embedding_provider = config.embedding_provider
        else:
            self.embedding_provider = SentenceTransformerEmbeddingProvider(
                sentence_transformer_model_path=config.embeddings_config.sentence_transformer_model_path,
                trust_remote_code=config.embeddings_config.trust_remote_code,
                device=config.embeddings_config.device,
                prefix_config=config.embeddings_config.prefix_config,
                **config.embeddings_config.embedding_kwargs
            )

        # Initialize vector database provider
        if config.vector_database_provider:
            self.vector_db = config.vector_database_provider
        else:
            # Default to ChromaDB implementation
            self.vector_db = ChromaDbVectorDatabaseProvider(
                embedding_provider=self.embedding_provider,
                reranking_provider=config.reranking_provider,
                persistent_db_path=config.persist_directory,
                default_collection_name="working_memory",
                persistent=config.persist
            )

        # Store configuration
        self.enable_long_term_memory = config.enable_long_term_memory
        self.prefix_config = config.embeddings_config.prefix_config or PrefixConfig()
        self.embedding_kwargs = config.embeddings_config.embedding_kwargs
        self.minimum_similarity_threshold = config.minimum_similarity_threshold
        self.debug_mode = config.debug_mode

        self.decay_factor = config.decay_factor
        self.query_result_multiplier = config.query_result_multiplier
        self.minimum_cluster_size = config.minimum_cluster_size
        self.minimum_cluster_similarity = config.minimum_cluster_similarity

        # Bind strategies
        self._extract_pattern = config.extract_pattern_strategy.extract_pattern
        self._cluster_embeddings = config.cluster_embeddings_strategy.cluster_embeddings

        self.cleanup_strategy = config.cleanup_strategy
        self.last_cleanup = datetime.now()
        self.cleanup_interval = timedelta(hours=config.cleanup_interval_hours)

        # Create collections
        self.vector_db.create_or_set_current_collection("working_memory")
        if self.enable_long_term_memory:
            self.vector_db.create_or_set_current_collection("long_term_memory")
            # Switch back to working memory as default
            self.vector_db.create_or_set_current_collection("working_memory")

    def _maybe_cleanup(self, current_date: datetime = datetime.now()):
        """Run cleanup routines if the configured interval has passed."""
        if (
                not self.cleanup_strategy
                or current_date - self.last_cleanup < self.cleanup_interval
        ):
            return

        self.cleanup_strategy.cleanup_working_memory(self.vector_db, current_date)
        if self.enable_long_term_memory:
            self.cleanup_strategy.cleanup_long_term_memory(self.vector_db, current_date)
        self.last_cleanup = current_date

    def store(
            self,
            content: str,
            context: Optional[Dict] = None,
            timestamp=datetime.now().isoformat(),
    ) -> str:
        """Store a new memory with optional context metadata."""
        memory_id = f"mem_{uuid.uuid4()}"

        # Switch to working memory
        self.vector_db.create_or_set_current_collection("working_memory")

        # Prepare metadata
        metadata = {
            "timestamp": timestamp,
            "last_access_timestamp": timestamp,
            "access_count": 1,
            "memory_id": memory_id,
            "type": "memory",
        }
        if context:
            metadata.update(context)

        # Add to database (embedding will be generated automatically)
        self.vector_db.add_texts([content], [metadata])

        # Consolidate patterns if needed
        if self.enable_long_term_memory:
            self._consolidate_patterns(timestamp)

        # Cleanup if necessary
        self._maybe_cleanup(datetime.fromisoformat(timestamp))

        return memory_id

    def recall(
            self,
            query: str,
            n_results: int = 5,
            context_filter: Optional[Dict] = None,
            current_date: datetime = datetime.now(),
            alpha_recency=1,
            alpha_relevance=1,
            alpha_frequency=1,
    ) -> List[Dict]:
        """Recall memories similar to the query."""
        self._maybe_cleanup(current_date)

        def search_collection(collection_name: str, mem_type: str):
            """Search a specific collection."""
            self.vector_db.create_or_set_current_collection(collection_name)

            # Check if collection has documents
            doc_count = self.vector_db.get_document_count()
            if doc_count == 0:
                return []

            try:
                # Use the recall prefix for queries
                search_results = self.vector_db.query(
                    query=query,
                    query_filter=context_filter,
                    k=min(n_results * self.query_result_multiplier, doc_count)
                )

                # Format results
                formatted = []
                for i in range(len(search_results.ids)):
                    # Get full document details
                    doc_entry = self.vector_db.get_documents_by_ids(
                        [search_results.ids[i]]
                    )[0]

                    similarity = 1 - search_results.scores[i]  # Convert distance to similarity

                    formatted.append({
                        "content": search_results.chunks[i],
                        "metadata": doc_entry.metadata or {},
                        "similarity": similarity,
                        "memory_type": mem_type,
                    })

                return formatted

            except Exception as e:
                print(f"Warning: Error querying {mem_type} memory: {str(e)}")
                return []

        # Search both collections in parallel
        with ThreadPoolExecutor() as executor:
            if self.enable_long_term_memory:
                futures = {
                    executor.submit(search_collection, "working_memory", "working"),
                    executor.submit(search_collection, "long_term_memory", "long_term"),
                }
                results = []
                for future in futures:
                    results.extend(future.result())
            else:
                results = search_collection("working_memory", "working")

        # Deduplicate and rank results
        seen_contents = ""
        unique_results = []

        for result in results:
            if result["content"] not in seen_contents:
                if self.debug_mode:
                    print(f"Similarity: {result['similarity']}", flush=True)

                if result["similarity"] > self.minimum_similarity_threshold:
                    seen_contents += result["content"] + "\n"
                    result["rank_score"] = self.compute_memory_score(
                        result["metadata"],
                        result["similarity"],
                        current_date,
                        alpha_recency,
                        alpha_relevance,
                        alpha_frequency,
                    )

                    if self.debug_mode:
                        print(
                            f"RS: {result['rank_score']}, S: {result['similarity']}",
                            flush=True,
                        )
                    unique_results.append(result)

        # Sort and select top results
        unique_results.sort(key=lambda x: x["rank_score"], reverse=True)
        unique_results = unique_results[:n_results]

        # Update access metadata
        for result in unique_results:
            result["metadata"]["last_access_timestamp"] = current_date.isoformat()
            result["metadata"]["access_count"] = result["metadata"].get("access_count", 0) + 1

            # Update in appropriate collection
            if result["memory_type"] == "working":
                self.vector_db.create_or_set_current_collection("working_memory")
                doc_id = result["metadata"].get("memory_id")
            else:
                self.vector_db.create_or_set_current_collection("long_term_memory")
                doc_id = result["metadata"].get("pattern_id")

            if doc_id:
                self.vector_db.update_document_metadata(
                    doc_id, result["metadata"], merge=True
                )

        return unique_results

    def _consolidate_patterns(self, timestamp):
        """Consolidate clusters of similar memories into patterns."""
        self.vector_db.create_or_set_current_collection("working_memory")

        # Get all working memories
        working_memories = self.vector_db.get_all_documents(include_embeddings=True)

        if not working_memories:
            return

        # Extract embeddings and documents
        embeddings = [mem.embedding for mem in working_memories if mem.embedding is not None]
        documents = [mem.content for mem in working_memories]
        metadatas = [mem.metadata or {} for mem in working_memories]

        if not embeddings:
            return

        # Cluster embeddings
        clusters = self._cluster_embeddings(embeddings, self.minimum_cluster_similarity)

        # Process each cluster
        for cluster_idx, cluster in enumerate(clusters):
            if len(cluster) < self.minimum_cluster_size:
                continue

            pattern_id = f"pattern_{uuid.uuid4()}"

            # Extract pattern from cluster
            pattern = self._extract_pattern(
                pattern_id,
                [documents[i] for i in cluster],
                [metadatas[i] for i in cluster],
                timestamp,
            )

            # Delete consolidated memories from working memory
            ids_to_delete = [
                metadatas[i].get("memory_id") for i in cluster
                if metadatas[i].get("memory_id")
            ]
            if ids_to_delete:
                self.vector_db.remove_documents(ids_to_delete)

            # Add pattern to long-term memory
            self.vector_db.create_or_set_current_collection("long_term_memory")

            # Check for duplicates
            search_results = self.vector_db.query(
                query=pattern["content"],
                k=1
            )

            if search_results.scores and search_results.scores[0] <= 0.01:
                continue  # Skip duplicate pattern

            # Add new pattern
            self.vector_db.add_texts(
                [pattern["content"]],
                [pattern["metadata"]]
            )

        # Switch back to working memory
        self.vector_db.create_or_set_current_collection("working_memory")

    def compute_memory_score(
            self, metadata, relevance, date, alpha_recency, alpha_relevance, alpha_frequency
    ):
        """Compute a composite score for a memory."""
        recency = self.compute_recency(metadata, date)
        frequency = np.log1p(metadata.get("access_count", 1))

        return (
                alpha_recency * recency
                + alpha_relevance * relevance
                + alpha_frequency * frequency
        )

    def compute_recency(self, metadata, date):
        """Compute the recency score for a memory."""
        last_access = metadata.get("last_access_timestamp")
        if not last_access:
            return 0

        time_diff = date - datetime.fromisoformat(last_access)
        hours_diff = time_diff.total_seconds() / 3600
        recency = self.decay_factor ** hours_diff
        return recency

    def get_stats(self) -> Dict:
        """Get statistics about the current state of the memory system."""
        self.vector_db.create_or_set_current_collection("working_memory")
        working_count = self.vector_db.get_document_count()

        if self.enable_long_term_memory:
            self.vector_db.create_or_set_current_collection("long_term_memory")
            long_term_count = self.vector_db.get_document_count()
        else:
            long_term_count = 0

        # Reset to working memory
        self.vector_db.create_or_set_current_collection("working_memory")

        return {
            "working_count": working_count,
            "long_term_count": long_term_count,
        }

    def clear_all_memories(self):
        """Clear all memories from both working and long-term collections."""
        self.vector_db.clear_collection("working_memory")
        if self.enable_long_term_memory:
            self.vector_db.clear_collection("long_term_memory")

    def export_memories(self, include_embeddings: bool = False) -> Dict[str, Any]:
        """Export all memories for backup."""
        export_data = {}

        export_data["working_memory"] = self.vector_db.export_collection(
            "working_memory", include_embeddings
        )

        if self.enable_long_term_memory:
            export_data["long_term_memory"] = self.vector_db.export_collection(
                "long_term_memory", include_embeddings
            )

        export_data["config"] = {
            "minimum_cluster_size": self.minimum_cluster_size,
            "minimum_cluster_similarity": self.minimum_cluster_similarity,
            "minimum_similarity_threshold": self.minimum_similarity_threshold,
            "decay_factor": self.decay_factor,
        }

        return export_data

    def import_memories(self, data: Dict[str, Any], overwrite: bool = False) -> bool:
        """Import memories from a backup."""
        success = True

        if "working_memory" in data:
            success &= self.vector_db.import_collection(
                data["working_memory"], "working_memory", overwrite
            )

        if "long_term_memory" in data and self.enable_long_term_memory:
            success &= self.vector_db.import_collection(
                data["long_term_memory"], "long_term_memory", overwrite
            )

        return success