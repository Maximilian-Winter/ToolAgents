
import abc
import dataclasses
import re
import uuid
import json
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor
from typing import List, Dict, Optional, Any, Mapping

import chromadb
from chromadb.api.types import IncludeEnum

import numpy as np
from torch import Tensor

from ToolAgents.messages import ChatMessage, MessageTemplate
from ToolAgents.provider.llm_provider import ProviderSettings
from ToolAgents.agents.base_llm_agent import BaseToolAgent



# =============================================================================
# Extraction Pattern Strategies
# =============================================================================

class ExtractPatternStrategy(abc.ABC):
    """
    Abstract base class for extraction pattern strategies.

    An extraction pattern strategy defines how to extract a summarized or
    aggregated pattern from a group of similar memories.
    """

    @abc.abstractmethod
    def extract_pattern(self, pattern_id, documents: List[str],
                        metadatas: list[Mapping[str, str | int | float | bool]],
                        timestamp=datetime.now().isoformat()) -> Dict:
        """
        Extract a pattern from the given documents and metadata.

        Args:
            pattern_id: Unique identifier for the pattern.
            documents: List of document strings to be consolidated.
            metadatas: List of metadata dictionaries corresponding to the documents.
            timestamp: The timestamp for the pattern extraction (default: now).

        Returns:
            A dictionary containing the consolidated content and its metadata.
        """
        pass


class SimpleExtractPatternStrategy(ExtractPatternStrategy):
    """
    A simple extraction strategy that aggregates documents by concatenation
    and merges metadata by summing access counts and combining timestamps.
    """

    def extract_pattern(self, pattern_id, documents: List[str],
                        metadatas: List[Dict],
                        timestamp=datetime.now().isoformat()) -> Dict:
        """Extract pattern from a cluster of similar memories by simple concatenation."""
        # Extract the 'access_count' from each metadata entry
        counters = [m['access_count'] for m in metadatas]

        # Prepare metadata for the extracted pattern
        pattern_metadata = {
            "type": "pattern",
            "timestamp": timestamp,
            "pattern_id": pattern_id,
            "last_access_timestamp": timestamp,
            "access_count": sum(counters),
            "source_count": len(documents),
            # Convert source timestamps list into a JSON string
            "source_timestamps": json.dumps([m['timestamp'] for m in metadatas])
        }

        # Return the consolidated content and metadata
        return {
            "content": '\n'.join(documents),
            "metadata": pattern_metadata
        }


sum_chat_turns_template = MessageTemplate.from_string("""You will be analyzing a collection of chat turns to extract information about {USER_NAME} and their relationship with {ASSISTANT_NAME}. Here are the chat turns:

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

Remember to base your analysis solely on the information provided in the chat turns. Do not make assumptions or include information that is not directly stated or strongly implied in the conversation.""")
class SummarizationExtractPatternStrategy(ExtractPatternStrategy):
    """
    An extraction strategy that uses a language model (via a provided agent)
    to summarize and extract a pattern from multiple documents.
    """

    def __init__(self, agent: BaseToolAgent, summarizer_settings: ProviderSettings,
                 user_name: str, assistant_name: str, chat_turn_summary: MessageTemplate = sum_chat_turns_template,
                 debug_mode: bool = False):
        """
        Initialize the summarization extraction strategy.

        Args:
            agent: The language model agent used to generate summaries.
            summarizer_settings: Sampling settings for the language model.
            user_name: The name of the user
            assistant_name: The name of the assistant.
            debug_mode: If True, print debug information.
        """
        self.agent = agent
        self.user_name = user_name
        self.assistant_name = assistant_name
        self.debug_mode = debug_mode
        self.summarizer_settings = summarizer_settings
        self.chat_turn_summary = chat_turn_summary

    def extract_pattern(self, pattern_id, documents: List[str],
                        metadatas: List[Dict],
                        timestamp=datetime.now().isoformat()) -> Dict:
        """Extract pattern from a cluster of similar memories using a language model for summarization."""
        # Sum the access counts from all provided metadata
        counters = [m['access_count'] for m in metadatas]
        pattern_metadata = {
            "type": "pattern",
            "timestamp": timestamp,
            "pattern_id": pattern_id,
            "last_access_timestamp": timestamp,
            "access_count": sum(counters),
            "source_count": len(documents),
            "source_timestamps": json.dumps([m['timestamp'] for m in metadatas])
        }
        prompt = self.chat_turn_summary.generate_message_content(CHAT_TURNS="\n\n---\n\n".join(documents), USER_NAME=self.user_name, ASSISTANT_NAME=self.assistant_name)
        if self.debug_mode:
            print(prompt)
        # Use the language model agent to generate a summary based on the prompt
        result = self.agent.get_response(
            messages=[ChatMessage.create_user_message(prompt)
            ], settings=self.summarizer_settings
        )
        match = re.findall(r'<memory_summary>(.*?)</memory_summary>', result.response, re.DOTALL)
        patterns = []
        for content in match:
            patterns.append(content.replace("**", ""))
            if self.debug_mode:
                print(content.replace("**", ""), flush=True)
        return {
            "content": '\n'.join(patterns),
            "metadata": pattern_metadata
        }


# =============================================================================
# Embedding Clustering Strategies
# =============================================================================

class ClusterEmbeddingsStrategy(abc.ABC):
    """
    Abstract base class for clustering embeddings.

    A clustering strategy defines how to group similar embeddings into clusters.
    """

    @abc.abstractmethod
    def cluster_embeddings(self, embeddings: List[np.ndarray], minimum_cluster_similarity: float = 0.75) -> List[
        List[int]]:
        """
        Cluster the provided embeddings based on a similarity threshold.

        Args:
            embeddings: List of embedding vectors.
            minimum_cluster_similarity: Minimum cosine similarity to consider embeddings part of the same cluster.

        Returns:
            A list of clusters, where each cluster is a list of indices of embeddings.
        """
        pass


class SimpleClusterEmbeddingsStrategy(ClusterEmbeddingsStrategy):
    """
    A simple clustering strategy that uses cosine similarity to group embeddings.
    """

    def cluster_embeddings(self, embeddings: List[np.ndarray | Tensor],
                           minimum_cluster_similarity: float = 0.75) -> List[List[int]]:
        """Cluster embeddings using cosine similarity."""
        # Stack embeddings into a numpy array for vectorized operations
        embeddings_array = np.stack(embeddings)

        # Compute the norm for each embedding and avoid division by zero
        norms = np.linalg.norm(embeddings_array, axis=1)
        norms[norms == 0] = 1  # Avoid division by zero for zero vectors

        # Normalize the embeddings
        normalized = embeddings_array / norms[:, np.newaxis]
        # Compute the cosine similarity matrix between all pairs of embeddings
        similarity_matrix = np.dot(normalized, normalized.T)

        clusters = []
        used_indices = set()

        # Iterate over embeddings to form clusters based on similarity threshold
        for i in range(len(embeddings)):
            if i in used_indices:
                continue

            cluster = [i]
            used_indices.add(i)

            # Check subsequent embeddings for similarity
            for j in range(i + 1, len(embeddings)):
                if j not in used_indices and similarity_matrix[i, j] > minimum_cluster_similarity:
                    cluster.append(j)
                    used_indices.add(j)

            clusters.append(cluster)

        return clusters


# =============================================================================
# Cleanup Strategies
# =============================================================================

class CleanupStrategy(abc.ABC):
    """
    Abstract base class for memory cleanup strategies.

    A cleanup strategy defines how to remove outdated or less frequently used memories.
    """

    @abc.abstractmethod
    def cleanup_working_memory(self, collection: chromadb.Collection,
                               current_date: datetime) -> None:
        """
        Clean up the working memory based on time and access counts.
        """
        pass

    @abc.abstractmethod
    def cleanup_long_term_memory(self, collection: chromadb.Collection,
                                 current_date: datetime) -> None:
        """
        Clean up the long-term memory based on time since last access.
        """
        pass


class TimeBasedCleanupStrategy(CleanupStrategy):
    """
    A cleanup strategy that removes memories based on their age and access frequency.
    """

    def __init__(self,
                 working_memory_ttl_hours: float = 24.0,
                 long_term_memory_ttl_days: float = 30.0,
                 min_access_count: int = 3):
        """
        Initialize the time-based cleanup strategy.

        Args:
            working_memory_ttl_hours: Time-to-live in hours for working memory entries.
            long_term_memory_ttl_days: Time-to-live in days for long-term memory entries.
            min_access_count: Minimum number of accesses required to keep a memory.
        """
        self.working_memory_ttl = timedelta(hours=working_memory_ttl_hours)
        self.long_term_memory_ttl = timedelta(days=long_term_memory_ttl_days)
        self.min_access_count = min_access_count

    def cleanup_working_memory(self, collection: chromadb.Collection,
                               current_date: datetime) -> None:
        """Remove old working memories with low access counts."""
        # Retrieve all memories along with their metadata
        memories = collection.get(include=[IncludeEnum.metadatas])

        if not memories['metadatas']:
            return

        to_delete = []
        # Iterate over memories to determine which should be deleted
        for memory_id, metadata in zip(memories['ids'], memories['metadatas']):
            last_access = datetime.fromisoformat(metadata['last_access_timestamp'])
            age = current_date - last_access

            # Delete if the memory is older than the TTL and has low access frequency
            if (age > self.working_memory_ttl and
                    metadata['access_count'] < self.min_access_count):
                to_delete.append(memory_id)

        if to_delete:
            collection.delete(ids=to_delete)

    def cleanup_long_term_memory(self, collection: chromadb.Collection,
                                 current_date: datetime) -> None:
        """Remove very old long-term memories that haven't been accessed recently."""
        memories = collection.get(include=[IncludeEnum.metadatas])

        if not memories['metadatas']:
            return

        to_delete = []
        for memory_id, metadata in zip(memories['ids'], memories['metadatas']):
            last_access = datetime.fromisoformat(metadata['last_access_timestamp'])
            age = current_date - last_access

            # Delete if the memory's age exceeds the long-term TTL
            if age > self.long_term_memory_ttl:
                to_delete.append(memory_id)

        if to_delete:
            collection.delete(ids=to_delete)


# =============================================================================
# Configuration Dataclasses
# =============================================================================

@dataclasses.dataclass
class EmbeddingsConfig:
    """
    Configuration for generating embeddings.

    Attributes:
        sentence_transformer_model_path: Path or name of the sentence transformer model.
        trust_remote_code: Whether to trust remote code from the model repository.
        device: Device to use for computation ('cpu' or 'cuda').
        embedding_kwargs: Additional keyword arguments for embedding generation.
        embeddings_store_prefix: Optional prefix to add when storing embeddings.
        embeddings_recall_prefix: Optional prefix to add when recalling embeddings.
        embeddings_clusters_prefix: Optional prefix to add when clustering embeddings.
    """
    sentence_transformer_model_path: str = "all-MiniLM-L6-v2"
    trust_remote_code: bool = False
    device: str = "cpu"
    embedding_kwargs: Dict[str, Any] = dataclasses.field(default_factory=dict)
    embeddings_store_prefix: str = None
    embeddings_recall_prefix: str = None
    embeddings_clusters_prefix: str = None


@dataclasses.dataclass
class SemanticMemoryConfig:
    """
    Configuration for the semantic memory system.

    Attributes:
        persist: Whether to persist memory to disk.
        persist_directory: Directory path for persisting memory.
        embeddings_config: Embeddings configuration object.
        extract_pattern_strategy: Strategy for extracting patterns.
        cluster_embeddings_strategy: Strategy for clustering embeddings.
        cleanup_strategy: Strategy for cleaning up memories.
        enable_long_term_memory: Whether to enable long-term memory.
        cleanup_interval_hours: How often to run cleanup (in hours).
        decay_factor: Decay factor for recency scoring.
        query_result_multiplier: Multiplier to adjust number of query results.
        minimum_cluster_size: Minimum number of memories to form a cluster.
        minimum_cluster_similarity: Minimum similarity to group memories in a cluster.
        minimum_similarity_threshold: Minimum similarity threshold to consider memories relevant.
        debug_mode: Whether to enable debug logging.
    """
    persist: bool = True
    persist_directory: Optional[str] = "./memory"
    embeddings_config: EmbeddingsConfig = dataclasses.field(default_factory=EmbeddingsConfig)
    extract_pattern_strategy: ExtractPatternStrategy = SimpleExtractPatternStrategy()
    cluster_embeddings_strategy: ClusterEmbeddingsStrategy = SimpleClusterEmbeddingsStrategy()
    cleanup_strategy: CleanupStrategy = None
    enable_long_term_memory: bool = True
    cleanup_interval_hours: float = 1.0  # How often to run cleanup
    decay_factor: float = 0.98
    query_result_multiplier = 8
    minimum_cluster_size: int = 4
    minimum_cluster_similarity: float = 0.875
    minimum_similarity_threshold: float = 0.70
    debug_mode: bool = False


# Example specialized embeddings configuration (e.g., for GPU-based processing)
nomic_text_embeddings_gpu_config = EmbeddingsConfig(
    sentence_transformer_model_path="nomic-ai/nomic-embed-text-v1.5",
    embeddings_store_prefix="search_document: ",
    embeddings_recall_prefix="search_query: ",
    embeddings_clusters_prefix="cluster: ",
    trust_remote_code=True,
    device="cuda"
)
semantic_memory_nomic_text_gpu_config = SemanticMemoryConfig(embeddings_config=nomic_text_embeddings_gpu_config)


# =============================================================================
# SemanticMemory Class
# =============================================================================

class SemanticMemory:
    """
    A semantic memory system that supports storing, recalling, and consolidating memories.

    This class uses a sentence transformer encoder to generate embeddings for text,
    stores memories in working and long-term collections, and consolidates patterns
    based on clustering strategies.
    """

    def __init__(self, config: SemanticMemoryConfig = SemanticMemoryConfig()):
        """
        Initialize the semantic memory system with the provided configuration.

        Args:
            config: An instance of SemanticMemoryConfig defining system parameters.
        """
        from sentence_transformers import SentenceTransformer
        # Initialize the sentence transformer encoder using the provided model configuration
        self.encoder = SentenceTransformer(
            config.embeddings_config.sentence_transformer_model_path,
            trust_remote_code=config.embeddings_config.trust_remote_code,
            device=config.embeddings_config.device
        )

        self.enable_long_term_memory = config.enable_long_term_memory
        self.embeddings_store_prefix = config.embeddings_config.embeddings_store_prefix
        self.embeddings_recall_prefix = config.embeddings_config.embeddings_recall_prefix
        self.embedding_kwargs = config.embeddings_config.embedding_kwargs
        self.embeddings_clusters_prefix = config.embeddings_config.embeddings_clusters_prefix
        self.minimum_similarity_threshold = config.minimum_similarity_threshold
        self.debug_mode = config.debug_mode

        # Initialize the database client based on persistence setting
        if config.persist:
            self.client = chromadb.PersistentClient(path=config.persist_directory)
        else:
            self.client = chromadb.EphemeralClient()

        self.decay_factor = config.decay_factor
        self.query_result_multiplier = config.query_result_multiplier
        self.minimum_cluster_size = config.minimum_cluster_size
        self.minimum_cluster_similarity = config.minimum_cluster_similarity

        # Bind the extraction and clustering strategies for later use
        self._extract_pattern = config.extract_pattern_strategy.extract_pattern
        self._cluster_embeddings = config.cluster_embeddings_strategy.cluster_embeddings

        self.cleanup_strategy = config.cleanup_strategy
        self.last_cleanup = datetime.now()
        self.cleanup_interval = timedelta(hours=config.cleanup_interval_hours)

        # Create or get the working memory collection with cosine similarity as the metric
        self.working = self.client.get_or_create_collection(
            name="working_memory",
            metadata={"hnsw:space": "cosine"}
        )
        # Create or get the long-term memory collection if enabled
        if self.enable_long_term_memory:
            self.long_term = self.client.get_or_create_collection(
                name="long_term_memory",
                metadata={"hnsw:space": "cosine"}
            )
        else:
            self.long_term = None

    def _maybe_cleanup(self, current_date: datetime = datetime.now()):
        """
        Run cleanup routines if the configured cleanup interval has passed.

        Args:
            current_date: The current date/time used to compare against the last cleanup.
        """
        if not self.cleanup_strategy or current_date - self.last_cleanup < self.cleanup_interval:
            return

        # Clean up working and long-term memories
        self.cleanup_strategy.cleanup_working_memory(self.working, current_date)
        self.cleanup_strategy.cleanup_long_term_memory(self.long_term, current_date)
        self.last_cleanup = current_date

    def store(self, content: str, context: Optional[Dict] = None,
              timestamp=datetime.now().isoformat()) -> str:
        """
        Store a new memory with optional context metadata.

        Args:
            content: The text content to store.
            context: Additional contextual metadata to store along with the content.
            timestamp: The timestamp of storage (default: current time in ISO format).

        Returns:
            A unique memory identifier for the stored memory.
        """
        # Generate a unique memory ID
        memory_id = f"mem_{uuid.uuid4()}"

        # Compute the embedding; optionally prepend a store prefix if configured
        if self.embeddings_store_prefix:
            embedding = self.encoder.encode(self.embeddings_store_prefix + content, **self.embedding_kwargs).tolist()
        else:
            embedding = self.encoder.encode(content, **self.embedding_kwargs).tolist()

        # Prepare metadata for the memory entry
        metadata = {
            "timestamp": timestamp,
            "last_access_timestamp": timestamp,
            "access_count": 1,
            "memory_id": memory_id,
            "type": "memory"
        }
        # Update metadata with any additional context provided
        if context:
            metadata.update(context)

        # Add the new memory to the working memory collection
        self.working.add(
            documents=[content],
            metadatas=[metadata],
            embeddings=[embedding],
            ids=[memory_id]
        )
        # If long-term memory is enabled and enough memories are available, consolidate patterns
        if self.enable_long_term_memory:
            self._consolidate_patterns(timestamp)
        # Perform cleanup if necessary
        self._maybe_cleanup(datetime.fromisoformat(timestamp))
        return memory_id

    def recall(self, query: str, n_results: int = 5, context_filter: Optional[Dict] = None,
               current_date: datetime = datetime.now(), alpha_recency=1, alpha_relevance=1, alpha_frequency=1) -> List[
        Dict]:
        """
        Recall memories that are similar to the query by performing a parallel search.

        Args:
            query: The query string used to search for similar memories.
            n_results: The number of top results to return.
            context_filter: Optional filter criteria for the memory search.
            current_date: The current date/time for scoring purposes.
            alpha_recency: Weight for recency in scoring.
            alpha_relevance: Weight for similarity score in scoring.
            alpha_frequency: Weight for access frequency in scoring.

        Returns:
            A list of memory dictionaries containing content, metadata, similarity score, and rank score.
        """
        # Run cleanup if needed before performing the recall
        self._maybe_cleanup(current_date)

        # Compute the query embedding; optionally add a recall prefix if configured
        if self.embeddings_recall_prefix:
            query_embedding = self.encoder.encode(self.embeddings_recall_prefix + query,
                                                  **self.embedding_kwargs).tolist()
        else:
            query_embedding = self.encoder.encode(query, **self.embedding_kwargs).tolist()

        def search(collection: chromadb.Collection, mem_type):
            """
            Helper function to query a collection and format its results.

            Args:
                collection: The memory collection to query.
                mem_type: A label indicating whether it's 'working' or 'long_term' memory.

            Returns:
                A list of formatted results from the query.
            """
            if collection.count() == 0:
                return []
            try:
                # Query the collection with the query embedding and an adjusted number of results
                layer_results = collection.query(
                    query_embeddings=[query_embedding],
                    n_results=min(n_results * self.query_result_multiplier, collection.count()),
                    where=context_filter if context_filter and len(context_filter) > 0 else None
                )
                return self._format_results(layer_results, mem_type)
            except Exception as e:
                print(f"Warning: Error querying {mem_type} memory: {str(e)}")
                return []

        # Use a thread pool to query both working and long-term memories in parallel
        with ThreadPoolExecutor() as executor:
            if self.enable_long_term_memory:
                futures = {
                    executor.submit(search, self.working, "working"),
                    executor.submit(search, self.long_term, "long_term")
                }
                results = []
                for future in futures:
                    results.extend(future.result())
            else:
                results = search(self.working, "working")

        # Deduplicate and sort results based on a computed rank score
        seen_contents = ""
        unique_results = []
        for result in results:
            if result['content'] not in seen_contents:
                if self.debug_mode:
                    print(f"Similarity: {result['similarity']}", flush=True)
                if result['similarity'] > self.minimum_similarity_threshold:
                    seen_contents += result['content'] + '\n'
                    # Compute a rank score based on recency, relevance, and frequency
                    result["rank_score"] = self.compute_memory_score(
                        result["metadata"],
                        result["similarity"],
                        current_date,
                        alpha_recency,
                        alpha_relevance,
                        alpha_frequency
                    )
                    if self.debug_mode:
                        print(f"RS: {result['rank_score']}, S: {result['similarity']}", flush=True)
                    unique_results.append(result)

        # Sort the unique results in descending order of rank score and select the top n_results
        unique_results.sort(key=lambda x: x['rank_score'], reverse=True)
        unique_results = unique_results[:n_results]

        # Update metadata for each recalled memory to reflect the access
        for unique_result in unique_results:
            unique_result["metadata"]["last_access_timestamp"] = current_date.isoformat()
            unique_result["metadata"]["access_count"] = unique_result["metadata"]["access_count"] + 1
            if unique_result["metadata"]["type"] == "working":
                self.working.update([unique_result["metadata"]["memory_id"]], metadatas=[unique_result["metadata"]])
            elif unique_result["metadata"]["type"] == "long_term":
                self.long_term.update([unique_result["metadata"]["pattern_id"]], metadatas=[unique_result["metadata"]])

        return unique_results

    def _consolidate_patterns(self, timestamp):
        """
        Consolidate clusters of similar memories into patterns.

        This method uses the clustering strategy to group embeddings from the working memory.
        For each sufficiently large cluster, it extracts a pattern and moves it to long-term memory.
        """
        # Depending on the configuration, either use the clusters prefix or not when generating embeddings
        if self.embeddings_clusters_prefix is None:
            working_memories = self.working.get(
                include=[IncludeEnum.metadatas, IncludeEnum.embeddings, IncludeEnum.documents])
            if not working_memories['documents']:
                return
            # Directly use the stored embeddings
            embeddings = [embed for embed in working_memories['embeddings']]
        else:
            working_memories = self.working.get(include=[IncludeEnum.metadatas, IncludeEnum.documents])
            # Compute embeddings with the clusters prefix
            embeddings = [self.encoder.encode(self.embeddings_clusters_prefix + doc, **self.embedding_kwargs)
                          for doc in working_memories['documents']]
        # Cluster the embeddings based on the similarity threshold
        clusters = self._cluster_embeddings(embeddings, self.minimum_cluster_similarity)

        # Process each cluster: if it's large enough, extract a pattern and store it in long-term memory
        for cluster_idx, cluster in enumerate(clusters):
            if len(cluster) < self.minimum_cluster_size:  # Skip clusters that are too small
                continue

            pattern_id = f"pattern_{uuid.uuid4()}"

            # Extract a pattern from the cluster using the configured extraction strategy
            pattern = self._extract_pattern(
                pattern_id,
                [working_memories['documents'][i] for i in cluster],
                [working_memories['metadatas'][i] for i in cluster],
                timestamp
            )

            # Compute an embedding for the consolidated pattern
            if self.embeddings_store_prefix:
                pattern_embedding = self.encoder.encode(self.embeddings_store_prefix + pattern['content']).tolist()
            else:
                pattern_embedding = self.encoder.encode(pattern['content']).tolist()

            # Delete the individual memories that were consolidated into a pattern
            self.working.delete(ids=[t["memory_id"] for t in [working_memories['metadatas'][i] for i in cluster]])
            # Check for duplicate patterns in long-term memory
            existing = self.long_term.query(
                query_embeddings=[pattern_embedding],
                n_results=1
            )
            # If a nearly identical pattern already exists, skip adding the duplicate
            if existing['distances'] and len(existing['distances'][0]) > 0 and existing['distances'][0][0] <= 0.01:
                continue

            # Add the new pattern to long-term memory
            self.long_term.add(
                documents=[pattern['content']],
                metadatas=[pattern['metadata']],
                embeddings=[pattern_embedding],
                ids=[pattern_id]
            )

    def _format_results(self, results: Dict, memory_type: str) -> List[Dict]:
        """
        Format the raw query results into a more friendly structure.

        Args:
            results: Raw results from a chromadb query.
            memory_type: A label to indicate whether the result is from working or long-term memory.

        Returns:
            A list of dictionaries containing the content, metadata, similarity, and memory type.
        """
        formatted = []

        if not results['documents']:
            return formatted

        # Extract the first (and only) set of documents, metadata, and distances from the query results
        documents = results['documents'][0]
        metadatas = results['metadatas'][0]
        distances = results['distances'][0]

        for doc, meta, dist in zip(documents, metadatas, distances):
            # Convert distance to similarity score (ensuring it remains between 0 and 1)
            similarity = max(0, min(1, 1 - dist))
            if similarity > 0:  # Only include results with meaningful similarity
                formatted.append({
                    'content': doc,
                    'metadata': meta,
                    'similarity': similarity,
                    'memory_type': memory_type
                })

        return formatted

    def compute_memory_score(self, metadata, relevance, date, alpha_recency, alpha_relevance, alpha_frequency):
        """
        Compute a composite score for a memory based on recency, relevance, and frequency.

        Args:
            metadata: The metadata dictionary for the memory.
            relevance: The similarity score between the query and the memory.
            date: The current date/time.
            alpha_recency: Weight for the recency component.
            alpha_relevance: Weight for the relevance component.
            alpha_frequency: Weight for the frequency component.

        Returns:
            A floating point score representing the memory's rank.
        """
        # Compute the recency score (which decays over time)
        recency = self.compute_recency(metadata, date)
        # Compute the frequency score using logarithmic scaling to reduce the impact of very high counts
        frequency = np.log1p(metadata["access_count"])
        return (
                alpha_recency * recency +  # Contribution from recency
                alpha_relevance * relevance +  # Contribution from similarity
                alpha_frequency * frequency  # Contribution from frequency of access
        )

    def compute_recency(self, metadata, date):
        """
        Compute the recency score for a memory.

        This score decays exponentially with the number of hours passed since the last access.

        Args:
            metadata: The metadata dictionary for the memory.
            date: The current date/time.

        Returns:
            A floating point number representing the recency score.
        """
        time_diff = date - datetime.fromisoformat(metadata["last_access_timestamp"])
        hours_diff = time_diff.total_seconds() / 3600
        recency = self.decay_factor ** hours_diff
        return recency

    def get_stats(self) -> Dict:
        """
        Get statistics about the current state of the memory system.

        Returns:
            A dictionary with counts for working and long-term memories.
        """
        return {
            "working_count": self.working.count(),
            "long_term_count": self.long_term.count()
        }
