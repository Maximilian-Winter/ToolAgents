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
from ToolAgents.knowledge.vector_database import (
    VectorDatabaseProvider,
    EmbeddingProvider,
    EmbeddingTask,
    VectorSearchResult,
    VectorCollection
)


class ExtractPatternStrategy(abc.ABC):
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


class ClusterEmbeddingsStrategy(abc.ABC):
    @abc.abstractmethod
    def cluster_embeddings(
            self, embeddings: List[np.ndarray], minimum_cluster_similarity: float = 0.75
    ) -> List[List[int]]:
        pass


class SimpleClusterEmbeddingsStrategy(ClusterEmbeddingsStrategy):
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


class CleanupStrategy(abc.ABC):
    @abc.abstractmethod
    def cleanup_working_memory(
            self, vector_db: VectorDatabaseProvider, current_date: datetime
    ) -> None:
        pass

    @abc.abstractmethod
    def cleanup_long_term_memory(
            self, vector_db: VectorDatabaseProvider, current_date: datetime
    ) -> None:
        pass


class TimeBasedCleanupStrategy(CleanupStrategy):
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
        memories = vector_db.get_all_entries()

        if not memories.metadata:
            return

        to_delete = []
        for memory_id, metadata in zip(memories.ids, memories.metadata):
            if metadata.get("type") != "memory":
                continue

            last_access = datetime.fromisoformat(metadata["last_access_timestamp"])
            age = current_date - last_access

            if (
                    age > self.working_memory_ttl
                    and metadata["access_count"] < self.min_access_count
            ):
                to_delete.append(memory_id)

        if to_delete:
            vector_db.remove_by_ids(to_delete)

    def cleanup_long_term_memory(
            self, vector_db: VectorDatabaseProvider, current_date: datetime
    ) -> None:
        memories = vector_db.get_all_entries()

        if not memories.metadata:
            return

        to_delete = []
        for memory_id, metadata in zip(memories.ids, memories.metadata):
            if metadata.get("type") != "pattern":
                continue

            last_access = datetime.fromisoformat(metadata["last_access_timestamp"])
            age = current_date - last_access

            if age > self.long_term_memory_ttl:
                to_delete.append(memory_id)

        if to_delete:
            vector_db.remove_by_ids(to_delete)


@dataclasses.dataclass
class SemanticMemoryConfig:
    extract_pattern_strategy: ExtractPatternStrategy = SimpleExtractPatternStrategy()
    cluster_embeddings_strategy: ClusterEmbeddingsStrategy = SimpleClusterEmbeddingsStrategy()
    cleanup_strategy: CleanupStrategy = None
    enable_long_term_memory: bool = True
    cleanup_interval_hours: float = 1.0
    decay_factor: float = 0.98
    query_result_multiplier = 8
    minimum_cluster_size: int = 4
    minimum_cluster_similarity: float = 0.875
    minimum_similarity_threshold: float = 0.70
    debug_mode: bool = False


class SemanticMemory:
    def __init__(
            self,
            vector_db_provider: VectorDatabaseProvider,
            embedding_provider: EmbeddingProvider,
            config: SemanticMemoryConfig = SemanticMemoryConfig()
    ):
        self.vector_db = vector_db_provider
        self.embedding_provider = embedding_provider
        self.config = config

        self.enable_long_term_memory = config.enable_long_term_memory
        self.minimum_similarity_threshold = config.minimum_similarity_threshold
        self.debug_mode = config.debug_mode
        self.decay_factor = config.decay_factor
        self.query_result_multiplier = config.query_result_multiplier
        self.minimum_cluster_size = config.minimum_cluster_size
        self.minimum_cluster_similarity = config.minimum_cluster_similarity

        self._extract_pattern = config.extract_pattern_strategy.extract_pattern
        self._cluster_embeddings = config.cluster_embeddings_strategy.cluster_embeddings

        self.cleanup_strategy = config.cleanup_strategy
        self.last_cleanup = datetime.now()
        self.cleanup_interval = timedelta(hours=config.cleanup_interval_hours)

        self.vector_db.create_or_set_current_collection("working_memory")
        self.working_collection = "working_memory"

        if self.enable_long_term_memory:
            self.vector_db.create_or_set_current_collection("long_term_memory")
            self.long_term_collection = "long_term_memory"
        else:
            self.long_term_collection = None

    def _maybe_cleanup(self, current_date: datetime = datetime.now()):
        if (
                not self.cleanup_strategy
                or current_date - self.last_cleanup < self.cleanup_interval
        ):
            return

        self.vector_db.create_or_set_current_collection(self.working_collection)
        self.cleanup_strategy.cleanup_working_memory(self.vector_db, current_date)

        if self.long_term_collection:
            self.vector_db.create_or_set_current_collection(self.long_term_collection)
            self.cleanup_strategy.cleanup_long_term_memory(self.vector_db, current_date)

        self.last_cleanup = current_date

    def store(
            self,
            content: str,
            context: Optional[Dict] = None,
            timestamp=datetime.now().isoformat(),
    ) -> str:
        memory_id = f"mem_{uuid.uuid4()}"

        metadata = {
            "timestamp": timestamp,
            "last_access_timestamp": timestamp,
            "access_count": 1,
            "memory_id": memory_id,
            "type": "memory",
        }

        if context:
            metadata.update(context)

        self.vector_db.create_or_set_current_collection(self.working_collection)
        self.vector_db.add_texts_with_id(
            ids=[memory_id],
            texts=[content],
            metadata=[metadata]
        )

        if self.enable_long_term_memory:
            self._consolidate_patterns(timestamp)

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
        self._maybe_cleanup(current_date)

        def search(collection_name: str, mem_type: str):
            self.vector_db.create_or_set_current_collection(collection_name)
            all_entries = self.vector_db.get_all_entries()

            if not all_entries.ids:
                return []

            try:
                results = self.vector_db.query(
                    query=query,
                    query_filter=context_filter,
                    k=min(n_results * self.query_result_multiplier, len(all_entries.ids))
                )
                return self._format_results(results, all_entries, mem_type)
            except Exception as e:
                print(f"Warning: Error querying {mem_type} memory: {str(e)}")
                return []

        with ThreadPoolExecutor() as executor:
            if self.enable_long_term_memory:
                futures = {
                    executor.submit(search, self.working_collection, "working"),
                    executor.submit(search, self.long_term_collection, "long_term"),
                }
                results = []
                for future in futures:
                    results.extend(future.result())
            else:
                results = search(self.working_collection, "working")

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

        unique_results.sort(key=lambda x: x["rank_score"], reverse=True)
        unique_results = unique_results[:n_results]

        for unique_result in unique_results:
            unique_result["metadata"]["last_access_timestamp"] = current_date.isoformat()
            unique_result["metadata"]["access_count"] = unique_result["metadata"]["access_count"] + 1

            if unique_result["memory_type"] == "working":
                self.vector_db.create_or_set_current_collection(self.working_collection)
                self.vector_db.update_metadata(
                    [unique_result["metadata"]["memory_id"]],
                    [unique_result["metadata"]]
                )
            elif unique_result["memory_type"] == "long_term":
                self.vector_db.create_or_set_current_collection(self.long_term_collection)
                self.vector_db.update_metadata(
                    [unique_result["metadata"]["pattern_id"]],
                    [unique_result["metadata"]]
                )

        return unique_results

    def _consolidate_patterns(self, timestamp):
        self.vector_db.create_or_set_current_collection(self.working_collection)
        working_memories = self.vector_db.get_all_entries()

        if not working_memories.chunks:
            return

        embeddings_result = self.embedding_provider.get_embedding(
            working_memories.chunks,
            EmbeddingTask.CLUSTER
        )
        embeddings = embeddings_result.embeddings

        clusters = self._cluster_embeddings(embeddings, self.minimum_cluster_similarity)

        for cluster_idx, cluster in enumerate(clusters):
            if len(cluster) < self.minimum_cluster_size:
                continue

            pattern_id = f"pattern_{uuid.uuid4()}"

            pattern = self._extract_pattern(
                pattern_id,
                [working_memories.chunks[i] for i in cluster],
                [working_memories.metadata[i] for i in cluster],
                timestamp,
            )

            self.vector_db.remove_by_ids(
                [working_memories.metadata[i]["memory_id"] for i in cluster]
            )

            self.vector_db.create_or_set_current_collection(self.long_term_collection)
            existing = self.vector_db.query(
                query=pattern["content"],
                k=1
            )

            if existing.scores and existing.scores[0] <= 0.01:
                continue

            self.vector_db.add_texts_with_id(
                ids=[pattern_id],
                texts=[pattern["content"]],
                metadata=[pattern["metadata"]]
            )

    def _format_results(
            self,
            results: VectorSearchResult,
            all_entries: VectorCollection,
            memory_type: str
    ) -> List[Dict]:
        formatted = []

        if not results.chunks:
            return formatted

        id_to_metadata = {id: meta for id, meta in zip(all_entries.ids, all_entries.metadata)}

        for doc_id, chunk, score in zip(results.ids, results.chunks, results.scores):
            similarity = max(0, min(1, 1 - score))
            if similarity > 0:
                metadata = id_to_metadata.get(doc_id, {})
                formatted.append({
                    "content": chunk,
                    "metadata": metadata,
                    "similarity": similarity,
                    "memory_type": memory_type,
                })

        return formatted

    def compute_memory_score(
            self, metadata, relevance, date, alpha_recency, alpha_relevance, alpha_frequency
    ):
        recency = self.compute_recency(metadata, date)
        frequency = np.log1p(metadata["access_count"])
        return (
                alpha_recency * recency
                + alpha_relevance * relevance
                + alpha_frequency * frequency
        )

    def compute_recency(self, metadata, date):
        time_diff = date - datetime.fromisoformat(metadata["last_access_timestamp"])
        hours_diff = time_diff.total_seconds() / 3600
        recency = self.decay_factor ** hours_diff
        return recency

    def get_stats(self) -> Dict:
        self.vector_db.create_or_set_current_collection(self.working_collection)
        working_count = len(self.vector_db.get_all_entries().ids)

        long_term_count = 0
        if self.long_term_collection:
            self.vector_db.create_or_set_current_collection(self.long_term_collection)
            long_term_count = len(self.vector_db.get_all_entries().ids)

        return {
            "working_count": working_count,
            "long_term_count": long_term_count,
        }