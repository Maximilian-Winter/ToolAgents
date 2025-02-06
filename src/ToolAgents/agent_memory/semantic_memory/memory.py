import abc
import dataclasses
import uuid

import chromadb
from chromadb.api.types import IncludeEnum
from sentence_transformers import SentenceTransformer
import numpy as np
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor
from typing import List, Dict, Optional, Any, Mapping
import json

from torch import Tensor

from ToolAgents.interfaces import LLMSamplingSettings
from ToolAgents.interfaces.base_llm_agent import BaseToolAgent


#import shutil
#import os
# Clean up any existing test database
#persist_directory = "./test_semantic_memory"
#if os.path.exists(persist_directory):
#    shutil.rmtree(persist_directory)

class ExtractPatternStrategy(abc.ABC):
    @abc.abstractmethod
    def extract_pattern(self, pattern_id, documents: List[str],
                        metadatas: list[Mapping[str, str | int | float | bool]], timestamp=datetime.now().isoformat()) -> Dict:
        pass


class SimpleExtractPatternStrategy(ExtractPatternStrategy):
    def extract_pattern(self, pattern_id, documents: List[str],
                        metadatas: List[Dict], timestamp=datetime.now().isoformat()) -> Dict:
        """Extract pattern from cluster of similar memories"""
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

        return {
            "content": '\n'.join(documents),
            "metadata": pattern_metadata
        }

class SummarizationExtractPatternStrategy(ExtractPatternStrategy):
    def __init__(self, agent: BaseToolAgent, summarizer_settings: LLMSamplingSettings, system_prompt_and_prefix: tuple[str, str] = None, pattern_type: str = "other", debug_mode: bool = False):
        self.agent = agent
        self.system_prompt = system_prompt_and_prefix
        if self.system_prompt is None:
            self.system_prompt = self.get_dynamic_prompt(pattern_type)
        self.debug_mode = debug_mode
        self.summarizer_settings = summarizer_settings


    @staticmethod
    def get_dynamic_prompt(pattern_type: str):
        if pattern_type == "learning":
            return "Summarize this knowledge into a structured explanation." , "Knowledge:\n"
        elif pattern_type == "observation":
            return "Condense these observations into a high-level summary.", "Observations:\n"
        elif pattern_type == "conversation":
            return "Extract key discussion points from this chat history.", "Conversation:\n"
        else:
            return "Summarize the information from different chat turns into one summary while keeping information as close to the original as possible. Only summarize what is explicitly mentioned. Keep the context (a chat) in which the information appeared clear in your summary!", "Chat turns:\n"

    def extract_pattern(self, pattern_id, documents: List[str],
                        metadatas: List[Dict], timestamp=datetime.now().isoformat()) -> Dict:
        """Extract pattern from cluster of similar memories"""
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

        if len(self.system_prompt) == 2:
            docs_prompt = f"{self.system_prompt[1]}"
        else:
            docs_prompt = "Chat turns:\n"

        for meta, doc in zip(metadatas, documents):
            docs_prompt += meta["timestamp"] + "\n"
            docs_prompt += doc + "\n---\n"

        result = self.agent.get_response(messages=[{"role": "system", "content": self.system_prompt[0] },
                                                   {"role": "user", "content": docs_prompt}],
                                         settings=self.summarizer_settings)
        if self.debug_mode:
            print(result, flush=True)
        return {
            "content": result,
            "metadata": pattern_metadata
        }

class ClusterEmbeddingsStrategy(abc.ABC):
    @abc.abstractmethod
    def cluster_embeddings(self, embeddings: List[np.ndarray], minimum_cluster_similarity: float = 0.75) -> List[
        List[int]]:
        pass


class SimpleClusterEmbeddingsStrategy(ClusterEmbeddingsStrategy):
    def cluster_embeddings(self, embeddings: List[np.ndarray | Tensor], minimum_cluster_similarity: float = 0.75) -> \
    List[List[int]]:
        """Cluster embeddings using cosine similarity"""
        embeddings_array = np.stack(embeddings)

        # Compute cosine similarity
        norms = np.linalg.norm(embeddings_array, axis=1)
        norms[norms == 0] = 1  # Avoid division by zero
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
                if j not in used_indices and similarity_matrix[i, j] > minimum_cluster_similarity:
                    cluster.append(j)
                    used_indices.add(j)

            clusters.append(cluster)

        return clusters


class CleanupStrategy(abc.ABC):
    @abc.abstractmethod
    def cleanup_working_memory(self, collection: chromadb.Collection,
                               current_date: datetime) -> None:
        pass

    @abc.abstractmethod
    def cleanup_long_term_memory(self, collection: chromadb.Collection,
                                 current_date: datetime) -> None:
        pass


class TimeBasedCleanupStrategy(CleanupStrategy):
    def __init__(self,
                 working_memory_ttl_hours: float = 24.0,
                 long_term_memory_ttl_days: float = 30.0,
                 min_access_count: int = 3):
        self.working_memory_ttl = timedelta(hours=working_memory_ttl_hours)
        self.long_term_memory_ttl = timedelta(days=long_term_memory_ttl_days)
        self.min_access_count = min_access_count

    def cleanup_working_memory(self, collection: chromadb.Collection,
                               current_date: datetime) -> None:
        """Remove old working memories with low access counts"""
        memories = collection.get(include=[IncludeEnum.metadatas])

        if not memories['metadatas']:
            return

        to_delete = []
        for memory_id, metadata in zip(memories['ids'], memories['metadatas']):
            last_access = datetime.fromisoformat(metadata['last_access_timestamp'])
            age = current_date - last_access

            if (age > self.working_memory_ttl and
                    metadata['access_count'] < self.min_access_count):
                to_delete.append(memory_id)

        if to_delete:
            collection.delete(ids=to_delete)

    def cleanup_long_term_memory(self, collection: chromadb.Collection,
                                 current_date: datetime) -> None:
        """Remove very old long-term memories that haven't been accessed"""
        memories = collection.get(include=[IncludeEnum.metadatas])

        if not memories['metadatas']:
            return

        to_delete = []
        for memory_id, metadata in zip(memories['ids'], memories['metadatas']):
            last_access = datetime.fromisoformat(metadata['last_access_timestamp'])
            age = current_date - last_access

            if age > self.long_term_memory_ttl:
                to_delete.append(memory_id)

        if to_delete:
            collection.delete(ids=to_delete)

@dataclasses.dataclass
class EmbeddingsConfig:
    sentence_transformer_model_path: str = "all-MiniLM-L6-v2"
    trust_remote_code: bool = False
    device: str = "cpu"
    embedding_kwargs: Dict[str, Any] = dataclasses.field(default_factory=dict)
    embeddings_store_prefix: str = None
    embeddings_recall_prefix: str = None
    embeddings_clusters_prefix: str = None



@dataclasses.dataclass
class SemanticMemoryConfig:
    persist: bool = True
    persist_directory: Optional[str] = "./memory"
    embeddings_config: EmbeddingsConfig = dataclasses.field(default_factory=EmbeddingsConfig)
    extract_pattern_strategy: ExtractPatternStrategy = SimpleExtractPatternStrategy()
    cluster_embeddings_strategy: ClusterEmbeddingsStrategy = SimpleClusterEmbeddingsStrategy()
    cleanup_strategy: CleanupStrategy = TimeBasedCleanupStrategy()
    enable_long_term_memory: bool = True
    cleanup_interval_hours: float = 1.0  # How often to run cleanup
    decay_factor: float = 0.98
    query_result_multiplier = 4
    minimum_cluster_size: int = 4
    minimum_cluster_similarity: float = 0.75
    minimum_similarity_threshold: float = 0.60
    debug_mode: bool = False

nomic_text_embeddings_gpu_config = EmbeddingsConfig(sentence_transformer_model_path="nomic-ai/nomic-embed-text-v1.5", embeddings_store_prefix="search_document: ", embeddings_recall_prefix="search_query: ", embeddings_clusters_prefix="cluster: ", trust_remote_code=True, device="cuda")
semantic_memory_nomic_text_gpu_config = SemanticMemoryConfig(embeddings_config=nomic_text_embeddings_gpu_config)

class SemanticMemory:
    def __init__(self, config: SemanticMemoryConfig = SemanticMemoryConfig()):
        """Initialize the semantic memory system"""
        self.encoder = SentenceTransformer(config.embeddings_config.sentence_transformer_model_path, trust_remote_code=config.embeddings_config.trust_remote_code,
                                           device=config.embeddings_config.device)

        self.enable_long_term_memory = config.enable_long_term_memory
        self.embeddings_store_prefix = config.embeddings_config.embeddings_store_prefix
        self.embeddings_recall_prefix = config.embeddings_config.embeddings_recall_prefix
        self.embedding_kwargs = config.embeddings_config.embedding_kwargs
        self.embeddings_clusters_prefix = config.embeddings_config.embeddings_clusters_prefix
        self.minimum_similarity_threshold = config.minimum_similarity_threshold
        self.debug_mode = config.debug_mode
        if config.persist:
            self.client = chromadb.PersistentClient(path=config.persist_directory)
        else:
            self.client = chromadb.EphemeralClient()

        self.decay_factor = config.decay_factor
        self.query_result_multiplier = config.query_result_multiplier
        self.minimum_cluster_size = config.minimum_cluster_size
        self.minimum_cluster_similarity = config.minimum_cluster_similarity

        self._extract_pattern = config.extract_pattern_strategy.extract_pattern
        self._cluster_embeddings = config.cluster_embeddings_strategy.cluster_embeddings

        self.cleanup_strategy = config.cleanup_strategy
        self.last_cleanup = datetime.now()
        self.cleanup_interval = timedelta(hours=config.cleanup_interval_hours)

        # Create collections with cosine similarity
        self.working = self.client.get_or_create_collection(
            name="working_memory",
            metadata={"hnsw:space": "cosine"}
        )
        if self.enable_long_term_memory:
            self.long_term = self.client.get_or_create_collection(
                name="long_term_memory",
                metadata={"hnsw:space": "cosine"}
            )
        else:
            self.long_term = None

    def _maybe_cleanup(self, current_date: datetime = datetime.now()):
        """Run cleanup if enough time has passed"""
        if (not self.cleanup_strategy or
                current_date - self.last_cleanup < self.cleanup_interval):
            return

        self.cleanup_strategy.cleanup_working_memory(self.working, current_date)
        self.cleanup_strategy.cleanup_long_term_memory(self.long_term, current_date)
        self.last_cleanup = current_date

    def store(self, content: str, context: Optional[Dict] = None, timestamp=datetime.now().isoformat()) -> str:
        """Store new memory with optional context"""

        memory_id = f"mem_{uuid.uuid4()}"

        if self.embeddings_store_prefix:
            embedding = self.encoder.encode(self.embeddings_store_prefix + content, **self.embedding_kwargs).tolist()
        else:
            embedding = self.encoder.encode(content, **self.embedding_kwargs).tolist()

        metadata = {
            "timestamp": timestamp,
            "last_access_timestamp": timestamp,
            "access_count": 1,
            "memory_id": memory_id,
            "type": "memory"
        }
        if context:
            metadata.update(context)

        # Store in immediate memory
        self.working.add(
            documents=[content],
            metadatas=[metadata],
            embeddings=[embedding],
            ids=[memory_id]
        )
        if  self.enable_long_term_memory and self.working.count() > self.minimum_cluster_size:
            self._consolidate_patterns(timestamp)
        self._maybe_cleanup(datetime.fromisoformat(timestamp))
        return memory_id

    def recall(self, query: str, n_results: int = 5, context_filter: Optional[Dict] = None,
               current_date: datetime = datetime.now(), alpha_recency=1, alpha_relevance=1, alpha_frequency=1) -> List[
        Dict]:
        """Recall memories similar to query using parallel search."""
        self._maybe_cleanup(current_date)

        if self.embeddings_recall_prefix:
            query_embedding = self.encoder.encode(self.embeddings_recall_prefix + query, **self.embedding_kwargs).tolist()
        else:
            query_embedding = self.encoder.encode(query, **self.embedding_kwargs).tolist()
        def search(collection: chromadb.Collection, mem_type):
            if collection.count() == 0:
                return []
            try:
                layer_results = collection.query(
                    query_embeddings=[query_embedding],
                    n_results=min(n_results * self.query_result_multiplier, collection.count()),
                    where=context_filter if context_filter and len(context_filter) > 0 else None
                )
                return self._format_results(layer_results, mem_type)
            except Exception as e:
                print(f"Warning: Error querying {mem_type} memory: {str(e)}")
                return []

        with (ThreadPoolExecutor() as executor):
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

        # Deduplicate & sort results
        seen_contents = ""
        unique_results = []
        for result in results:
            if result['content'] not in seen_contents:
                if result['similarity'] > self.minimum_similarity_threshold:
                    seen_contents += result['content'] + '\n'
                    result["rank_score"] = self.compute_memory_score(result["metadata"], result["similarity"], current_date,
                                                                     alpha_recency, alpha_relevance, alpha_frequency)
                    if self.debug_mode:
                        print(f"RS: {result['rank_score']}, S: {result['similarity']}", flush=True)
                    unique_results.append(result)

        unique_results.sort(key=lambda x: x['rank_score'], reverse=True)
        unique_results = unique_results[:n_results]
        for unique_result in unique_results:
            unique_result["metadata"]["last_access_timestamp"] = current_date.isoformat()
            unique_result["metadata"]["access_count"] = unique_result["metadata"]["access_count"] + 1
            if unique_result["metadata"]["type"] == "working":
                self.working.update([unique_result["metadata"]["memory_id"]], metadatas=[unique_result["metadata"]])
            elif unique_result["metadata"]["type"] == "long_term":
                self.long_term.update([unique_result["metadata"]["pattern_id"]], metadatas=[unique_result["metadata"]])

        return unique_results

    def _consolidate_patterns(self, timestamp):
        """Consolidate patterns across memory layers"""
        if self.embeddings_clusters_prefix is None:
            working_memories = self.working.get(include=[IncludeEnum.metadatas,IncludeEnum.embeddings, IncludeEnum.documents])

            if not working_memories['documents']:
                return

            # Generate embeddings
            embeddings = [
                embed
                for embed in working_memories['embeddings']
            ]
        else:
            working_memories = self.working.get(
                include=[IncludeEnum.metadatas, IncludeEnum.documents])
            embeddings = [self.encoder.encode(self.embeddings_clusters_prefix + doc, **self.embedding_kwargs) for doc in working_memories['documents']]
        clusters = self._cluster_embeddings(embeddings, self.minimum_cluster_similarity)

        # Process immediate to working memory
        for cluster_idx, cluster in enumerate(clusters):
            if len(cluster) < self.minimum_cluster_size:  # Skip singleton clusters
                continue

            pattern_id = f"pattern_{uuid.uuid4()}"

            pattern = self._extract_pattern(
                pattern_id,
                [working_memories['documents'][i] for i in cluster],
                [working_memories['metadatas'][i] for i in cluster],
                timestamp
            )

            pattern_embedding = self.encoder.encode(pattern['content']).tolist()

            self.working.delete(ids=[t["memory_id"] for t in [working_memories['metadatas'][i] for i in cluster]])
            # Check for duplicates
            existing = self.long_term.query(
                query_embeddings=[pattern_embedding],
                n_results=1
            )
            if existing['distances'] and len(existing['distances'][0]) > 0 and existing['distances'][0][0] <= 0.01:
                continue

            self.long_term.add(
                documents=[pattern['content']],
                metadatas=[pattern['metadata']],
                embeddings=[pattern_embedding],
                ids=[pattern_id]
            )

    def _format_results(self, results: Dict, memory_type: str) -> List[Dict]:
        """Format query results"""
        formatted = []

        if not results['documents']:
            return formatted

        documents = results['documents'][0]
        metadatas = results['metadatas'][0]
        distances = results['distances'][0]

        for doc, meta, dist in zip(documents, metadatas, distances):
            similarity = max(0, min(1, 1 - dist))  # Ensure similarity in [0,1]

            if similarity > 0:  # Only return meaningful results
                formatted.append({
                    'content': doc,
                    'metadata': meta,
                    'similarity': similarity,
                    'memory_type': memory_type
                })

        return formatted

    def compute_memory_score(self, metadata, relevance, date, alpha_recency, alpha_relevance, alpha_frequency):
        recency = self.compute_recency(metadata, date)  # Decays over time
        frequency = np.log1p(metadata["access_count"])  # Log scaling
        return (
                alpha_recency * recency +  # Recency importance
                alpha_relevance * relevance +  # Similarity score importance
                alpha_frequency * frequency  # Frequent recall importance
        )

    def compute_recency(self, metadata, date):
        decay_factor = self.decay_factor
        time_diff = date - datetime.fromisoformat(
            metadata["last_access_timestamp"]
        )
        hours_diff = time_diff.total_seconds() / 3600
        recency = decay_factor ** hours_diff
        return recency

    def get_stats(self) -> Dict:
        """Get memory system statistics"""
        return {
            "working_count": self.working.count(),
            "long_term_count": self.long_term.count()
        }
