import uuid


import chromadb
from sentence_transformers import SentenceTransformer
import numpy as np
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor
from typing import List, Dict, Optional
import json
import shutil
import os

# Clean up any existing test database
persist_directory = "./test_semantic_memory"
if os.path.exists(persist_directory):
    shutil.rmtree(persist_directory)

class SemanticMemory:
    def __init__(self, persist_directory: str = "./memory", sentence_transformer_model_path: str = "Snowflake/snowflake-arctic-embed-l-v2.0", trust_remote_code: bool = False, device: str = "cpu"):
        """Initialize the semantic memory system"""
        self.encoder = SentenceTransformer(sentence_transformer_model_path, trust_remote_code=trust_remote_code, device=device)
        self.client = chromadb.PersistentClient(path=persist_directory)

        self.decay_factor = 0.98
        self.query_result_multiplier = 4
        self.minimum_cluster_size = 2
        self.minimum_cluster_similarity = 0.7

        # Create collections with cosine similarity
        self.working = self.client.get_or_create_collection(
            name="working_memory",
            metadata={"hnsw:space": "cosine"}
        )

        self.long_term = self.client.get_or_create_collection(
            name="long_term_memory",
            metadata={"hnsw:space": "cosine"}
        )


    def store(self, content: str, context: Optional[Dict] = None, timestamp = datetime.now().isoformat()) -> str:
        """Store new memory with optional context"""

        memory_id = f"mem_{uuid.uuid4()}"

        # Generate embedding
        embedding = self.encoder.encode(content).tolist()

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

        self._consolidate_patterns(timestamp)
        return memory_id



    def recall(self, query: str, n_results: int = 5, context_filter: Optional[Dict] = None, current_date: datetime = datetime.now(), alpha_recency=1, alpha_relevance=1, alpha_frequency=1) -> List[Dict]:
        """Recall memories similar to query using parallel search."""
        query_embedding = self.encoder.encode(query).tolist()

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

        with ThreadPoolExecutor() as executor:
            futures = {
                executor.submit(search, self.working, "working"),
                executor.submit(search, self.long_term, "long_term")
            }
            results = []
            for future in futures:
                results.extend(future.result())

        # Deduplicate & sort results
        seen_contents = set()
        unique_results = []
        for result in results:
            if result['content'] not in seen_contents:
                seen_contents.add(result['content'])
                result["rank_score"] = self.compute_memory_score(result["metadata"], result["similarity"], current_date, alpha_recency, alpha_relevance, alpha_frequency)
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
        working_memories = self.working.get()

        if not working_memories['documents']:
            return

        # Generate embeddings
        embeddings = [
            self.encoder.encode(doc)
            for doc in working_memories['documents']
        ]

        clusters = self._cluster_embeddings(embeddings)

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

    def _cluster_embeddings(self, embeddings: List[np.ndarray]) -> List[List[int]]:
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
                if j not in used_indices and similarity_matrix[i, j] > self.minimum_cluster_similarity:
                    cluster.append(j)
                    used_indices.add(j)

            clusters.append(cluster)

        return clusters

    def _extract_pattern(self, pattern_id, documents: List[str],
                         metadatas: List[Dict], timestamp = datetime.now().isoformat()) -> Dict:
        """Extract pattern from cluster of similar memories"""

        pattern_metadata = {
            "type": "pattern",
            "timestamp": timestamp,
            "pattern_id": pattern_id,
            "last_access_timestamp": timestamp,
            "access_count": 1,
            "source_count": len(documents),
            "source_timestamps": json.dumps([m['timestamp'] for m in metadatas])
        }

        return {
            "content": '\n'.join(documents),
            "metadata": pattern_metadata
        }

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
        recency = decay_factor**hours_diff
        return recency


    def get_stats(self) -> Dict:
        """Get memory system statistics"""
        return {
            "working_count": self.working.count(),
            "long_term_count": self.long_term.count()
        }
