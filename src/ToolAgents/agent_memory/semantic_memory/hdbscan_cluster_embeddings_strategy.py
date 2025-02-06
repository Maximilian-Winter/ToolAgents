import hdbscan
import numpy as np
from typing import List
from ToolAgents.agent_memory.semantic_memory.memory import ClusterEmbeddingsStrategy
from torch import Tensor


class HDBSCANClusterEmbeddingsStrategy(ClusterEmbeddingsStrategy):
    def __init__(self, min_cluster_size: int = 3, min_samples: int = 2):
        """
        HDBSCAN-based clustering strategy.

        :param min_cluster_size: Minimum number of points to form a cluster.
        :param min_samples: Minimum samples per cluster (adjusts how conservative clustering is).
        """
        self.min_cluster_size = min_cluster_size
        self.min_samples = min_samples

    def cluster_embeddings(self, embeddings: List[np.ndarray | Tensor], minimum_cluster_similarity: float = 0.75) -> List[List[int]]:
        """Cluster embeddings using HDBSCAN with cosine similarity."""

        if len(embeddings) < self.min_cluster_size:
            return [[i] for i in range(len(embeddings))]  # If too few points, return single-point clusters

        # Convert embeddings to numpy array (force float64)
        embeddings_array = np.array([e.astype(np.float64) for e in embeddings])

        # Normalize embeddings for cosine similarity
        norms = np.linalg.norm(embeddings_array, axis=1, keepdims=True)
        norms[norms == 0] = 1  # Avoid division by zero
        normalized_embeddings = embeddings_array / norms

        # Compute cosine distance matrix (1 - cosine similarity)
        similarity_matrix = np.dot(normalized_embeddings, normalized_embeddings.T)
        distance_matrix = (1 - similarity_matrix).astype(np.float64)  # Convert to float64 explicitly

        # Run HDBSCAN clustering
        clusterer = hdbscan.HDBSCAN(
            metric="precomputed",  # Using cosine distance matrix
            min_cluster_size=self.min_cluster_size,
            min_samples=self.min_samples,
            cluster_selection_method="eom"  # Excess of mass, good for dense regions
        )

        labels = clusterer.fit_predict(distance_matrix)

        # Organize clusters
        clusters = {}
        for idx, label in enumerate(labels):
            if label == -1:
                continue  # Ignore noise points (outliers)
            if label not in clusters:
                clusters[label] = []
            clusters[label].append(idx)

        return list(clusters.values())  # Convert dict to list of lists
