"""
STREAM: Self-Transforming Recursive Embedding Associative Memory
A zero-LLM-intervention memory system using pure mathematical transformations
"""

import time
import threading
from typing import Optional, List, Dict, Any, Deque, Union
from dataclasses import dataclass, field
from collections import deque
import numpy as np
import torch
import torch.nn.functional as F
import pickle
import json
from pathlib import Path
from typing import Union
import hashlib

from ToolAgents.knowledge.vector_database import EmbeddingProvider, EmbeddingResult


@dataclass
class MemoryConfig:
    """Configuration for STREAM memory system"""
    embedding_dim: int = 384
    memory_rank: int = 128  # Maximum rank for SVD compression
    decay_rate: float = 0.995  # Temporal decay factor
    context_momentum: float = 0.7  # Context vector momentum
    semantic_weight: float = 0.6  # Weight for semantic vs temporal
    compression_threshold: int = 256  # When to trigger compression
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    dtype: torch.dtype = torch.float32
    max_sequence_length: int = 512  # Maximum tokens per input
    association_temperature: float = 0.5  # Temperature for association
    enable_hierarchical: bool = True  # Enable multi-scale memory
    memory_window: int = 1000  # Max number of memory traces to keep
    similarity_threshold: float = 0.3  # Threshold for pattern matching


@dataclass
class MemoryTrace:
    """A single memory trace - embedding with metadata"""
    embedding: torch.Tensor
    content: str
    timestamp: float
    activation_count: int = 0
    last_activation: float = 0.0
    associations: List[int] = field(default_factory=list)  # Indices of associated traces


class AssociativeIndex:
    """Fast associative retrieval using embedding similarity"""

    def __init__(self, config: MemoryConfig):
        self.config = config
        self.traces: Deque[MemoryTrace] = deque(maxlen=config.memory_window)
        self.embedding_matrix = None  # Will be built from traces
        self.content_hash = {}  # Hash -> trace index for deduplication

    def add_trace(self, embedding: torch.Tensor, content: str) -> int:
        """Add a new memory trace"""
        # Check for duplicate content
        content_hash = hashlib.md5(content.encode()).hexdigest()[:8]

        if content_hash in self.content_hash:
            # Reinforce existing trace instead of duplicating
            idx = self.content_hash[content_hash]
            if idx < len(self.traces):
                self.traces[idx].activation_count += 1
                self.traces[idx].last_activation = time.time()
                return idx

        # Create new trace
        trace = MemoryTrace(
            embedding=embedding,
            content=content,
            timestamp=time.time()
        )

        # Find associations with existing traces
        if self.embedding_matrix is not None:
            similarities = F.cosine_similarity(
                embedding.unsqueeze(0),
                self.embedding_matrix,
                dim=1
            )
            associated_indices = torch.where(
                similarities > self.config.similarity_threshold
            )[0].tolist()
            trace.associations = associated_indices

        self.traces.append(trace)
        idx = len(self.traces) - 1
        self.content_hash[content_hash] = idx

        # Rebuild embedding matrix
        self._rebuild_embedding_matrix()

        return idx

    def _rebuild_embedding_matrix(self):
        """Rebuild the embedding matrix from traces"""
        if len(self.traces) > 0:
            embeddings = [trace.embedding for trace in self.traces]
            self.embedding_matrix = torch.stack(embeddings)

    def retrieve(self, query_embedding: torch.Tensor, top_k: int = 5) -> List[MemoryTrace]:
        """Retrieve top-k most similar traces"""
        if self.embedding_matrix is None or len(self.traces) == 0:
            return []

        # Compute similarities
        similarities = F.cosine_similarity(
            query_embedding.unsqueeze(0),
            self.embedding_matrix,
            dim=1
        )

        # Apply temporal decay
        now = time.time()
        for i, trace in enumerate(self.traces):
            age = now - trace.timestamp
            decay = np.exp(-age / 3600)  # Decay over hours
            similarities[i] *= decay

        # Get top-k
        top_k = min(top_k, len(self.traces))
        top_values, top_indices = torch.topk(similarities, k=top_k)

        # Update activation counts
        retrieved_traces = []
        for idx in top_indices:
            if idx < len(self.traces):
                trace = self.traces[idx]
                trace.activation_count += 1
                trace.last_activation = now
                retrieved_traces.append(trace)

        return retrieved_traces


class StreamMemory:
    """Core STREAM memory implementation using PyTorch"""

    def __init__(self,
                 embedding_provider: EmbeddingProvider,
                 config: Optional[MemoryConfig] = None):
        self.embedding_provider = embedding_provider
        self.config = config or MemoryConfig()

        # Move to specified device
        self.device = torch.device(self.config.device)
        self.dtype = self.config.dtype

        # Initialize memory components
        self._init_memory()

        # Thread safety
        self.lock = threading.RLock()

        # Memory statistics
        self.stats = {
            'total_updates': 0,
            'compressions': 0,
            'last_update': time.time(),
            'memory_utilization': 0.0,
            'unique_memories': 0
        }

    def _init_memory(self):
        """Initialize memory components"""
        dim = self.config.embedding_dim

        # Associative index for content storage and retrieval
        self.associative_index = AssociativeIndex(self.config)

        # Running statistics tensors for pattern detection
        self.running_mean = torch.zeros(dim, device=self.device, dtype=self.dtype)
        self.running_covariance = torch.eye(dim, device=self.device, dtype=self.dtype)

        # Context state - maintains conversation flow
        self.context_state = torch.zeros(dim, device=self.device, dtype=self.dtype)
        self.context_momentum = self.config.context_momentum

        # Attention matrix - learns which dimensions correlate
        self.attention_matrix = torch.eye(dim, device=self.device, dtype=self.dtype)

        # Semantic and temporal fusion streams
        self.semantic_memory = torch.zeros((dim, dim), device=self.device, dtype=self.dtype)
        self.temporal_weights = torch.ones(dim, device=self.device, dtype=self.dtype)

        # Pattern bank - stores discovered patterns
        self.pattern_bank = []
        self.max_patterns = 50

    def process_input(self, text: str) -> Dict[str, Any]:
        """
        Process input text and update memory automatically
        Returns memory update results
        """
        with self.lock:
            # Get embedding
            embedding = self._get_embedding(text)

            # Add to associative index (stores content)
            trace_idx = self.associative_index.add_trace(embedding, text)

            # Update running statistics
            self._update_statistics(embedding)

            # Update context state
            self._update_context(embedding)

            # Detect and store patterns
            pattern = self._detect_pattern(embedding)
            if pattern is not None:
                self._store_pattern(pattern)

            # Update attention based on context flow
            self._update_attention(embedding)

            # Update stats
            self.stats['total_updates'] += 1
            self.stats['last_update'] = time.time()
            self.stats['unique_memories'] = len(self.associative_index.traces)

            return {
                'trace_index': trace_idx,
                'embedding_norm': torch.norm(embedding).item(),
                'context_similarity': F.cosine_similarity(
                    embedding.unsqueeze(0),
                    self.context_state.unsqueeze(0)
                ).item(),
                'pattern_detected': pattern is not None
            }

    def _get_embedding(self, text: str) -> torch.Tensor:
        """Get embedding from provider"""
        result = self.embedding_provider.get_embedding([text])
        embedding = torch.from_numpy(result.embeddings[0]).to(
            device=self.device, dtype=self.dtype
        )
        return F.normalize(embedding, p=2, dim=0)

    def _update_statistics(self, embedding: torch.Tensor):
        """Update running statistics for pattern detection"""
        # Update running mean
        alpha = 0.05
        self.running_mean = (1 - alpha) * self.running_mean + alpha * embedding

        # Update running covariance
        centered = embedding - self.running_mean
        cov_update = torch.outer(centered, centered)
        self.running_covariance = (1 - alpha) * self.running_covariance + alpha * cov_update

        # Update semantic memory (accumulates semantic patterns)
        self.semantic_memory = 0.99 * self.semantic_memory + torch.outer(embedding, embedding)

    def _update_context(self, embedding: torch.Tensor):
        """Update context state with momentum"""
        self.context_state = (
            self.context_momentum * self.context_state +
            (1 - self.context_momentum) * embedding
        )

        # Adjust momentum based on similarity (maintain context when similar)
        similarity = F.cosine_similarity(
            embedding.unsqueeze(0),
            self.context_state.unsqueeze(0)
        ).item()

        # Higher similarity -> higher momentum (keep context)
        self.context_momentum = 0.5 + 0.4 * similarity

    def _detect_pattern(self, embedding: torch.Tensor) -> Optional[torch.Tensor]:
        """Detect if embedding represents a significant pattern"""
        # Check if embedding deviates significantly from running mean
        centered = embedding - self.running_mean

        # Mahalanobis distance (how unusual is this embedding?)
        try:
            cov_inv = torch.linalg.pinv(self.running_covariance)
            mahalanobis = torch.sqrt(centered @ cov_inv @ centered)

            # If unusual enough, it's a pattern worth remembering
            if mahalanobis > 2.0:  # 2 standard deviations
                return embedding
        except:
            pass

        return None

    def _store_pattern(self, pattern: torch.Tensor):
        """Store discovered pattern"""
        if len(self.pattern_bank) >= self.max_patterns:
            # Remove oldest pattern
            self.pattern_bank.pop(0)

        self.pattern_bank.append({
            'pattern': pattern,
            'timestamp': time.time(),
            'strength': torch.norm(pattern).item()
        })

    def _update_attention(self, embedding: torch.Tensor):
        """Update attention matrix based on context flow"""
        # Learn which dimensions tend to co-activate
        outer_product = torch.outer(embedding, embedding)
        self.attention_matrix = 0.995 * self.attention_matrix + 0.005 * outer_product

        # Normalize to prevent explosion
        self.attention_matrix = self.attention_matrix / (torch.norm(self.attention_matrix) + 1e-6)

    def recall(self, query_text: str, top_k: int = 5) -> Dict[str, Any]:
        """
        Recall relevant memories using associative retrieval
        """
        with self.lock:
            # Get query embedding
            query_embedding = self._get_embedding(query_text)

            # Apply attention transformation
            attended_query = self.attention_matrix @ query_embedding

            # Retrieve similar traces from associative index
            similar_traces = self.associative_index.retrieve(attended_query, top_k)

            # Compute semantic activation
            semantic_activation = self.semantic_memory @ query_embedding
            semantic_score = torch.norm(semantic_activation).item() / np.sqrt(len(semantic_activation))

            # Check pattern matches
            pattern_matches = self._match_patterns(query_embedding)

            # Build recall results
            memories = []
            for trace in similar_traces:
                # Compute contextual relevance
                context_relevance = F.cosine_similarity(
                    trace.embedding.unsqueeze(0),
                    self.context_state.unsqueeze(0)
                ).item()

                memories.append({
                    'content': trace.content,
                    'timestamp': trace.timestamp,
                    'activation_count': trace.activation_count,
                    'context_relevance': context_relevance,
                    'associations': len(trace.associations)
                })

            return {
                'memories': memories,
                'semantic_score': semantic_score,
                'pattern_matches': pattern_matches,
                'context_similarity': F.cosine_similarity(
                    query_embedding.unsqueeze(0),
                    self.context_state.unsqueeze(0)
                ).item()
            }

    def _match_patterns(self, embedding: torch.Tensor) -> List[Dict]:
        """Match embedding against stored patterns"""
        matches = []
        for pattern_info in self.pattern_bank[-10:]:  # Check recent patterns
            pattern = pattern_info['pattern']
            similarity = F.cosine_similarity(
                embedding.unsqueeze(0),
                pattern.unsqueeze(0)
            ).item()

            if similarity > 0.7:
                matches.append({
                    'similarity': similarity,
                    'pattern_strength': pattern_info['strength'],
                    'pattern_age': time.time() - pattern_info['timestamp']
                })

        return matches

    def save(self, filepath: Union[str, Path]) -> None:
        """
        Save the complete memory state to disk

        Args:
            filepath: Path to save the memory state (will create .pt and .pkl files)
        """
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        with self.lock:
            # Prepare tensor state dict
            tensor_state = {
                'running_mean': self.running_mean.cpu(),
                'running_covariance': self.running_covariance.cpu(),
                'context_state': self.context_state.cpu(),
                'attention_matrix': self.attention_matrix.cpu(),
                'semantic_memory': self.semantic_memory.cpu(),
                'temporal_weights': self.temporal_weights.cpu(),
                'context_momentum': torch.tensor(self.context_momentum),
            }

            # Save tensors using PyTorch
            torch.save(tensor_state, f"{filepath}_tensors.pt")

            # Prepare non-tensor state
            # Convert traces to serializable format
            traces_data = []
            for trace in self.associative_index.traces:
                traces_data.append({
                    'embedding': trace.embedding.cpu().numpy(),
                    'content': trace.content,
                    'timestamp': trace.timestamp,
                    'activation_count': trace.activation_count,
                    'last_activation': trace.last_activation,
                    'associations': trace.associations
                })

            # Convert pattern bank to serializable format
            patterns_data = []
            for pattern_info in self.pattern_bank:
                patterns_data.append({
                    'pattern': pattern_info['pattern'].cpu().numpy(),
                    'timestamp': pattern_info['timestamp'],
                    'strength': pattern_info['strength']
                })

            # Prepare complete state
            state = {
                'config': {
                    'embedding_dim': self.config.embedding_dim,
                    'memory_rank': self.config.memory_rank,
                    'decay_rate': self.config.decay_rate,
                    'context_momentum': self.config.context_momentum,
                    'semantic_weight': self.config.semantic_weight,
                    'compression_threshold': self.config.compression_threshold,
                    'max_sequence_length': self.config.max_sequence_length,
                    'association_temperature': self.config.association_temperature,
                    'enable_hierarchical': self.config.enable_hierarchical,
                    'memory_window': self.config.memory_window,
                    'similarity_threshold': self.config.similarity_threshold,
                },
                'associative_index': {
                    'traces': traces_data,
                    'content_hash': self.associative_index.content_hash,
                },
                'pattern_bank': patterns_data,
                'stats': self.stats.copy(),
                'max_patterns': self.max_patterns,
            }

            # Save non-tensor state using pickle
            with open(f"{filepath}_state.pkl", 'wb') as f:
                pickle.dump(state, f, protocol=pickle.HIGHEST_PROTOCOL)

            # Save a metadata file for versioning and info
            metadata = {
                'version': '1.0',
                'save_time': time.time(),
                'total_memories': len(self.associative_index.traces),
                'total_patterns': len(self.pattern_bank),
                'device': str(self.device),
                'dtype': str(self.dtype),
            }

            with open(f"{filepath}_metadata.json", 'w') as f:
                json.dump(metadata, f, indent=2)

            print(f"Memory saved to {filepath} (3 files created)")

    def load(self, filepath: Union[str, Path]) -> None:
        """
        Load memory state from disk

        Args:
            filepath: Path to load the memory state from
        """
        filepath = Path(filepath)

        with self.lock:
            # Load metadata first to check compatibility
            try:
                with open(f"{filepath}_metadata.json", 'r') as f:
                    metadata = json.load(f)
                    print(f"Loading memory saved at {metadata['save_time']}")
            except FileNotFoundError:
                print("Warning: No metadata file found, loading anyway...")

            # Load tensor state
            tensor_state = torch.load(f"{filepath}_tensors.pt", map_location=self.device)

            # Restore tensors
            self.running_mean = tensor_state['running_mean'].to(self.device, dtype=self.dtype)
            self.running_covariance = tensor_state['running_covariance'].to(self.device, dtype=self.dtype)
            self.context_state = tensor_state['context_state'].to(self.device, dtype=self.dtype)
            self.attention_matrix = tensor_state['attention_matrix'].to(self.device, dtype=self.dtype)
            self.semantic_memory = tensor_state['semantic_memory'].to(self.device, dtype=self.dtype)
            self.temporal_weights = tensor_state['temporal_weights'].to(self.device, dtype=self.dtype)
            self.context_momentum = tensor_state['context_momentum'].item()

            # Load non-tensor state
            with open(f"{filepath}_state.pkl", 'rb') as f:
                state = pickle.load(f)

            # Restore configuration (update existing config)
            for key, value in state['config'].items():
                if hasattr(self.config, key):
                    setattr(self.config, key, value)

            # Restore associative index
            self.associative_index = AssociativeIndex(self.config)
            self.associative_index.content_hash = state['associative_index']['content_hash']

            # Restore traces
            for trace_data in state['associative_index']['traces']:
                trace = MemoryTrace(
                    embedding=torch.from_numpy(trace_data['embedding']).to(self.device, dtype=self.dtype),
                    content=trace_data['content'],
                    timestamp=trace_data['timestamp'],
                    activation_count=trace_data['activation_count'],
                    last_activation=trace_data['last_activation'],
                    associations=trace_data['associations']
                )
                self.associative_index.traces.append(trace)

            # Rebuild embedding matrix
            self.associative_index._rebuild_embedding_matrix()

            # Restore pattern bank
            self.pattern_bank = []
            for pattern_data in state['pattern_bank']:
                self.pattern_bank.append({
                    'pattern': torch.from_numpy(pattern_data['pattern']).to(self.device, dtype=self.dtype),
                    'timestamp': pattern_data['timestamp'],
                    'strength': pattern_data['strength']
                })

            # Restore stats
            self.stats = state['stats']
            self.max_patterns = state['max_patterns']

            print(f"Memory loaded: {len(self.associative_index.traces)} traces, {len(self.pattern_bank)} patterns")

    def export_to_json(self, filepath: Union[str, Path], include_embeddings: bool = False) -> None:
        """
        Export memory contents to human-readable JSON format

        Args:
            filepath: Path to save the JSON file
            include_embeddings: Whether to include embedding vectors (makes file much larger)
        """
        filepath = Path(filepath)

        with self.lock:
            export_data = {
                'metadata': {
                    'export_time': time.time(),
                    'total_memories': len(self.associative_index.traces),
                    'total_patterns': len(self.pattern_bank),
                    'memory_diversity': self.get_memory_summary()['memory_diversity'],
                    'context_stability': self.context_momentum,
                },
                'memories': [],
                'patterns': [],
                'stats': self.stats.copy()
            }

            # Export memories
            for i, trace in enumerate(self.associative_index.traces):
                memory_entry = {
                    'index': i,
                    'content': trace.content,
                    'timestamp': trace.timestamp,
                    'activation_count': trace.activation_count,
                    'last_activation': trace.last_activation,
                    'associations': trace.associations,
                }

                if include_embeddings:
                    memory_entry['embedding'] = trace.embedding.cpu().numpy().tolist()

                export_data['memories'].append(memory_entry)

            # Export patterns
            for i, pattern_info in enumerate(self.pattern_bank):
                pattern_entry = {
                    'index': i,
                    'timestamp': pattern_info['timestamp'],
                    'strength': pattern_info['strength'],
                }

                if include_embeddings:
                    pattern_entry['pattern'] = pattern_info['pattern'].cpu().numpy().tolist()

                export_data['patterns'].append(pattern_entry)

            # Sort memories by activation count for readability
            export_data['memories'].sort(key=lambda x: x['activation_count'], reverse=True)

            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(export_data, f, indent=2, ensure_ascii=False)

            print(f"Memory exported to {filepath} (human-readable JSON)")

    def merge_with(self, other_memory_path: Union[str, Path],
                   dedup: bool = True) -> None:
        """
        Merge another saved memory into this one

        Args:
            other_memory_path: Path to the other memory to merge
            dedup: Whether to deduplicate identical content
        """
        # Create temporary memory instance to load the other memory
        temp_memory = StreamMemory(self.embedding_provider, self.config)
        temp_memory.load(other_memory_path)

        with self.lock:
            # Merge traces
            merged_count = 0
            duplicate_count = 0

            for trace in temp_memory.associative_index.traces:
                if dedup:
                    # Check if content already exists
                    content_hash = hashlib.md5(trace.content.encode()).hexdigest()[:8]
                    if content_hash in self.associative_index.content_hash:
                        duplicate_count += 1
                        continue

                # Add the trace
                self.associative_index.add_trace(trace.embedding, trace.content)
                merged_count += 1

            # Merge patterns (keep unique ones based on similarity)
            for new_pattern_info in temp_memory.pattern_bank:
                is_duplicate = False

                if dedup:
                    for existing_pattern_info in self.pattern_bank:
                        similarity = F.cosine_similarity(
                            new_pattern_info['pattern'].unsqueeze(0),
                            existing_pattern_info['pattern'].unsqueeze(0)
                        ).item()

                        if similarity > 0.95:  # Very similar patterns
                            is_duplicate = True
                            break

                if not is_duplicate:
                    self.pattern_bank.append(new_pattern_info)

            # Update stats
            self.stats['total_updates'] += temp_memory.stats['total_updates']

            print(f"Merged {merged_count} new memories ({duplicate_count} duplicates skipped)")
            print(f"Total memories now: {len(self.associative_index.traces)}")
    def get_memory_summary(self) -> Dict[str, Any]:
        """Get comprehensive memory summary"""
        with self.lock:
            # Compute diversity of memory
            if self.associative_index.embedding_matrix is not None:
                pairwise_sim = F.cosine_similarity(
                    self.associative_index.embedding_matrix.unsqueeze(1),
                    self.associative_index.embedding_matrix.unsqueeze(0),
                    dim=2
                )
                avg_similarity = pairwise_sim.mean().item()
                diversity = 1.0 - avg_similarity
            else:
                diversity = 0.0

            # Get most activated memories
            if len(self.associative_index.traces) > 0:
                top_memories = sorted(
                    self.associative_index.traces,
                    key=lambda t: t.activation_count,
                    reverse=True
                )[:5]

                most_activated = [
                    {
                        'content': t.content[:50],
                        'activations': t.activation_count
                    }
                    for t in top_memories
                ]
            else:
                most_activated = []

            return {
                'total_memories': len(self.associative_index.traces),
                'unique_patterns': len(self.pattern_bank),
                'memory_diversity': diversity,
                'total_updates': self.stats['total_updates'],
                'most_activated': most_activated,
                'context_stability': self.context_momentum
            }
