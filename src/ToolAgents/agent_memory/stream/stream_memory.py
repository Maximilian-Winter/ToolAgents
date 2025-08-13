"""
STREAM: Self-Transforming Recursive Embedding Associative Memory
A zero-LLM-intervention memory system using pure mathematical transformations
"""

import time
import threading
from typing import Optional, List, Dict, Tuple, Any, Deque
from dataclasses import dataclass, field
from collections import deque
import numpy as np
import torch
import torch.nn.functional as F
from datetime import datetime
import hashlib
import json
import heapq

# Import the provided embedding interface
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


class StreamPipeline:
    """High-level pipeline for using STREAM with LLMs"""

    def __init__(self,
                 embedding_provider: EmbeddingProvider,
                 config: Optional[MemoryConfig] = None):
        self.memory = StreamMemory(embedding_provider, config)
        self.conversation_history = deque(maxlen=100)
        self.last_context = None

    def process_conversation(self,
                            user_input: str,
                            include_context: bool = True) -> Dict[str, Any]:
        """
        Process conversation turn and generate context
        """
        # Update memory with user input
        update_result = self.memory.process_input(user_input)

        # Store in conversation history
        self.conversation_history.append({
            'role': 'user',
            'content': user_input,
            'timestamp': time.time()
        })

        # Generate context if requested
        context = ""
        relevance_score = 0.0

        if include_context:
            recall_result = self.memory.recall(user_input, top_k=3)
            context = self._build_context(recall_result)
            relevance_score = recall_result['semantic_score']
            self.last_context = context

        return {
            'context': context,
            'relevance_score': relevance_score,
            'pattern_detected': update_result['pattern_detected'],
            'context_similarity': update_result['context_similarity']
        }

    def update_with_response(self, response: str):
        """Update memory with LLM response"""
        update_result = self.memory.process_input(response)

        self.conversation_history.append({
            'role': 'assistant',
            'content': response,
            'timestamp': time.time()
        })

        return update_result

    def _build_context(self, recall_result: Dict[str, Any]) -> str:
        """Build natural language context from recall results"""
        parts = []

        # Add relevant memories
        memories = recall_result['memories']
        if memories:
            # Group memories by relevance
            high_relevance = [m for m in memories if m['context_relevance'] > 0.7]
            moderate_relevance = [m for m in memories if 0.4 <= m['context_relevance'] <= 0.7]

            if high_relevance:
                parts.append("Highly relevant context:")
                for mem in high_relevance[:2]:
                    parts.append(f"- {mem['content']}")

            if moderate_relevance and len(parts) < 3:
                parts.append("Related context:")
                for mem in moderate_relevance[:2]:
                    parts.append(f"- {mem['content']}")

        # Add pattern information if significant
        if recall_result['pattern_matches']:
            parts.append(f"(Pattern similarity detected: {len(recall_result['pattern_matches'])} matches)")

        # Only return context if we have something meaningful
        if parts:
            return "\n".join(parts)
        else:
            return ""

    def get_conversation_summary(self) -> Dict[str, Any]:
        """Get conversation and memory summary"""
        memory_summary = self.memory.get_memory_summary()

        # Add conversation-specific stats
        conversation_stats = {
            'total_turns': len(self.conversation_history),
            'last_context': self.last_context if self.last_context else "No context generated yet"
        }

        return {**memory_summary, **conversation_stats}


def example_usage():
    """Demonstrate STREAM usage with your embedding provider"""
    from ToolAgents.knowledge.vector_database.implementations.sentence_transformer_embeddings import SentenceTransformerEmbeddingProvider

    provider = SentenceTransformerEmbeddingProvider()
    config = MemoryConfig(
        embedding_dim=384,  # all-MiniLM-L6-v2 uses 384 dimensions
        memory_rank=64,
        decay_rate=0.995,
        device="cpu"
    )

    # Create pipeline
    pipeline = StreamPipeline(provider, config)

    # Simulate conversation
    conversation = [
        "Tell me about quantum computing",
        "What are qubits?",
        "How do quantum gates work?",
        "Let's switch topics - what's machine learning?",
        "How do neural networks learn?",
        "Going back to quantum - how does entanglement work?"
    ]

    print("STREAM Memory System Demo\n" + "="*50)

    for i, user_input in enumerate(conversation, 1):
        print(f"\n[Turn {i}] User: {user_input}")

        # Process input and get context
        result = pipeline.process_conversation(user_input)

        if result['context']:
            print(f"📊 Retrieved Context:\n{result['context']}")
        else:
            print(f"📊 No relevant context yet (building memory...)")

        print(f"📈 Relevance Score: {result['relevance_score']:.3f}")
        print(f"🔄 Context Similarity: {result['context_similarity']:.3f}")

        if result['pattern_detected']:
            print("✨ New pattern detected and stored!")

        # Simulate LLM response
        mock_response = f"Here's information about {user_input.lower().replace('?', '')}..."
        pipeline.update_with_response(mock_response)

    # Final summary
    print("\n" + "="*50)
    print("Memory Summary:")
    summary = pipeline.get_conversation_summary()
    for key, value in summary.items():
        if key != 'most_activated' and key != 'last_context':
            print(f"  {key}: {value}")

    if summary['most_activated']:
        print("\nMost Activated Memories:")
        for mem in summary['most_activated']:
            print(f"  - '{mem['content']}...' (activated {mem['activations']} times)")


if __name__ == "__main__":
    example_usage()