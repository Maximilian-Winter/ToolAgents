"""
STREAM: Self-Transforming Recursive Embedding Associative Memory
Enhanced version with hybrid retrieval, memory consolidation, and multi-scale context
"""

import time
import threading
from typing import Optional, List, Dict, Any, Deque, Union, Set, Tuple
from dataclasses import dataclass, field
from collections import deque, Counter
import numpy as np
import torch
import torch.nn.functional as F
import pickle
import json
from pathlib import Path
import hashlib
import re
from sklearn.cluster import DBSCAN

from ToolAgents.knowledge.vector_database import EmbeddingProvider, EmbeddingResult


@dataclass
class MemoryConfig:
    """Configuration for STREAM memory system"""
    embedding_dim: int = 768
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

    # New configuration options
    consolidation_threshold: float = 0.85  # Similarity threshold for consolidation
    min_cluster_size: int = 3  # Minimum memories to form a consolidated cluster
    pattern_importance_threshold: float = 0.5  # Minimum importance to keep pattern
    enable_keyword_retrieval: bool = True  # Enable hybrid keyword search
    confidence_threshold: float = 0.3  # Minimum confidence for retrieval


@dataclass
class MemoryTrace:
    """A single memory trace - embedding with metadata"""
    embedding: torch.Tensor
    content: str
    timestamp: float
    activation_count: int = 0
    last_activation: float = 0.0
    associations: List[int] = field(default_factory=list)  # Indices of associated traces
    keywords: Set[str] = field(default_factory=set)  # Extracted keywords
    confidence: float = 1.0  # Confidence score for this memory
    is_consolidated: bool = False  # Whether this is a consolidated memory
    source_traces: List[int] = field(default_factory=list)  # For consolidated memories


class MultiScaleContext:
    """Maintains context at multiple temporal scales"""

    def __init__(self, dim: int, device: torch.device, dtype: torch.dtype):
        self.dim = dim
        self.device = device
        self.dtype = dtype

        # Different timescale contexts
        self.short_term = torch.zeros(dim, device=device, dtype=dtype)  # Last 1-3 exchanges
        self.medium_term = torch.zeros(dim, device=device, dtype=dtype)  # Current topic (~10 exchanges)
        self.long_term = torch.zeros(dim, device=device, dtype=dtype)  # Entire conversation

        # Momentum values for each scale
        self.short_momentum = 0.3
        self.medium_momentum = 0.7
        self.long_momentum = 0.95

    def update(self, embedding: torch.Tensor) -> None:
        """Update all context scales"""
        self.short_term = self.short_momentum * self.short_term + (1 - self.short_momentum) * embedding
        self.medium_term = self.medium_momentum * self.medium_term + (1 - self.medium_momentum) * embedding
        self.long_term = self.long_momentum * self.long_term + (1 - self.long_momentum) * embedding

    def get_combined_context(self, weights: Tuple[float, float, float] = (0.5, 0.3, 0.2)) -> torch.Tensor:
        """Get weighted combination of all contexts"""
        return (weights[0] * self.short_term +
                weights[1] * self.medium_term +
                weights[2] * self.long_term)

    def get_multi_scale_similarity(self, embedding: torch.Tensor) -> Dict[str, float]:
        """Get similarity at each scale"""
        return {
            'short_term': F.cosine_similarity(embedding.unsqueeze(0), self.short_term.unsqueeze(0)).item(),
            'medium_term': F.cosine_similarity(embedding.unsqueeze(0), self.medium_term.unsqueeze(0)).item(),
            'long_term': F.cosine_similarity(embedding.unsqueeze(0), self.long_term.unsqueeze(0)).item()
        }


class EnhancedAssociativeIndex:
    """Enhanced associative retrieval with keyword search and better deduplication"""

    def __init__(self, config: MemoryConfig):
        self.config = config
        self.traces: Deque[MemoryTrace] = deque(maxlen=config.memory_window)
        self.embedding_matrix = None  # Will be built from traces
        self.content_hash = {}  # Full hash -> trace index for deduplication
        self.keyword_index = {}  # Keyword -> set of trace indices

    def _extract_keywords(self, content: str) -> Set[str]:
        """Extract meaningful keywords from content"""
        # Remove common stop words and extract meaningful terms
        stop_words = {'the', 'is', 'at', 'which', 'on', 'a', 'an', 'and', 'or', 'but',
                      'in', 'with', 'to', 'for', 'of', 'as', 'by', 'that', 'this',
                      'it', 'from', 'what', 'when', 'where', 'how', 'why', 'who'}

        # Simple keyword extraction (can be enhanced with NLTK or spaCy)
        words = re.findall(r'\b[a-z]+\b', content.lower())
        keywords = {word for word in words if word not in stop_words and len(word) > 2}

        # Also extract any capitalized words (likely proper nouns)
        proper_nouns = re.findall(r'\b[A-Z][a-z]+\b', content)
        keywords.update(word.lower() for word in proper_nouns)

        return keywords

    def add_trace(self, embedding: torch.Tensor, content: str,
                  is_consolidated: bool = False, source_traces: List[int] = None) -> int:
        """Add a new memory trace with enhanced deduplication"""
        # Use full SHA-256 hash for better deduplication
        content_hash = hashlib.sha256(content.encode()).hexdigest()

        # Check for semantic similarity to existing traces
        if self.embedding_matrix is not None and not is_consolidated:
            similarities = F.cosine_similarity(
                embedding.unsqueeze(0),
                self.embedding_matrix,
                dim=1
            )

            # If very similar content exists (>0.95 similarity), reinforce it
            max_similarity, max_idx = torch.max(similarities, dim=0)
            if max_similarity > 0.95 and max_idx < len(self.traces):
                existing_trace = self.traces[max_idx]
                existing_trace.activation_count += 1
                existing_trace.last_activation = time.time()
                existing_trace.confidence = min(1.0, existing_trace.confidence * 1.1)
                return max_idx

        if content_hash in self.content_hash and not is_consolidated:
            # Reinforce existing trace
            idx = self.content_hash[content_hash]
            if idx < len(self.traces):
                self.traces[idx].activation_count += 1
                self.traces[idx].last_activation = time.time()
                self.traces[idx].confidence = min(1.0, self.traces[idx].confidence * 1.1)
                return idx

        # Extract keywords
        keywords = self._extract_keywords(content)

        # Create new trace
        trace = MemoryTrace(
            embedding=embedding,
            content=content,
            timestamp=time.time(),
            keywords=keywords,
            is_consolidated=is_consolidated,
            source_traces=source_traces or []
        )

        # Find associations with existing traces
        if self.embedding_matrix is not None:
            similarities = F.cosine_similarity(
                embedding.unsqueeze(0),
                self.embedding_matrix,
                dim=1
            )
            # Use adaptive threshold based on content type
            threshold = self.config.similarity_threshold if not is_consolidated else 0.5
            associated_indices = torch.where(similarities > threshold)[0].tolist()
            trace.associations = associated_indices

        self.traces.append(trace)
        idx = len(self.traces) - 1
        self.content_hash[content_hash] = idx

        # Update keyword index
        for keyword in keywords:
            if keyword not in self.keyword_index:
                self.keyword_index[keyword] = set()
            self.keyword_index[keyword].add(idx)

        # Rebuild embedding matrix
        self._rebuild_embedding_matrix()

        return idx

    def _rebuild_embedding_matrix(self):
        """Rebuild the embedding matrix from traces"""
        if len(self.traces) > 0:
            embeddings = [trace.embedding for trace in self.traces]
            self.embedding_matrix = torch.stack(embeddings)

    def retrieve_semantic(self, query_embedding: torch.Tensor, top_k: int = 5) -> List[Tuple[MemoryTrace, float]]:
        """Retrieve top-k most similar traces with scores"""
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
            # Boost consolidated memories
            if trace.is_consolidated:
                decay *= 1.5
            similarities[i] *= decay * trace.confidence

        # Get top-k
        top_k = min(top_k, len(self.traces))
        top_values, top_indices = torch.topk(similarities, k=top_k)

        # Update activation counts and return with scores
        retrieved_traces = []
        for idx, score in zip(top_indices, top_values):
            if idx < len(self.traces):
                trace = self.traces[idx]
                trace.activation_count += 1
                trace.last_activation = now
                retrieved_traces.append((trace, score.item()))

        return retrieved_traces

    def retrieve_keywords(self, keywords: Set[str], top_k: int = 5) -> List[Tuple[MemoryTrace, float]]:
        """Retrieve traces matching keywords"""
        trace_scores = Counter()

        for keyword in keywords:
            if keyword in self.keyword_index:
                for trace_idx in self.keyword_index[keyword]:
                    if trace_idx < len(self.traces):
                        trace_scores[trace_idx] += 1

        # Get top traces by keyword match count
        top_traces = []
        for trace_idx, match_count in trace_scores.most_common(top_k):
            if trace_idx < len(self.traces):
                trace = self.traces[trace_idx]
                # Score based on keyword match percentage
                score = match_count / max(len(keywords), len(trace.keywords))
                top_traces.append((trace, score))

        return top_traces


class PatternBank:
    """Enhanced pattern storage with importance scoring"""

    def __init__(self, config: MemoryConfig):
        self.config = config
        self.patterns = []
        self.pattern_embeddings = None
        self.max_patterns = 100  # Increased from 50

    def add_pattern(self, pattern: torch.Tensor, importance: float, metadata: Dict[str, Any] = None):
        """Add pattern with importance score"""
        pattern_info = {
            'pattern': pattern,
            'timestamp': time.time(),
            'strength': torch.norm(pattern).item(),
            'importance': importance,
            'activation_count': 0,
            'metadata': metadata or {}
        }

        # Check if similar pattern exists
        if self.pattern_embeddings is not None:
            similarities = F.cosine_similarity(
                pattern.unsqueeze(0),
                self.pattern_embeddings,
                dim=1
            )
            max_sim, max_idx = torch.max(similarities, dim=0)

            if max_sim > 0.9:  # Very similar pattern exists
                # Update importance of existing pattern
                self.patterns[max_idx]['importance'] = max(
                    self.patterns[max_idx]['importance'],
                    importance
                )
                self.patterns[max_idx]['activation_count'] += 1
                return

        self.patterns.append(pattern_info)
        self._prune_patterns()
        self._rebuild_embeddings()

    def _calculate_pattern_importance(self, pattern_info: Dict) -> float:
        """Calculate current importance of a pattern"""
        age_factor = np.exp(-(time.time() - pattern_info['timestamp']) / 86400)  # Decay over days
        activation_factor = np.log1p(pattern_info['activation_count'])
        return pattern_info['importance'] * age_factor * (1 + activation_factor * 0.1)

    def _prune_patterns(self):
        """Remove low-importance patterns if over limit"""
        if len(self.patterns) > self.max_patterns:
            # Calculate current importance for all patterns
            for p in self.patterns:
                p['current_importance'] = self._calculate_pattern_importance(p)

            # Sort by current importance and keep top patterns
            self.patterns.sort(key=lambda p: p['current_importance'], reverse=True)
            self.patterns = self.patterns[:self.max_patterns]

    def _rebuild_embeddings(self):
        """Rebuild pattern embedding matrix"""
        if len(self.patterns) > 0:
            embeddings = [p['pattern'] for p in self.patterns]
            self.pattern_embeddings = torch.stack(embeddings)

    def match_patterns(self, embedding: torch.Tensor, threshold: float = 0.7) -> List[Dict]:
        """Match embedding against stored patterns"""
        if self.pattern_embeddings is None or len(self.patterns) == 0:
            return []

        similarities = F.cosine_similarity(
            embedding.unsqueeze(0),
            self.pattern_embeddings,
            dim=1
        )

        matches = []
        for idx, sim in enumerate(similarities):
            if sim > threshold:
                self.patterns[idx]['activation_count'] += 1
                matches.append({
                    'similarity': sim.item(),
                    'pattern_strength': self.patterns[idx]['strength'],
                    'pattern_importance': self._calculate_pattern_importance(self.patterns[idx]),
                    'pattern_age': time.time() - self.patterns[idx]['timestamp'],
                    'metadata': self.patterns[idx].get('metadata', {})
                })

        return matches


class StreamMemory:
    """Enhanced STREAM memory implementation"""

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
            'consolidations': 0,
            'last_update': time.time(),
            'memory_utilization': 0.0,
            'unique_memories': 0,
            'keyword_hits': 0,
            'semantic_hits': 0
        }

    def _init_memory(self):
        """Initialize enhanced memory components"""
        dim = self.config.embedding_dim

        # Enhanced associative index
        self.associative_index = EnhancedAssociativeIndex(self.config)

        # Multi-scale context
        self.context = MultiScaleContext(dim, self.device, self.dtype)

        # Enhanced pattern bank
        self.pattern_bank = PatternBank(self.config)

        # Running statistics tensors
        self.running_mean = torch.zeros(dim, device=self.device, dtype=self.dtype)
        self.running_covariance = torch.eye(dim, device=self.device, dtype=self.dtype)

        # Attention matrix
        self.attention_matrix = torch.eye(dim, device=self.device, dtype=self.dtype)

        # Semantic and temporal fusion streams
        self.semantic_memory = torch.zeros((dim, dim), device=self.device, dtype=self.dtype)
        self.temporal_weights = torch.ones(dim, device=self.device, dtype=self.dtype)

        # Consolidation tracking
        self.last_consolidation = time.time()
        self.consolidation_interval = 3600  # Consolidate every hour

    def process_input(self, text: str) -> Dict[str, Any]:
        """Process input text with enhanced memory update"""
        with self.lock:
            # Get embedding
            embedding = self._get_embedding(text)

            # Add to associative index
            trace_idx = self.associative_index.add_trace(embedding, text)

            # Update multi-scale context
            self.context.update(embedding)
            context_similarities = self.context.get_multi_scale_similarity(embedding)

            # Update running statistics
            self._update_statistics(embedding)

            # Detect and store patterns with importance
            pattern_importance = self._calculate_pattern_importance(embedding)
            if pattern_importance > self.config.pattern_importance_threshold:
                self.pattern_bank.add_pattern(
                    embedding,
                    pattern_importance,
                    metadata={'source_text': text[:100]}
                )

            # Update attention based on context flow
            self._update_attention(embedding)

            # Check if consolidation is needed
            if time.time() - self.last_consolidation > self.consolidation_interval:
                self.consolidate_memories()

            # Update stats
            self.stats['total_updates'] += 1
            self.stats['last_update'] = time.time()
            self.stats['unique_memories'] = len(self.associative_index.traces)

            return {
                'trace_index': trace_idx,
                'embedding_norm': torch.norm(embedding).item(),
                'context_similarities': context_similarities,
                'pattern_importance': pattern_importance,
                'memory_count': len(self.associative_index.traces)
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
        alpha = 0.05
        self.running_mean = (1 - alpha) * self.running_mean + alpha * embedding

        centered = embedding - self.running_mean
        cov_update = torch.outer(centered, centered)
        self.running_covariance = (1 - alpha) * self.running_covariance + alpha * cov_update

        self.semantic_memory = 0.99 * self.semantic_memory + torch.outer(embedding, embedding)

    def _calculate_pattern_importance(self, embedding: torch.Tensor) -> float:
        """Calculate importance score for a potential pattern"""
        centered = embedding - self.running_mean

        try:
            cov_inv = torch.linalg.pinv(self.running_covariance)
            mahalanobis = torch.sqrt(centered @ cov_inv @ centered).item()

            # Consider multiple factors for importance
            novelty = min(mahalanobis / 2.0, 1.0)  # Normalized novelty

            # Context coherence across scales
            context_sims = self.context.get_multi_scale_similarity(embedding)
            context_coherence = np.mean(list(context_sims.values()))

            # Combined importance score
            importance = novelty * 0.6 + context_coherence * 0.4

            return importance
        except:
            return 0.0

    def _update_attention(self, embedding: torch.Tensor):
        """Update attention matrix based on context flow"""
        outer_product = torch.outer(embedding, embedding)
        self.attention_matrix = 0.995 * self.attention_matrix + 0.005 * outer_product
        self.attention_matrix = self.attention_matrix / (torch.norm(self.attention_matrix) + 1e-6)

    def recall_hybrid(self, query_text: str, top_k: int = 5) -> Dict[str, Any]:
        """Enhanced hybrid recall with semantic and keyword search"""
        with self.lock:
            # Get query embedding
            query_embedding = self._get_embedding(query_text)

            # Apply attention transformation
            attended_query = self.attention_matrix @ query_embedding

            # Semantic retrieval
            semantic_results = self.associative_index.retrieve_semantic(attended_query, top_k * 2)

            # Keyword retrieval if enabled
            keyword_results = []
            if self.config.enable_keyword_retrieval:
                keywords = self.associative_index._extract_keywords(query_text)
                keyword_results = self.associative_index.retrieve_keywords(keywords, top_k)
                self.stats['keyword_hits'] += len(keyword_results)

            self.stats['semantic_hits'] += len(semantic_results)

            # Merge results with confidence scores
            merged_results = self._merge_retrieval_results(
                semantic_results,
                keyword_results,
                query_embedding,
                top_k
            )

            # Check pattern matches
            pattern_matches = self.pattern_bank.match_patterns(query_embedding)

            # Compute semantic activation
            semantic_activation = self.semantic_memory @ query_embedding
            semantic_score = torch.norm(semantic_activation).item() / np.sqrt(len(semantic_activation))

            return {
                'memories': merged_results,
                'semantic_score': semantic_score,
                'pattern_matches': pattern_matches,
                'context_similarities': self.context.get_multi_scale_similarity(query_embedding),
                'retrieval_stats': {
                    'semantic_count': len(semantic_results),
                    'keyword_count': len(keyword_results),
                    'merged_count': len(merged_results)
                }
            }

    def recall(self, query_text: str, top_k: int = 5) -> Dict[str, Any]:
        """Backward compatible recall method that uses hybrid retrieval"""
        return self.recall_hybrid(query_text, top_k)

    def _merge_retrieval_results(self,
                                 semantic_results: List[Tuple[MemoryTrace, float]],
                                 keyword_results: List[Tuple[MemoryTrace, float]],
                                 query_embedding: torch.Tensor,
                                 top_k: int) -> List[Dict[str, Any]]:
        """Merge and rank results from different retrieval methods"""
        # Combine all results with weighted scores
        trace_scores = {}

        # Add semantic results
        for trace, score in semantic_results:
            trace_id = id(trace)
            if trace_id not in trace_scores:
                trace_scores[trace_id] = {
                    'trace': trace,
                    'semantic_score': 0,
                    'keyword_score': 0,
                    'combined_score': 0
                }
            trace_scores[trace_id]['semantic_score'] = score

        # Add keyword results
        for trace, score in keyword_results:
            trace_id = id(trace)
            if trace_id not in trace_scores:
                trace_scores[trace_id] = {
                    'trace': trace,
                    'semantic_score': 0,
                    'keyword_score': 0,
                    'combined_score': 0
                }
            trace_scores[trace_id]['keyword_score'] = score

        # Calculate combined scores with confidence
        for trace_id, scores in trace_scores.items():
            trace = scores['trace']

            # Calculate confidence based on multiple factors
            confidence = self._calculate_confidence(
                trace=trace,
                semantic_score=scores['semantic_score'],
                keyword_score=scores['keyword_score'],
                query_embedding=query_embedding
            )

            # Weighted combination
            scores['combined_score'] = (
                                               0.7 * scores['semantic_score'] +
                                               0.3 * scores['keyword_score']
                                       ) * confidence
            scores['confidence'] = confidence

        # Sort by combined score and return top-k
        sorted_results = sorted(
            trace_scores.values(),
            key=lambda x: x['combined_score'],
            reverse=True
        )[:top_k]

        # Format results
        memories = []
        for result in sorted_results:
            trace = result['trace']

            # Get context relevance
            context_relevance = F.cosine_similarity(
                trace.embedding.unsqueeze(0),
                self.context.get_combined_context().unsqueeze(0)
            ).item()

            memories.append({
                'content': trace.content,
                'timestamp': trace.timestamp,
                'activation_count': trace.activation_count,
                'context_relevance': context_relevance,
                'associations': len(trace.associations),
                'confidence': result['confidence'],
                'is_consolidated': trace.is_consolidated,
                'semantic_score': result['semantic_score'],
                'keyword_score': result['keyword_score']
            })

        return memories

    def _calculate_confidence(self,
                              trace: MemoryTrace,
                              semantic_score: float,
                              keyword_score: float,
                              query_embedding: torch.Tensor) -> float:
        """Calculate confidence score for a retrieved memory"""
        # Base confidence from trace
        base_confidence = trace.confidence

        # Recency factor
        age_hours = (time.time() - trace.timestamp) / 3600
        recency_factor = np.exp(-age_hours / 24)  # Decay over days

        # Activation factor (how often this memory has been accessed)
        activation_factor = np.tanh(trace.activation_count / 10)

        # Retrieval strength (how well it matches the query)
        retrieval_strength = max(semantic_score, keyword_score)

        # Consolidated memory boost
        consolidation_boost = 1.2 if trace.is_consolidated else 1.0

        # Combined confidence
        confidence = (
                             base_confidence * 0.3 +
                             recency_factor * 0.2 +
                             activation_factor * 0.2 +
                             retrieval_strength * 0.3
                     ) * consolidation_boost

        return min(1.0, confidence)

    def consolidate_memories(self, similarity_threshold: Optional[float] = None) -> Dict[str, Any]:
        """Consolidate similar memories into higher-level concepts"""
        with self.lock:
            if similarity_threshold is None:
                similarity_threshold = self.config.consolidation_threshold

            if self.associative_index.embedding_matrix is None:
                return {'consolidated_count': 0, 'clusters_found': 0}

            # Cluster memories using DBSCAN
            embeddings_np = self.associative_index.embedding_matrix.cpu().numpy()

            # Use cosine distance for clustering
            from sklearn.metrics.pairwise import cosine_distances
            distance_matrix = cosine_distances(embeddings_np)

            clustering = DBSCAN(
                eps=1 - similarity_threshold,  # Convert similarity to distance
                min_samples=self.config.min_cluster_size,
                metric='precomputed'
            ).fit(distance_matrix)

            # Process clusters
            unique_labels = set(clustering.labels_)
            unique_labels.discard(-1)  # Remove noise label

            consolidated_count = 0
            for label in unique_labels:
                cluster_indices = np.where(clustering.labels_ == label)[0]

                if len(cluster_indices) >= self.config.min_cluster_size:
                    # Create consolidated memory for this cluster
                    consolidated = self._create_consolidated_memory(cluster_indices)
                    if consolidated:
                        self.associative_index.add_trace(
                            consolidated['embedding'],
                            consolidated['content'],
                            is_consolidated=True,
                            source_traces=cluster_indices.tolist()
                        )
                        consolidated_count += 1

            self.last_consolidation = time.time()
            self.stats['consolidations'] += 1

            return {
                'consolidated_count': consolidated_count,
                'clusters_found': len(unique_labels),
                'total_memories': len(self.associative_index.traces)
            }

    def _create_consolidated_memory(self, cluster_indices: np.ndarray) -> Optional[Dict[str, Any]]:
        """Create a consolidated memory from a cluster of similar memories"""
        if len(cluster_indices) == 0:
            return None

        traces = [self.associative_index.traces[i] for i in cluster_indices
                  if i < len(self.associative_index.traces)]

        if not traces:
            return None

        # Average embeddings
        embeddings = torch.stack([t.embedding for t in traces])
        consolidated_embedding = torch.mean(embeddings, dim=0)
        consolidated_embedding = F.normalize(consolidated_embedding, p=2, dim=0)

        # Extract common themes from content
        all_keywords = set()
        for trace in traces:
            all_keywords.update(trace.keywords)

        # Find most common keywords
        keyword_counts = Counter()
        for trace in traces:
            keyword_counts.update(trace.keywords)

        top_keywords = [kw for kw, _ in keyword_counts.most_common(10)]

        # Create summary content
        summary_parts = [
            f"[CONSOLIDATED from {len(traces)} memories]",
            f"Common themes: {', '.join(top_keywords[:5])}",
            f"Time range: {min(t.timestamp for t in traces):.0f} - {max(t.timestamp for t in traces):.0f}",
            f"Sample content: {traces[0].content[:100]}..."
        ]

        consolidated_content = " | ".join(summary_parts)

        return {
            'embedding': consolidated_embedding,
            'content': consolidated_content,
            'source_count': len(traces)
        }

    def get_memory_summary(self) -> Dict[str, Any]:
        """Get comprehensive memory summary with enhanced statistics"""
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
                        'content': t.content[:100],
                        'activations': t.activation_count,
                        'is_consolidated': t.is_consolidated
                    }
                    for t in top_memories
                ]

                # Count consolidated memories
                consolidated_count = sum(
                    1 for t in self.associative_index.traces
                    if t.is_consolidated
                )
            else:
                most_activated = []
                consolidated_count = 0

            # Get pattern statistics
            if self.pattern_bank.patterns:
                avg_pattern_importance = np.mean([
                    self.pattern_bank._calculate_pattern_importance(p)
                    for p in self.pattern_bank.patterns
                ])
            else:
                avg_pattern_importance = 0.0

            return {
                'total_memories': len(self.associative_index.traces),
                'consolidated_memories': consolidated_count,
                'unique_patterns': len(self.pattern_bank.patterns),
                'avg_pattern_importance': avg_pattern_importance,
                'memory_diversity': diversity,
                'total_updates': self.stats['total_updates'],
                'total_consolidations': self.stats['consolidations'],
                'most_activated': most_activated,
                'context_stability': {
                    'short_term': torch.norm(self.context.short_term).item(),
                    'medium_term': torch.norm(self.context.medium_term).item(),
                    'long_term': torch.norm(self.context.long_term).item()
                },
                'keyword_index_size': len(self.associative_index.keyword_index),
                'retrieval_stats': {
                    'keyword_hits': self.stats['keyword_hits'],
                    'semantic_hits': self.stats['semantic_hits']
                }
            }

    def save(self, filepath: Union[str, Path]) -> None:
        """Save the complete enhanced memory state to disk"""
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        with self.lock:
            # Prepare tensor state dict
            tensor_state = {
                'running_mean': self.running_mean.cpu(),
                'running_covariance': self.running_covariance.cpu(),
                'attention_matrix': self.attention_matrix.cpu(),
                'semantic_memory': self.semantic_memory.cpu(),
                'temporal_weights': self.temporal_weights.cpu(),
                # Multi-scale context
                'context_short_term': self.context.short_term.cpu(),
                'context_medium_term': self.context.medium_term.cpu(),
                'context_long_term': self.context.long_term.cpu(),
            }

            # Save tensors using PyTorch
            torch.save(tensor_state, f"{filepath}_tensors.pt")

            # Prepare non-tensor state
            traces_data = []
            for trace in self.associative_index.traces:
                traces_data.append({
                    'embedding': trace.embedding.cpu().numpy(),
                    'content': trace.content,
                    'timestamp': trace.timestamp,
                    'activation_count': trace.activation_count,
                    'last_activation': trace.last_activation,
                    'associations': trace.associations,
                    'keywords': list(trace.keywords),
                    'confidence': trace.confidence,
                    'is_consolidated': trace.is_consolidated,
                    'source_traces': trace.source_traces
                })

            # Convert pattern bank
            patterns_data = []
            for pattern_info in self.pattern_bank.patterns:
                patterns_data.append({
                    'pattern': pattern_info['pattern'].cpu().numpy(),
                    'timestamp': pattern_info['timestamp'],
                    'strength': pattern_info['strength'],
                    'importance': pattern_info['importance'],
                    'activation_count': pattern_info['activation_count'],
                    'metadata': pattern_info.get('metadata', {})
                })

            # Complete state
            state = {
                'config': vars(self.config),
                'associative_index': {
                    'traces': traces_data,
                    'content_hash': self.associative_index.content_hash,
                    'keyword_index': {k: list(v) for k, v in self.associative_index.keyword_index.items()}
                },
                'pattern_bank': {
                    'patterns': patterns_data,
                    'max_patterns': self.pattern_bank.max_patterns
                },
                'stats': self.stats.copy(),
                'last_consolidation': self.last_consolidation,
                'consolidation_interval': self.consolidation_interval,
                'context_momentums': {
                    'short': self.context.short_momentum,
                    'medium': self.context.medium_momentum,
                    'long': self.context.long_momentum
                }
            }

            # Save non-tensor state
            with open(f"{filepath}_state.pkl", 'wb') as f:
                pickle.dump(state, f, protocol=pickle.HIGHEST_PROTOCOL)

            # Save metadata
            metadata = {
                'version': '2.0',  # Updated version
                'save_time': time.time(),
                'total_memories': len(self.associative_index.traces),
                'consolidated_memories': sum(1 for t in self.associative_index.traces if t.is_consolidated),
                'total_patterns': len(self.pattern_bank.patterns),
                'device': str(self.device),
                'dtype': str(self.dtype),
            }

            with open(f"{filepath}_metadata.json", 'w') as f:
                json.dump(metadata, f, indent=2)

            print(f"Enhanced memory saved to {filepath} (3 files created)")

    def load(self, filepath: Union[str, Path]) -> None:
        """Load enhanced memory state from disk"""
        filepath = Path(filepath)

        with self.lock:
            # Load metadata
            try:
                with open(f"{filepath}_metadata.json", 'r') as f:
                    metadata = json.load(f)
                    print(f"Loading memory v{metadata['version']} saved at {metadata['save_time']}")
            except FileNotFoundError:
                print("Warning: No metadata file found, loading anyway...")

            # Load tensor state
            tensor_state = torch.load(f"{filepath}_tensors.pt", map_location=self.device)

            # Restore tensors
            self.running_mean = tensor_state['running_mean'].to(self.device, dtype=self.dtype)
            self.running_covariance = tensor_state['running_covariance'].to(self.device, dtype=self.dtype)
            self.attention_matrix = tensor_state['attention_matrix'].to(self.device, dtype=self.dtype)
            self.semantic_memory = tensor_state['semantic_memory'].to(self.device, dtype=self.dtype)
            self.temporal_weights = tensor_state['temporal_weights'].to(self.device, dtype=self.dtype)

            # Restore multi-scale context
            if 'context_short_term' in tensor_state:
                self.context.short_term = tensor_state['context_short_term'].to(self.device, dtype=self.dtype)
                self.context.medium_term = tensor_state['context_medium_term'].to(self.device, dtype=self.dtype)
                self.context.long_term = tensor_state['context_long_term'].to(self.device, dtype=self.dtype)

            # Load non-tensor state
            with open(f"{filepath}_state.pkl", 'rb') as f:
                state = pickle.load(f)

            # Restore configuration
            for key, value in state['config'].items():
                if hasattr(self.config, key):
                    setattr(self.config, key, value)

            # Restore associative index
            self.associative_index = EnhancedAssociativeIndex(self.config)
            self.associative_index.content_hash = state['associative_index']['content_hash']

            # Restore keyword index
            if 'keyword_index' in state['associative_index']:
                self.associative_index.keyword_index = {
                    k: set(v) for k, v in state['associative_index']['keyword_index'].items()
                }

            # Restore traces
            for trace_data in state['associative_index']['traces']:
                trace = MemoryTrace(
                    embedding=torch.from_numpy(trace_data['embedding']).to(self.device, dtype=self.dtype),
                    content=trace_data['content'],
                    timestamp=trace_data['timestamp'],
                    activation_count=trace_data['activation_count'],
                    last_activation=trace_data['last_activation'],
                    associations=trace_data['associations'],
                    keywords=set(trace_data.get('keywords', [])),
                    confidence=trace_data.get('confidence', 1.0),
                    is_consolidated=trace_data.get('is_consolidated', False),
                    source_traces=trace_data.get('source_traces', [])
                )
                self.associative_index.traces.append(trace)

            # Rebuild embedding matrix
            self.associative_index._rebuild_embedding_matrix()

            # Restore pattern bank
            self.pattern_bank = PatternBank(self.config)
            if 'pattern_bank' in state:
                for pattern_data in state['pattern_bank']['patterns']:
                    pattern_info = {
                        'pattern': torch.from_numpy(pattern_data['pattern']).to(self.device, dtype=self.dtype),
                        'timestamp': pattern_data['timestamp'],
                        'strength': pattern_data['strength'],
                        'importance': pattern_data.get('importance', 0.5),
                        'activation_count': pattern_data.get('activation_count', 0),
                        'metadata': pattern_data.get('metadata', {})
                    }
                    self.pattern_bank.patterns.append(pattern_info)

                self.pattern_bank._rebuild_embeddings()

                if 'max_patterns' in state['pattern_bank']:
                    self.pattern_bank.max_patterns = state['pattern_bank']['max_patterns']

            # Restore other state
            self.stats = state['stats']
            self.last_consolidation = state.get('last_consolidation', time.time())
            self.consolidation_interval = state.get('consolidation_interval', 3600)

            # Restore context momentums if available
            if 'context_momentums' in state:
                self.context.short_momentum = state['context_momentums']['short']
                self.context.medium_momentum = state['context_momentums']['medium']
                self.context.long_momentum = state['context_momentums']['long']

            print(f"Enhanced memory loaded: {len(self.associative_index.traces)} traces, "
                  f"{len(self.pattern_bank.patterns)} patterns")