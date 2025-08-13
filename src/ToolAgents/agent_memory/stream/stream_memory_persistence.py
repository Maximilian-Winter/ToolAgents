"""
STREAM Memory Persistence Extension
Adds save/load functionality to the StreamMemory class
"""

import pickle
import json
from pathlib import Path
from typing import Union, Dict, Any
import torch


class StreamMemoryPersistence:
    """
    Mixin class for adding persistence to StreamMemory
    Add this to your StreamMemory class through inheritance
    """

    def save(self, filepath: Union[str, Path], compress: bool = True) -> None:
        """
        Save the complete memory state to disk

        Args:
            filepath: Path to save the memory state
            compress: Whether to use compression (reduces file size but slower)
        """
        filepath = Path(filepath)

        with self.lock:
            # Prepare state dictionary
            state = {
                'version': '1.0',  # Version for backward compatibility
                'config': self._serialize_config(),
                'tensors': self._gather_tensors(),
                'traces': self._serialize_traces(),
                'patterns': self._serialize_patterns(),
                'statistics': self.stats.copy(),
                'metadata': {
                    'save_time': time.time(),
                    'total_memories': len(self.associative_index.traces),
                    'device': str(self.device),
                    'dtype': str(self.dtype)
                }
            }

            # Save using torch's built-in serialization
            if compress:
                torch.save(state, filepath, pickle_protocol=pickle.HIGHEST_PROTOCOL)
            else:
                torch.save(state, filepath, pickle_protocol=pickle.HIGHEST_PROTOCOL,
                          _use_new_zipfile_serialization=False)

            print(f"✓ Memory saved to {filepath}")
            print(f"  - {len(self.associative_index.traces)} memory traces")
            print(f"  - {len(self.pattern_bank)} patterns")
            print(f"  - File size: {filepath.stat().st_size / 1024 / 1024:.2f} MB")

    def load(self, filepath: Union[str, Path], map_location: Optional[str] = None) -> None:
        """
        Load memory state from disk

        Args:
            filepath: Path to load the memory state from
            map_location: Device to load tensors to (None = same as saved)
        """
        filepath = Path(filepath)

        if not filepath.exists():
            raise FileNotFoundError(f"Memory file not found: {filepath}")

        with self.lock:
            # Load state
            state = torch.load(filepath, map_location=map_location or self.device)

            # Version check for compatibility
            version = state.get('version', '1.0')
            if version != '1.0':
                print(f"Warning: Loading memory from version {version}, current version is 1.0")

            # Restore configuration
            self._restore_config(state['config'])

            # Restore tensors
            self._restore_tensors(state['tensors'])

            # Restore traces
            self._restore_traces(state['traces'])

            # Restore patterns
            self._restore_patterns(state['patterns'])

            # Restore statistics
            self.stats = state['statistics']

            # Rebuild embedding matrix
            self.associative_index._rebuild_embedding_matrix()

            print(f"✓ Memory loaded from {filepath}")
            print(f"  - {len(self.associative_index.traces)} memory traces restored")
            print(f"  - {len(self.pattern_bank)} patterns restored")
            print(f"  - Device: {self.device}")

    def _serialize_config(self) -> Dict[str, Any]:
        """Serialize configuration to dictionary"""
        return {
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
            'similarity_threshold': self.config.similarity_threshold
        }

    def _restore_config(self, config_dict: Dict[str, Any]) -> None:
        """Restore configuration from dictionary"""
        for key, value in config_dict.items():
            if hasattr(self.config, key):
                setattr(self.config, key, value)

    def _gather_tensors(self) -> Dict[str, torch.Tensor]:
        """Gather all tensors for saving"""
        tensors = {
            'running_mean': self.running_mean.cpu(),
            'running_covariance': self.running_covariance.cpu(),
            'context_state': self.context_state.cpu(),
            'attention_matrix': self.attention_matrix.cpu(),
            'semantic_memory': self.semantic_memory.cpu(),
            'temporal_weights': self.temporal_weights.cpu(),
            'context_momentum_value': torch.tensor(self.context_momentum)
        }
        return tensors

    def _restore_tensors(self, tensors: Dict[str, torch.Tensor]) -> None:
        """Restore tensors from saved state"""
        self.running_mean = tensors['running_mean'].to(self.device, dtype=self.dtype)
        self.running_covariance = tensors['running_covariance'].to(self.device, dtype=self.dtype)
        self.context_state = tensors['context_state'].to(self.device, dtype=self.dtype)
        self.attention_matrix = tensors['attention_matrix'].to(self.device, dtype=self.dtype)
        self.semantic_memory = tensors['semantic_memory'].to(self.device, dtype=self.dtype)
        self.temporal_weights = tensors['temporal_weights'].to(self.device, dtype=self.dtype)
        self.context_momentum = tensors['context_momentum_value'].item()

    def _serialize_traces(self) -> List[Dict[str, Any]]:
        """Serialize memory traces for saving"""
        serialized = []
        for trace in self.associative_index.traces:
            serialized.append({
                'embedding': trace.embedding.cpu(),  # Move to CPU for saving
                'content': trace.content,
                'timestamp': trace.timestamp,
                'activation_count': trace.activation_count,
                'last_activation': trace.last_activation,
                'associations': trace.associations
            })

        # Also save the content hash mapping
        return {
            'traces': serialized,
            'content_hash': self.associative_index.content_hash.copy()
        }

    def _restore_traces(self, traces_data: Dict[str, Any]) -> None:
        """Restore memory traces from saved state"""
        # Clear existing traces
        self.associative_index.traces.clear()
        self.associative_index.content_hash.clear()

        # Restore each trace
        for trace_dict in traces_data['traces']:
            trace = MemoryTrace(
                embedding=trace_dict['embedding'].to(self.device, dtype=self.dtype),
                content=trace_dict['content'],
                timestamp=trace_dict['timestamp'],
                activation_count=trace_dict['activation_count'],
                last_activation=trace_dict['last_activation'],
                associations=trace_dict['associations']
            )
            self.associative_index.traces.append(trace)

        # Restore content hash mapping
        self.associative_index.content_hash = traces_data['content_hash'].copy()

    def _serialize_patterns(self) -> List[Dict[str, Any]]:
        """Serialize pattern bank for saving"""
        serialized = []
        for pattern_info in self.pattern_bank:
            serialized.append({
                'pattern': pattern_info['pattern'].cpu(),
                'timestamp': pattern_info['timestamp'],
                'strength': pattern_info['strength']
            })
        return serialized

    def _restore_patterns(self, patterns: List[Dict[str, Any]]) -> None:
        """Restore pattern bank from saved state"""
        self.pattern_bank = []
        for pattern_dict in patterns:
            self.pattern_bank.append({
                'pattern': pattern_dict['pattern'].to(self.device, dtype=self.dtype),
                'timestamp': pattern_dict['timestamp'],
                'strength': pattern_dict['strength']
            })

    def export_to_json(self, filepath: Union[str, Path]) -> None:
        """
        Export memory content to human-readable JSON
        (Note: This doesn't include embeddings, just the text content)
        """
        filepath = Path(filepath)

        with self.lock:
            export_data = {
                'metadata': {
                    'export_time': time.time(),
                    'total_memories': len(self.associative_index.traces),
                    'total_patterns': len(self.pattern_bank),
                    'memory_diversity': self.get_memory_summary()['memory_diversity']
                },
                'memories': [],
                'statistics': self.stats
            }

            # Export memory traces (without embeddings)
            for trace in self.associative_index.traces:
                export_data['memories'].append({
                    'content': trace.content,
                    'timestamp': trace.timestamp,
                    'activation_count': trace.activation_count,
                    'last_activation': trace.last_activation,
                    'num_associations': len(trace.associations)
                })

            # Sort by activation count for readability
            export_data['memories'].sort(key=lambda x: x['activation_count'], reverse=True)

            # Write to JSON
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(export_data, f, indent=2, ensure_ascii=False)

            print(f"✓ Memory exported to JSON: {filepath}")
            print(f"  - {len(export_data['memories'])} memories exported")


# Enhanced StreamMemory class with persistence
class StreamMemoryWithPersistence(StreamMemory, StreamMemoryPersistence):
    """
    StreamMemory with added save/load functionality
    Just inherit from both classes to get all features
    """
    pass


# Convenience functions for checkpoint management
class MemoryCheckpointManager:
    """
    Manages automatic checkpointing of memory
    """

    def __init__(self, memory: StreamMemoryWithPersistence,
                 checkpoint_dir: Union[str, Path],
                 max_checkpoints: int = 5):
        self.memory = memory
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.max_checkpoints = max_checkpoints

    def save_checkpoint(self, name: Optional[str] = None) -> Path:
        """Save a checkpoint with automatic naming"""
        if name is None:
            name = f"checkpoint_{int(time.time())}"

        filepath = self.checkpoint_dir / f"{name}.pt"
        self.memory.save(filepath)

        # Clean up old checkpoints
        self._cleanup_old_checkpoints()

        return filepath

    def load_latest_checkpoint(self) -> bool:
        """Load the most recent checkpoint"""
        checkpoints = list(self.checkpoint_dir.glob("checkpoint_*.pt"))
        if not checkpoints:
            return False

        latest = max(checkpoints, key=lambda p: p.stat().st_mtime)
        self.memory.load(latest)
        return True

    def _cleanup_old_checkpoints(self):
        """Remove old checkpoints beyond max_checkpoints"""
        checkpoints = sorted(
            self.checkpoint_dir.glob("checkpoint_*.pt"),
            key=lambda p: p.stat().st_mtime
        )

        if len(checkpoints) > self.max_checkpoints:
            for checkpoint in checkpoints[:-self.max_checkpoints]:
                checkpoint.unlink()
                print(f"  - Removed old checkpoint: {checkpoint.name}")


# Example usage
if __name__ == "__main__":
    from sentence_transformers import SentenceTransformer

    class SimpleEmbeddingProvider(EmbeddingProvider):
        def __init__(self):
            self.model = SentenceTransformer('all-MiniLM-L6-v2')

        def get_embedding(self, texts: list[str]) -> EmbeddingResult:
            embeddings = self.model.encode(texts, convert_to_numpy=True)
            return EmbeddingResult(embeddings=embeddings)

    # Create memory with persistence
    provider = SimpleEmbeddingProvider()
    memory = StreamMemoryWithPersistence(provider)

    # Process some data
    memory.process_input("The capital of France is Paris")
    memory.process_input("Machine learning is a subset of AI")

    # Save memory
    memory.save("my_memory.pt")

    # Create new memory and load
    new_memory = StreamMemoryWithPersistence(provider)
    new_memory.load("my_memory.pt")

    # Verify it works
    result = new_memory.recall("What's the capital of France?")
    print(f"Recalled: {result['memories'][0]['content'] if result['memories'] else 'Nothing'}")

    # Export to JSON for inspection
    memory.export_to_json("memory_export.json")

    # Use checkpoint manager for automatic saves
    checkpoint_mgr = MemoryCheckpointManager(memory, "./checkpoints")
    checkpoint_mgr.save_checkpoint()  # Auto-named checkpoint
    checkpoint_mgr.save_checkpoint("manual_checkpoint")  # Named checkpoint