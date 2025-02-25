"""
This module implements a hierarchical semantic memory system that extends the base
semantic memory with hierarchical organization, parent-child relationships, and
improved retrieval based on memory structure.
"""

import abc
import dataclasses
import enum
import json
import uuid
from datetime import datetime
from typing import List, Dict, Optional, Any, Union, Set, Tuple

import chromadb
from chromadb.api.types import IncludeEnum
import numpy as np

from ToolAgents.agent_memory.semantic_memory.memory import (
    SemanticMemory, 
    SemanticMemoryConfig,
    ExtractPatternStrategy,
    SimpleExtractPatternStrategy
)


class HierarchicalRelationshipType(enum.Enum):
    """
    Defines the types of relationships between memory nodes.
    """
    PARENT_CHILD = "parent_child"  # Direct parent-child relationship
    TEMPORAL = "temporal"  # Temporal sequence relationship
    CAUSAL = "causal"  # Cause-effect relationship
    THEMATIC = "thematic"  # Thematic or conceptual relationship
    SPATIAL = "spatial"  # Spatial or location-based relationship


@dataclasses.dataclass
class HierarchicalMemoryNode:
    """
    Represents a node in the hierarchical memory structure.
    
    A node can contain both content and references to child nodes, allowing for
    nested memory organization.
    """
    node_id: str  # Unique identifier for the node
    content: str  # The content stored in this node
    metadata: Dict[str, Any]  # Metadata for this node
    parent_id: Optional[str] = None  # Parent node ID (None for root nodes)
    child_ids: List[str] = dataclasses.field(default_factory=list)  # Child node IDs
    embedding: Optional[List[float]] = None  # Embedding vector for the content
    created_timestamp: str = dataclasses.field(default_factory=lambda: datetime.now().isoformat())
    last_access_timestamp: str = dataclasses.field(default_factory=lambda: datetime.now().isoformat())
    access_count: int = 1


@dataclasses.dataclass
class HierarchicalMemoryConfig:
    """
    Configuration for the hierarchical memory system.
    
    Extends the existing SemanticMemoryConfig with hierarchy-specific parameters.
    """
    base_config: SemanticMemoryConfig = dataclasses.field(default_factory=SemanticMemoryConfig)
    
    # Hierarchy-specific configurations
    max_hierarchy_depth: int = 5  # Maximum nesting depth allowed
    default_summarization_threshold: int = 10  # Number of child nodes before auto-summarization
    auto_organize_threshold: int = 20  # Number of nodes in a level before auto-organization
    summarize_on_hierarchy_change: bool = True  # Whether to auto-summarize when hierarchy changes
    
    # Retrieval settings
    include_parent_context: bool = True  # Include parent nodes in retrieval
    include_child_context: bool = True  # Include child nodes in retrieval
    parent_weight: float = 0.3  # Weight for parent nodes in ranking
    child_weight: float = 0.5  # Weight for child nodes in ranking
    sibling_weight: float = 0.2  # Weight for sibling nodes in ranking


class HierarchicalSemanticMemory:
    """
    A semantic memory system that supports hierarchical organization of memories.
    
    Extends the SemanticMemory system with hierarchical storage and retrieval capabilities,
    allowing for more structured memory organization and context-aware recall.
    """
    
    def __init__(self, config: HierarchicalMemoryConfig = HierarchicalMemoryConfig()):
        """
        Initialize the hierarchical semantic memory system.
        
        Args:
            config: Configuration parameters for the hierarchical memory system.
        """
        # Initialize the base semantic memory with the base configuration
        self.semantic_memory = SemanticMemory(config.base_config)
        
        # Store the hierarchical configuration
        self.config = config
        
        # Initialize collections for hierarchical storage
        self.node_collection = self.semantic_memory.client.get_or_create_collection(
            name="hierarchical_nodes",
            metadata={"hnsw:space": "cosine"}
        )
        
        # Initialize the relationship collection to store node relationships
        self.relationship_collection = self.semantic_memory.client.get_or_create_collection(
            name="hierarchical_relationships",
            metadata={"hnsw:space": "cosine"}
        )
        
        # Create node cache for faster access
        self.node_cache = {}

    def store(self, content: str, parent_id: Optional[str] = None, 
              metadata: Optional[Dict] = None, 
              relationship_type: HierarchicalRelationshipType = HierarchicalRelationshipType.PARENT_CHILD,
              timestamp=datetime.now().isoformat()) -> str:
        """
        Store a new memory in the hierarchical structure.
        
        Args:
            content: The text content to store.
            parent_id: Optional ID of the parent node (None for root nodes).
            metadata: Additional contextual metadata.
            relationship_type: Type of relationship to the parent.
            timestamp: Storage timestamp.
            
        Returns:
            Unique memory identifier for the stored node.
        """
        # Create the new node
        node_id = self.create_node(content, parent_id, metadata, timestamp)
        
        # If parent exists, create relationship and update parent's children list
        if parent_id:
            self.create_relationship(node_id, parent_id, relationship_type)
            parent_node = self.get_node(parent_id)
            if parent_node:
                # Update parent's children list
                import json
                parent_node.child_ids.append(node_id)
                self._update_node_metadata(
                    parent_id, 
                    {"child_ids_json": json.dumps(parent_node.child_ids)}
                )
                
                # Check if auto-summarization is needed
                if (self.config.summarize_on_hierarchy_change and 
                    len(parent_node.child_ids) >= self.config.default_summarization_threshold):
                    self.summarize_node_with_children(parent_id)
        
        return node_id

    def recall(self, query: str, n_results: int = 5, 
               context_filter: Optional[Dict] = None,
               include_hierarchy: bool = True,
               max_context_depth: int = 2,
               current_date: datetime = datetime.now()) -> List[Dict]:
        """
        Recall memories with hierarchical context.
        
        Args:
            query: The query string used for search.
            n_results: Number of top results to return.
            context_filter: Optional filter criteria.
            include_hierarchy: Whether to include hierarchical context.
            max_context_depth: Maximum depth of hierarchy to include.
            current_date: Current date for scoring purposes.
            
        Returns:
            List of memory dictionaries with hierarchical context.
        """
        # First, get base results using direct similarity search
        base_results = self._direct_recall(query, n_results * 2, context_filter, current_date)
        
        # If hierarchy context is not needed, return base results directly
        if not include_hierarchy:
            return base_results[:n_results]
            
        # Enhance results with hierarchical context
        enhanced_results = []
        seen_node_ids = set()
        
        for result in base_results:
            try:
                node_id = result['metadata']['node_id']
                if node_id in seen_node_ids:
                    continue
                    
                seen_node_ids.add(node_id)
                node = self.get_node(node_id)
                
                # Skip if node can't be found
                if node is None:
                    continue
            except Exception as e:
                print(f"Error processing result: {str(e)}")
                continue
            
            # Add hierarchical context
            result['hierarchy'] = {
                'parents': self._get_parent_chain(node, max_depth=max_context_depth),
                'children': self._get_children_summary(node, max_depth=1)
            }
            
            # Adjust the ranking score based on hierarchical context
            if self.config.include_parent_context or self.config.include_child_context:
                self._adjust_score_with_hierarchy(result, query)
                
            enhanced_results.append(result)
            
            # Add relevant siblings, parents, or children as separate results if highly relevant
            if len(enhanced_results) < n_results:
                related_nodes = self._get_related_nodes(node, query, seen_node_ids)
                for related_node, similarity in related_nodes:
                    if len(enhanced_results) >= n_results:
                        break
                        
                    related_result = self._node_to_result(related_node, query, similarity, current_date)
                    related_result['hierarchy'] = {
                        'parents': self._get_parent_chain(related_node, max_depth=max_context_depth),
                        'children': self._get_children_summary(related_node, max_depth=1)
                    }
                    
                    enhanced_results.append(related_result)
                    seen_node_ids.add(related_node.node_id)
        
        # Sort enhanced results by rank score and limit to requested number
        enhanced_results.sort(key=lambda x: x['rank_score'], reverse=True)
        return enhanced_results[:n_results]

    def create_node(self, content: str, parent_id: Optional[str] = None,
                   metadata: Optional[Dict] = None,
                   timestamp=datetime.now().isoformat()) -> str:
        """
        Create a new node in the hierarchy.
        
        Args:
            content: The text content to store in the node.
            parent_id: Optional ID of the parent node.
            metadata: Additional contextual metadata.
            timestamp: Creation timestamp.
            
        Returns:
            The unique ID of the created node.
        """
        # Generate a unique node ID
        node_id = f"node_{uuid.uuid4()}"
        
        # Generate embedding for the content
        embedding = self._generate_embedding(content)
        
        # Prepare node metadata
        node_metadata = {
            "timestamp": timestamp,
            "last_access_timestamp": timestamp,
            "access_count": 1,
            "node_id": node_id,
            "type": "hierarchical_node",
            "child_ids_json": "[]"  # Store as JSON string instead of a list
        }
        
        # Only add parent_id to metadata if it's not None
        if parent_id is not None:
            node_metadata["parent_id"] = parent_id
        
        # Add any additional metadata
        if metadata:
            node_metadata.update(metadata)
        
        # Create the node object with empty child_ids list (we store as JSON in metadata)
        node = HierarchicalMemoryNode(
            node_id=node_id,
            content=content,
            metadata=node_metadata,
            parent_id=parent_id,
            child_ids=[],  # Start with empty list in the object
            embedding=embedding
        )
        
        # Store the node in the collection
        self.node_collection.add(
            documents=[content],
            metadatas=[node_metadata],
            embeddings=[embedding],
            ids=[node_id]
        )
        
        # Cache the node for faster access
        self.node_cache[node_id] = node
        
        return node_id

    def update_node(self, node_id: str, content: Optional[str] = None,
                   metadata: Optional[Dict] = None) -> bool:
        """
        Update an existing node's content or metadata.
        
        Args:
            node_id: The ID of the node to update.
            content: New content for the node (None to keep existing).
            metadata: New metadata to merge with existing metadata.
            
        Returns:
            True if the update was successful, False otherwise.
        """
        # Get the current node
        node = self.get_node(node_id)
        if not node:
            return False
        
        # Update content if provided
        if content is not None:
            node.content = content
            node.embedding = self._generate_embedding(content)
            
            # Update the node in the collection
            self.node_collection.update(
                ids=[node_id],
                documents=[content],
                embeddings=[node.embedding]
            )
        
        # Update metadata if provided
        if metadata:
            node.metadata.update(metadata)
            node.metadata["last_access_timestamp"] = datetime.now().isoformat()
            
            # Update the node metadata in the collection
            self.node_collection.update(
                ids=[node_id],
                metadatas=[node.metadata]
            )
        
        # Update cache
        self.node_cache[node_id] = node
        
        return True

    def delete_node(self, node_id: str, recursive: bool = False) -> bool:
        """
        Delete a node and optionally its children.
        
        Args:
            node_id: The ID of the node to delete.
            recursive: If True, also delete all child nodes.
            
        Returns:
            True if deletion was successful, False otherwise.
        """
        # Get the node to delete
        node = self.get_node(node_id)
        if not node:
            return False
        
        # If recursive, delete all children first
        if recursive and node.child_ids:
            for child_id in node.child_ids.copy():
                self.delete_node(child_id, recursive=True)
        
        # Update parent's child_ids list if parent exists
        if node.parent_id:
            parent_node = self.get_node(node.parent_id)
            if parent_node and node_id in parent_node.child_ids:
                import json
                parent_node.child_ids.remove(node_id)
                self._update_node_metadata(
                    parent_node.node_id, 
                    {"child_ids_json": json.dumps(parent_node.child_ids)}
                )
        
        # Delete relationships
        self._delete_node_relationships(node_id)
        
        # Delete the node from the collection
        self.node_collection.delete(ids=[node_id])
        
        # Remove from cache
        if node_id in self.node_cache:
            del self.node_cache[node_id]
        
        return True

    def move_node(self, node_id: str, new_parent_id: Optional[str] = None) -> bool:
        """
        Move a node to a new parent.
        
        Args:
            node_id: The ID of the node to move.
            new_parent_id: The ID of the new parent node (None for root).
            
        Returns:
            True if the move was successful, False otherwise.
        """
        node = self.get_node(node_id)
        if not node:
            return False
            
        # Check for circular reference
        if new_parent_id:
            if new_parent_id == node_id:
                return False
                
            # Check that new_parent_id is not a descendant of node_id
            if self._is_descendant(node_id, new_parent_id):
                return False
                
            # Verify new parent exists
            new_parent = self.get_node(new_parent_id)
            if not new_parent:
                return False
        
        # Remove from old parent's children list
        if node.parent_id:
            old_parent = self.get_node(node.parent_id)
            if old_parent and node_id in old_parent.child_ids:
                import json
                old_parent.child_ids.remove(node_id)
                self._update_node_metadata(
                    old_parent.node_id, 
                    {"child_ids_json": json.dumps(old_parent.child_ids)}
                )
        
        # Add to new parent's children list
        if new_parent_id:
            import json
            new_parent = self.get_node(new_parent_id)
            new_parent.child_ids.append(node_id)
            self._update_node_metadata(
                new_parent_id, 
                {"child_ids_json": json.dumps(new_parent.child_ids)}
            )
        
        # Update node's parent_id
        node.parent_id = new_parent_id
        self._update_node_metadata(node_id, {"parent_id": new_parent_id})
        
        # Update relationships
        self._delete_node_relationships(node_id)
        if new_parent_id:
            self.create_relationship(
                node_id, 
                new_parent_id, 
                HierarchicalRelationshipType.PARENT_CHILD
            )
        
        return True

    def get_node(self, node_id: str) -> Optional[HierarchicalMemoryNode]:
        """
        Retrieve a specific node by ID.
        
        Args:
            node_id: The ID of the node to retrieve.
            
        Returns:
            The node object if found, None otherwise.
        """
        # Check cache first
        if node_id in self.node_cache:
            return self.node_cache[node_id]
        
        # Query the node collection
        try:
            result = self.node_collection.get(
                ids=[node_id],
                include=[IncludeEnum.metadatas, IncludeEnum.embeddings, IncludeEnum.documents]
            )
            
            if not result['ids'] or not result['metadatas']:
                return None
            
            # Create and cache the node
            # Convert child_ids from JSON string back to list
            child_ids_json = result['metadatas'][0].get('child_ids_json', '[]')
            try:
                import json
                child_ids = json.loads(child_ids_json)
            except:
                child_ids = []
                
            node = HierarchicalMemoryNode(
                node_id=node_id,
                content=result['documents'][0],
                metadata=result['metadatas'][0],
                parent_id=result['metadatas'][0].get('parent_id'),
                child_ids=child_ids,
                embedding=result['embeddings'][0] if result['embeddings'] else None,
                created_timestamp=result['metadatas'][0].get('timestamp'),
                last_access_timestamp=result['metadatas'][0].get('last_access_timestamp'),
                access_count=result['metadatas'][0].get('access_count', 1)
            )
            
            self.node_cache[node_id] = node
            return node
            
        except Exception as e:
            print(f"Error retrieving node {node_id}: {str(e)}")
            return None

    def get_children(self, node_id: str) -> List[HierarchicalMemoryNode]:
        """
        Get all children of a specific node.
        
        Args:
            node_id: The ID of the parent node.
            
        Returns:
            A list of child node objects.
        """
        node = self.get_node(node_id)
        if not node or not node.child_ids:
            return []
            
        children = []
        for child_id in node.child_ids:
            child_node = self.get_node(child_id)
            if child_node:
                children.append(child_node)
                
        return children

    def get_ancestors(self, node_id: str, max_depth: int = -1) -> List[HierarchicalMemoryNode]:
        """
        Get all ancestors of a node up to a specified depth.
        
        Args:
            node_id: The ID of the node.
            max_depth: Maximum depth to traverse upward (-1 for unlimited).
            
        Returns:
            A list of ancestor nodes, starting with the immediate parent.
        """
        node = self.get_node(node_id)
        if not node or not node.parent_id:
            return []
            
        ancestors = []
        current_id = node.parent_id
        depth = 0
        
        while current_id and (max_depth < 0 or depth < max_depth):
            parent_node = self.get_node(current_id)
            if not parent_node:
                break
                
            ancestors.append(parent_node)
            current_id = parent_node.parent_id
            depth += 1
            
        return ancestors

    def create_relationship(self, source_id: str, target_id: str,
                           relationship_type: HierarchicalRelationshipType) -> str:
        """
        Create a relationship between two nodes.
        
        Args:
            source_id: The ID of the source node.
            target_id: The ID of the target node.
            relationship_type: The type of relationship.
            
        Returns:
            The ID of the created relationship.
        """
        relationship_id = f"rel_{uuid.uuid4()}"
        
        # Create embedding for the relationship (combine source and target embeddings)
        source_node = self.get_node(source_id)
        target_node = self.get_node(target_id)
        
        if not source_node or not target_node:
            return None
            
        # Create a combined embedding for the relationship
        if source_node.embedding and target_node.embedding:
            relationship_embedding = self._combine_embeddings(
                source_node.embedding, 
                target_node.embedding
            )
        else:
            # If embeddings are missing, generate from relationship description
            relationship_text = f"Relationship from {source_id} to {target_id}: {relationship_type.value}"
            relationship_embedding = self._generate_embedding(relationship_text)
        
        # Store relationship metadata
        relationship_metadata = {
            "source_id": source_id,
            "target_id": target_id,
            "relationship_type": relationship_type.value,
            "timestamp": datetime.now().isoformat()
        }
        
        # Store the relationship in the collection
        self.relationship_collection.add(
            ids=[relationship_id],
            documents=[f"Relationship: {relationship_type.value}"],
            metadatas=[relationship_metadata],
            embeddings=[relationship_embedding]
        )
        
        return relationship_id

    def auto_organize(self, node_ids: List[str] = None) -> List[str]:
        """
        Automatically organize a set of nodes into a hierarchy.
        
        Args:
            node_ids: List of node IDs to organize (None to organize all root nodes).
            
        Returns:
            List of top-level node IDs in the created hierarchy.
        """
        # If no specific nodes provided, get all root nodes
        if node_ids is None:
            try:
                results = self.node_collection.get(
                    where={"parent_id": {"$exists": False}},  # Match documents where parent_id field doesn't exist
                    include=[IncludeEnum.metadatas]
                )
                if not results['ids'] or len(results['ids']) == 0:
                    return []
                node_ids = results['ids']
            except Exception as e:
                print(f"Error getting root nodes for auto-organize: {str(e)}")
                return []
        
        # If fewer than threshold nodes, no need to organize
        if len(node_ids) < self.config.auto_organize_threshold:
            return node_ids
            
        # Get all nodes and their embeddings
        nodes = []
        embeddings = []
        for node_id in node_ids:
            node = self.get_node(node_id)
            if node:
                nodes.append(node)
                if node.embedding:
                    embeddings.append(node.embedding)
                else:
                    embeddings.append(self._generate_embedding(node.content))
        
        # Use semantic memory's clustering strategy to group nodes
        clusters = self.semantic_memory._cluster_embeddings(
            embeddings, 
            self.config.base_config.minimum_cluster_similarity
        )
        
        # For each cluster, create a parent node summarizing the cluster
        new_parent_ids = []
        for cluster_idx, cluster in enumerate(clusters):
            if len(cluster) < 2:  # Skip singleton clusters
                if cluster:  # Add the single node to the result list
                    new_parent_ids.append(nodes[cluster[0]].node_id)
                continue
                
            # Create a summary for this cluster
            cluster_nodes = [nodes[i] for i in cluster]
            summary = self._generate_cluster_summary(cluster_nodes)
            
            # Create a parent node with the summary
            parent_id = self.create_node(
                content=summary,
                metadata={
                    "type": "auto_organized_node",
                    "cluster_size": len(cluster)
                }
            )
            
            # Add all cluster nodes as children of the new parent
            for i in cluster:
                node_id = nodes[i].node_id
                self.move_node(node_id, parent_id)
                
            new_parent_ids.append(parent_id)
            
        return new_parent_ids

    def summarize_node_with_children(self, node_id: str) -> Optional[str]:
        """
        Generate a summary of a node and its children.
        
        Args:
            node_id: The ID of the node to summarize.
            
        Returns:
            The generated summary text, or None if the operation failed.
        """
        try:
            node = self.get_node(node_id)
            if not node:
                print(f"Cannot summarize: Node {node_id} not found")
                return None
                
            children = self.get_children(node_id)
            if not children:
                return node.content
            
            # Generate a summary from node content and children
            all_content = [node.content] + [child.content for child in children]
            
            # Use the extract pattern strategy from semantic memory to generate a summary
            try:
                # Check if we're using the simple strategy - need to compare function identity
                extract_fn = self.semantic_memory._extract_pattern
                simple_extract_fn = SimpleExtractPatternStrategy().extract_pattern
                
                # Compare the function name since we can't directly compare function objects
                if extract_fn.__qualname__ == simple_extract_fn.__qualname__:
                    # If using the simple strategy, just concatenate content
                    summary = "\n\n".join(all_content)
                else:
                    # Use the more sophisticated extraction strategy if available
                    pattern_result = self.semantic_memory._extract_pattern(
                        f"summary_{node_id}",
                        all_content,
                        [node.metadata] + [child.metadata for child in children]
                    )
                    summary = pattern_result["content"]
                
                # Update the node with the new summary
                self.update_node(node_id, content=summary)
                
                return summary
            except Exception as e:
                print(f"Error generating summary: {str(e)}")
                return node.content  # Fall back to original content
        except Exception as e:
            print(f"Error in summarize_node_with_children: {str(e)}")
            return None

    def get_stats(self) -> Dict:
        """
        Get statistics about the current state of the memory system.
        
        Returns:
            A dictionary with counts and hierarchy statistics.
        """
        # Get basic counts
        node_count = self.node_collection.count()
        relationship_count = self.relationship_collection.count()
        
        # Count root nodes
        try:
            root_results = self.node_collection.get(
                where={"parent_id": {"$exists": False}},  # Match documents where parent_id field doesn't exist
                include=[IncludeEnum.metadatas]
            )
            root_count = len(root_results['ids']) if root_results['ids'] else 0
        except Exception as e:
            print(f"Error getting root nodes: {str(e)}")
            root_count = 0
            root_results = {"ids": []}
        
        # Calculate max depth (expensive operation, sample-based)
        max_depth = 0
        if root_count > 0:
            # Sample up to 10 root nodes to estimate max depth
            sample_size = min(10, root_count)
            for i in range(sample_size):
                depth = self._get_max_depth(root_results['ids'][i])
                max_depth = max(max_depth, depth)
        
        return {
            "node_count": node_count,
            "relationship_count": relationship_count,
            "root_node_count": root_count,
            "estimated_max_depth": max_depth,
            "cached_nodes": len(self.node_cache)
        }

    def clear_cache(self):
        """Clear the node cache to free memory."""
        self.node_cache.clear()

    # Helper methods
    def _direct_recall(self, query: str, n_results: int, context_filter: Optional[Dict], current_date: datetime) -> List[Dict]:
        """Perform a direct similarity search without hierarchical context."""
        query_embedding = self._generate_embedding(query)
        
        # Handle the case where no nodes exist yet
        if self.node_collection.count() == 0:
            return []
            
        try:
            results = self.node_collection.query(
                query_embeddings=[query_embedding],
                n_results=min(n_results, self.node_collection.count()),
                where=context_filter
            )
            
            formatted_results = []
            if results['distances'] and results['documents']:
                for i, (doc, meta, dist) in enumerate(zip(results['documents'][0], results['metadatas'][0], results['distances'][0])):
                    # Convert distance to similarity
                    similarity = max(0, min(1, 1 - dist))
                    
                    # Create a result entry
                    result = {
                        'content': doc,
                        'metadata': meta,
                        'similarity': similarity,
                        'memory_type': 'hierarchical'
                    }
                    
                    # Compute rank score
                    result["rank_score"] = self._compute_node_score(
                        meta, 
                        similarity, 
                        current_date
                    )
                    
                    formatted_results.append(result)
                    
                    # Update access metadata
                    self._update_node_access(meta['node_id'], current_date)
            
            return formatted_results
            
        except Exception as e:
            print(f"Error in direct recall: {str(e)}")
            return []

    def _compute_node_score(self, metadata, similarity, current_date):
        """Compute a score for a node based on recency, similarity, and access count."""
        # Similar to semantic memory's scoring but adapted for nodes
        recency = self._compute_recency(metadata, current_date)
        frequency = np.log1p(metadata.get("access_count", 1))
        
        return (
            recency * 0.3 +  # Recency component
            similarity * 0.5 +  # Similarity component
            frequency * 0.2  # Frequency component
        )

    def _compute_recency(self, metadata, current_date):
        """Compute the recency score for a node."""
        try:
            last_access = datetime.fromisoformat(metadata.get("last_access_timestamp", metadata.get("timestamp")))
            time_diff = current_date - last_access
            hours_diff = time_diff.total_seconds() / 3600
            recency = self.semantic_memory.decay_factor ** hours_diff
            return recency
        except:
            return 0.5  # Default value if calculation fails

    def _generate_embedding(self, content: str) -> List[float]:
        """Generate an embedding for the given content."""
        if self.semantic_memory.embeddings_store_prefix:
            return self.semantic_memory.encoder.encode(
                self.semantic_memory.embeddings_store_prefix + content,
                **self.semantic_memory.embedding_kwargs
            ).tolist()
        else:
            return self.semantic_memory.encoder.encode(
                content,
                **self.semantic_memory.embedding_kwargs
            ).tolist()

    def _combine_embeddings(self, embedding1: List[float], embedding2: List[float]) -> List[float]:
        """Combine two embeddings (simple average)."""
        if not embedding1 or not embedding2:
            return embedding1 or embedding2 or []
            
        # Convert to numpy arrays
        emb1 = np.array(embedding1)
        emb2 = np.array(embedding2)
        
        # Average the embeddings
        combined = (emb1 + emb2) / 2
        
        # Normalize the combined embedding
        norm = np.linalg.norm(combined)
        if norm > 0:
            combined = combined / norm
            
        return combined.tolist()

    def _update_node_metadata(self, node_id: str, metadata_updates: Dict):
        """Update specific metadata fields for a node."""
        node = self.get_node(node_id)
        if not node:
            return False
            
        # Update the node metadata
        node.metadata.update(metadata_updates)
        
        # Update in the collection
        self.node_collection.update(
            ids=[node_id],
            metadatas=[node.metadata]
        )
        
        # Update in cache
        self.node_cache[node_id] = node
        
        return True

    def _update_node_access(self, node_id: str, access_time: datetime):
        """Update the access metadata for a node."""
        node = self.get_node(node_id)
        if not node:
            return
            
        # Update access count and timestamp
        access_count = node.metadata.get("access_count", 0) + 1
        node.metadata["access_count"] = access_count
        node.metadata["last_access_timestamp"] = access_time.isoformat()
        
        # Update in collection
        self.node_collection.update(
            ids=[node_id],
            metadatas=[node.metadata]
        )
        
        # Update in cache
        node.access_count = access_count
        node.last_access_timestamp = access_time.isoformat()
        self.node_cache[node_id] = node

    def _delete_node_relationships(self, node_id: str):
        """Delete all relationships involving a node."""
        try:
            # Find relationships where node is source or target
            source_results = self.relationship_collection.get(
                where={"source_id": node_id}
            )
            target_results = self.relationship_collection.get(
                where={"target_id": node_id}
            )
            
            # Delete relationships
            if source_results['ids']:
                self.relationship_collection.delete(ids=source_results['ids'])
                
            if target_results['ids']:
                self.relationship_collection.delete(ids=target_results['ids'])
                
        except Exception as e:
            print(f"Error deleting relationships for node {node_id}: {str(e)}")

    def _get_parent_chain(self, node: Optional[HierarchicalMemoryNode], max_depth: int = 2) -> List[Dict]:
        """Get the chain of parents up to a specified depth."""
        if node is None or not node.parent_id or max_depth <= 0:
            return []
            
        ancestors = self.get_ancestors(node.node_id, max_depth=max_depth)
        return [
            {
                "node_id": ancestor.node_id,
                "content": ancestor.content,
                "level": i + 1  # 1 = immediate parent, 2 = grandparent, etc.
            }
            for i, ancestor in enumerate(ancestors)
        ]

    def _get_children_summary(self, node: Optional[HierarchicalMemoryNode], max_depth: int = 1) -> Dict:
        """Get a summary of children for a node."""
        if node is None or not node.child_ids:
            return {"count": 0, "children": []}
            
        children = self.get_children(node.node_id)
        
        # Get grandchildren count if needed
        grandchildren_count = 0
        if max_depth > 1:
            for child in children:
                child_children = self.get_children(child.node_id)
                grandchildren_count += len(child_children)
        
        return {
            "count": len(children),
            "grandchildren_count": grandchildren_count,
            "children": [
                {
                    "node_id": child.node_id,
                    "content": child.content
                }
                for child in children
            ]
        }

    def _adjust_score_with_hierarchy(self, result: Dict, query: str):
        """Adjust the ranking score based on hierarchical context."""
        try:
            base_score = result.get('rank_score', result.get('similarity', 0.5))
            
            # Get parent context contribution
            parent_context_score = 0
            if self.config.include_parent_context and 'hierarchy' in result and 'parents' in result['hierarchy']:
                parents = result['hierarchy']['parents']
                if parents:
                    parent_embeddings = []
                    for parent in parents:
                        try:
                            parent_node = self.get_node(parent['node_id'])
                            if parent_node and parent_node.embedding:
                                parent_embeddings.append(parent_node.embedding)
                        except Exception as e:
                            print(f"Error processing parent node: {str(e)}")
                            continue
                
                if parent_embeddings:
                    # Average parent embeddings
                    avg_parent_embedding = np.mean(np.array(parent_embeddings), axis=0).tolist()
                    
                    # Compare to query
                    query_embedding = self._generate_embedding(query)
                    parent_similarity = 1 - self._cosine_distance(avg_parent_embedding, query_embedding)
                    parent_context_score = parent_similarity * self.config.parent_weight
        
        # Get child context contribution
            child_context_score = 0
            if self.config.include_child_context and 'hierarchy' in result and 'children' in result['hierarchy']:
                children = result['hierarchy']['children'].get('children', [])
                if children:
                    child_embeddings = []
                    for child in children:
                        try:
                            child_node = self.get_node(child['node_id'])
                            if child_node and child_node.embedding:
                                child_embeddings.append(child_node.embedding)
                        except Exception as e:
                            print(f"Error processing child node: {str(e)}")
                            continue
                        
                if child_embeddings:
                    # Average child embeddings
                    avg_child_embedding = np.mean(np.array(child_embeddings), axis=0).tolist()
                    
                    # Compare to query
                    query_embedding = self._generate_embedding(query)
                    child_similarity = 1 - self._cosine_distance(avg_child_embedding, query_embedding)
                    child_context_score = child_similarity * self.config.child_weight
        
            # Combine scores
            result['rank_score'] = base_score + parent_context_score + child_context_score
        except Exception as e:
            print(f"Error adjusting score with hierarchy: {str(e)}")
            # Use base score if there was an error
            result['rank_score'] = result.get('rank_score', result.get('similarity', 0.5))

    def _get_related_nodes(self, node: Optional[HierarchicalMemoryNode], query: str, 
                          excluded_ids: Set[str]) -> List[Tuple[HierarchicalMemoryNode, float]]:
        """Get related nodes (siblings, children, etc.) that are relevant to the query."""
        related_nodes = []
        
        # Safety check - if node is None, return empty list
        if node is None:
            return related_nodes
            
        query_embedding = self._generate_embedding(query)
        
        # Add children if they are relevant
        try:
            children = self.get_children(node.node_id)
            for child in children:
                if child.node_id in excluded_ids:
                    continue
                    
                if child.embedding:
                    similarity = 1 - self._cosine_distance(child.embedding, query_embedding)
                    if similarity > self.semantic_memory.minimum_similarity_threshold:
                        related_nodes.append((child, similarity))
        except Exception as e:
            print(f"Error getting children: {str(e)}")
        
        # Add siblings if they are relevant
        if node.parent_id:
            try:
                parent = self.get_node(node.parent_id)
                if parent:
                    siblings = self.get_children(parent.node_id)
                    for sibling in siblings:
                        if sibling.node_id == node.node_id or sibling.node_id in excluded_ids:
                            continue
                            
                        if sibling.embedding:
                            similarity = 1 - self._cosine_distance(sibling.embedding, query_embedding)
                            if similarity > self.semantic_memory.minimum_similarity_threshold:
                                related_nodes.append((sibling, similarity))
            except Exception as e:
                print(f"Error getting siblings: {str(e)}")
        
        # Sort by similarity
        related_nodes.sort(key=lambda x: x[1], reverse=True)
        return related_nodes

    def _node_to_result(self, node: Optional[HierarchicalMemoryNode], query: str, 
                        similarity: float, current_date: datetime) -> Dict:
        """Convert a node to a result dictionary."""
        if node is None:
            # Return a minimal result if node is None
            return {
                'content': "Node not found",
                'metadata': {"node_id": "unknown", "type": "missing_node"},
                'similarity': 0.0,
                'memory_type': 'hierarchical',
                'rank_score': 0.0
            }
            
        return {
            'content': node.content,
            'metadata': node.metadata,
            'similarity': similarity,
            'memory_type': 'hierarchical',
            'rank_score': self._compute_node_score(node.metadata, similarity, current_date)
        }

    def _is_descendant(self, ancestor_id: str, potential_descendant_id: str) -> bool:
        """Check if a node is a descendant of another node."""
        if ancestor_id == potential_descendant_id:
            return True
            
        node = self.get_node(potential_descendant_id)
        if not node or not node.parent_id:
            return False
            
        if node.parent_id == ancestor_id:
            return True
            
        return self._is_descendant(ancestor_id, node.parent_id)

    def _get_max_depth(self, node_id: str, current_depth: int = 0) -> int:
        """Get the maximum depth of a node's hierarchy."""
        node = self.get_node(node_id)
        if not node or not node.child_ids:
            return current_depth
            
        max_child_depth = current_depth
        for child_id in node.child_ids:
            child_depth = self._get_max_depth(child_id, current_depth + 1)
            max_child_depth = max(max_child_depth, child_depth)
            
        return max_child_depth

    def _generate_cluster_summary(self, nodes: List[HierarchicalMemoryNode]) -> str:
        """Generate a summary for a cluster of nodes."""
        try:
            # Use semantic memory's extraction strategy if it's more sophisticated than simple concat
            # Compare function names to identify the simple strategy
            extract_fn = self.semantic_memory._extract_pattern
            simple_extract_fn = SimpleExtractPatternStrategy().extract_pattern
            
            # Check if we're using something other than the simple strategy
            if extract_fn.__qualname__ != simple_extract_fn.__qualname__:
                pattern_result = self.semantic_memory._extract_pattern(
                    f"cluster_{uuid.uuid4()}",
                    [node.content for node in nodes],
                    [node.metadata for node in nodes]
                )
                return pattern_result["content"]
        except Exception as e:
            print(f"Error generating cluster summary: {str(e)}")
            
        # Otherwise, create a simple summary
        all_content = [node.content for node in nodes]
        common_themes = "Cluster containing information about: " + ", ".join(all_content[:3])
        if len(all_content) > 3:
            common_themes += f", and {len(all_content) - 3} more items"
            
        return common_themes

    def _cosine_distance(self, emb1: List[float], emb2: List[float]) -> float:
        """Calculate cosine distance between two embeddings."""
        if not emb1 or not emb2:
            return 1.0
            
        # Convert to numpy arrays
        a = np.array(emb1)
        b = np.array(emb2)
        
        # Calculate cosine similarity
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        
        if norm_a == 0 or norm_b == 0:
            return 1.0
            
        cosine_similarity = np.dot(a, b) / (norm_a * norm_b)
        
        # Convert to distance (1 - similarity)
        return 1.0 - cosine_similarity