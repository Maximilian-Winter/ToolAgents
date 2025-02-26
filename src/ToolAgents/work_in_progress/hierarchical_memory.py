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
        self.minimum_similarity_threshold = 0.7

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
            parent_node = self.get_node(parent_id)
            if parent_node:  # Only proceed if parent exists
                self.create_relationship(node_id, parent_id, relationship_type)
                
                # Update parent's children list
                if node_id not in parent_node.child_ids:
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
        # Validate parameters
        n_results = max(1, min(n_results, 100))  # Limit to reasonable range
        
        # First, get base results using direct similarity search
        base_results = self._direct_recall(query, n_results * 2, context_filter, current_date)
        
        # If hierarchy context is not needed, return base results directly
        if not include_hierarchy or not base_results:
            return base_results[:n_results]
            
        # Enhance results with hierarchical context
        enhanced_results = []
        seen_node_ids = set()
        
        for result in base_results:
            try:
                # Extract node_id safely
                node_id = None
                if 'metadata' in result and result['metadata'] and 'node_id' in result['metadata']:
                    node_id = result['metadata']['node_id']
                
                if not node_id or node_id in seen_node_ids:
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
            try:
                result['hierarchy'] = {
                    'parents': self._get_parent_chain(node, max_depth=max_context_depth),
                    'children': self._get_children_summary(node, max_depth=1)
                }
                
                # Adjust the ranking score based on hierarchical context
                if self.config.include_parent_context or self.config.include_child_context:
                    self._adjust_score_with_hierarchy(result, query)
            except Exception as e:
                print(f"Error adding hierarchy to result: {str(e)}")
                # Continue without hierarchy if there's an error
                
            enhanced_results.append(result)
            
            # Add relevant siblings, parents, or children as separate results if highly relevant
            if len(enhanced_results) < n_results:
                try:
                    related_nodes = self._get_related_nodes(node, query, seen_node_ids)
                    for related_node, similarity in related_nodes:
                        if len(enhanced_results) >= n_results:
                            break
                            
                        if related_node and related_node.node_id:
                            related_result = self._node_to_result(related_node, query, similarity, current_date)
                            related_result['hierarchy'] = {
                                'parents': self._get_parent_chain(related_node, max_depth=max_context_depth),
                                'children': self._get_children_summary(related_node, max_depth=1)
                            }
                            
                            enhanced_results.append(related_result)
                            seen_node_ids.add(related_node.node_id)
                except Exception as e:
                    print(f"Error processing related nodes: {str(e)}")
        
        # Sort enhanced results by rank score and limit to requested number
        enhanced_results.sort(key=lambda x: x.get('rank_score', 0), reverse=True)
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
        try:
            embedding = self._generate_embedding(content)
        except Exception as e:
            print(f"Error generating embedding: {str(e)}")
            embedding = []  # Use empty list if embedding fails
        
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
        try:
            self.node_collection.add(
                documents=[content],
                metadatas=[node_metadata],
                embeddings=[embedding],
                ids=[node_id]
            )
        except Exception as e:
            print(f"Error storing node in collection: {str(e)}")
        
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
        
        try:
            # Update content if provided
            if content is not None:
                node.content = content
                try:
                    node.embedding = self._generate_embedding(content)
                except Exception as e:
                    print(f"Error generating embedding for update: {str(e)}")
                    # Keep existing embedding if generation fails
                
                # Update the node in the collection
                self.node_collection.update(
                    ids=[node_id],
                    documents=[content],
                    embeddings=[node.embedding] if node.embedding else None
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
        except Exception as e:
            print(f"Error updating node {node_id}: {str(e)}")
            return False

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
        
        try:
            # If recursive, delete all children first
            if recursive and node.child_ids:
                for child_id in node.child_ids.copy():  # Use copy to avoid modification during iteration
                    self.delete_node(child_id, recursive=True)
            
            # Update parent's child_ids list if parent exists
            if node.parent_id:
                parent_node = self.get_node(node.parent_id)
                if parent_node and node_id in parent_node.child_ids:
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
        except Exception as e:
            print(f"Error deleting node {node_id}: {str(e)}")
            return False

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
        
        try:    
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
                    old_parent.child_ids.remove(node_id)
                    self._update_node_metadata(
                        old_parent.node_id, 
                        {"child_ids_json": json.dumps(old_parent.child_ids)}
                    )
            
            # Add to new parent's children list
            if new_parent_id:
                new_parent = self.get_node(new_parent_id)
                if node_id not in new_parent.child_ids:
                    new_parent.child_ids.append(node_id)
                    self._update_node_metadata(
                        new_parent_id, 
                        {"child_ids_json": json.dumps(new_parent.child_ids)}
                    )
            
            # Update node's parent_id
            node.parent_id = new_parent_id
            update_metadata = {"parent_id": new_parent_id} if new_parent_id else {"parent_id": None}
            self._update_node_metadata(node_id, update_metadata)
            
            # Update relationships
            self._delete_node_relationships(node_id)
            if new_parent_id:
                self.create_relationship(
                    node_id, 
                    new_parent_id, 
                    HierarchicalRelationshipType.PARENT_CHILD
                )
            
            return True
        except Exception as e:
            print(f"Error moving node {node_id}: {str(e)}")
            return False

    def get_node(self, node_id: str) -> Optional[HierarchicalMemoryNode]:
        """
        Retrieve a specific node by ID.
        
        Args:
            node_id: The ID of the node to retrieve.
            
        Returns:
            The node object if found, None otherwise.
        """
        if not node_id:
            return None
            
        # Check cache first
        if node_id in self.node_cache:
            return self.node_cache[node_id]
        
        # Query the node collection
        try:
            result = self.node_collection.get(
                ids=[node_id],
                include=[IncludeEnum.metadatas, IncludeEnum.embeddings, IncludeEnum.documents]
            )
            
            # Validate result structure
            if (not isinstance(result, dict) or
                'ids' not in result or 
                not result['ids'] or 
                len(result['ids']) == 0 or 
                'metadatas' not in result or 
                not result['metadatas'] or 
                len(result['metadatas']) == 0 or 
                'documents' not in result or 
                not result['documents'] or 
                len(result['documents']) == 0):
                return None
            
            # Convert child_ids from JSON string back to list
            child_ids_json = result['metadatas'][0].get('child_ids_json', '[]')
            try:
                child_ids = json.loads(child_ids_json)
                if not isinstance(child_ids, list):
                    child_ids = []
            except Exception:
                child_ids = []
                
            # Extract embedding safely
            embedding = None
            if ('embeddings' in result and 
                isinstance(result['embeddings'], list) and 
                len(result['embeddings']) > 0):
                embedding = result['embeddings'][0]
            
            # Create the node object
            node = HierarchicalMemoryNode(
                node_id=node_id,
                content=result['documents'][0],
                metadata=result['metadatas'][0],
                parent_id=result['metadatas'][0].get('parent_id'),
                child_ids=child_ids,
                embedding=embedding,
                created_timestamp=result['metadatas'][0].get('timestamp', datetime.now().isoformat()),
                last_access_timestamp=result['metadatas'][0].get('last_access_timestamp', datetime.now().isoformat()),
                access_count=int(result['metadatas'][0].get('access_count', 1))
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
        visited = set()  # Track visited nodes to prevent cycles
        
        while current_id and (max_depth < 0 or depth < max_depth):
            # Prevent infinite loops due to cycles
            if current_id in visited:
                break
                
            visited.add(current_id)
            
            parent_node = self.get_node(current_id)
            if not parent_node:
                break
                
            ancestors.append(parent_node)
            current_id = parent_node.parent_id
            depth += 1
            
        return ancestors

    def create_relationship(self, source_id: str, target_id: str,
                           relationship_type: HierarchicalRelationshipType) -> Optional[str]:
        """
        Create a relationship between two nodes.
        
        Args:
            source_id: The ID of the source node.
            target_id: The ID of the target node.
            relationship_type: The type of relationship.
            
        Returns:
            The ID of the created relationship or None if failed.
        """
        if not source_id or not target_id:
            return None
            
        relationship_id = f"rel_{uuid.uuid4()}"
        
        try:
            # Get the nodes
            source_node = self.get_node(source_id)
            target_node = self.get_node(target_id)
            
            if not source_node or not target_node:
                return None
                
            # Create a combined embedding for the relationship
            relationship_embedding = None
            if (source_node.embedding and target_node.embedding and 
                isinstance(source_node.embedding, list) and 
                isinstance(target_node.embedding, list) and
                len(source_node.embedding) > 0 and 
                len(target_node.embedding) > 0):
                
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
                embeddings=[relationship_embedding] if relationship_embedding else None
            )
            
            return relationship_id
        except Exception as e:
            print(f"Error creating relationship: {str(e)}")
            return None

    def auto_organize(self, node_ids: List[str] = None) -> List[str]:
        """
        Automatically organize a set of nodes into a hierarchy.
        
        Args:
            node_ids: List of node IDs to organize (None to organize all root nodes).
            
        Returns:
            List of top-level node IDs in the created hierarchy.
        """
        try:
            # If no specific nodes provided, get all root nodes
            if node_ids is None:
                # Get all nodes first
                all_nodes = self.node_collection.get(
                    include=[IncludeEnum.metadatas, IncludeEnum.ids]
                )
                
                # Filter for root nodes (those without parent_id)
                node_ids = []
                if ('ids' in all_nodes and 
                    isinstance(all_nodes['ids'], list) and 
                    'metadatas' in all_nodes and 
                    isinstance(all_nodes['metadatas'], list)):
                    
                    for i, metadata in enumerate(all_nodes['metadatas']):
                        if i < len(all_nodes['ids']) and 'parent_id' not in metadata:
                            node_ids.append(all_nodes['ids'][i])
                
                if not node_ids:
                    return []
            
            # If fewer than threshold nodes, no need to organize
            if len(node_ids) < self.config.auto_organize_threshold:
                return node_ids
                
            # Get all nodes and their embeddings
            nodes = []
            embeddings = []
            node_objects = []
            
            for node_id in node_ids:
                node = self.get_node(node_id)
                if node:
                    node_objects.append(node)
                    if (node.embedding and 
                        isinstance(node.embedding, list) and 
                        len(node.embedding) > 0):
                        
                        nodes.append(node)
                        embeddings.append(node.embedding)
                    else:
                        try:
                            # Generate embedding if missing
                            emb = self._generate_embedding(node.content)
                            nodes.append(node)
                            embeddings.append(emb)
                        except Exception:
                            # Skip nodes with missing embeddings and errors
                            continue
            
            if not nodes or not embeddings:
                return node_ids
                
            # Use semantic memory's clustering strategy to group nodes
            try:
                clusters = self.semantic_memory._cluster_embeddings(
                    embeddings, 
                    getattr(self.config.base_config, 'minimum_cluster_similarity', 0.7)
                )
            except Exception as e:
                print(f"Error clustering embeddings: {str(e)}")
                return node_ids
            
            # For each cluster, create a parent node summarizing the cluster
            new_parent_ids = []
            
            for cluster_idx, cluster in enumerate(clusters):
                if not cluster:
                    continue
                    
                if len(cluster) < 2:  # Skip singleton clusters
                    if 0 <= cluster[0] < len(nodes):
                        new_parent_ids.append(nodes[cluster[0]].node_id)
                    continue
                    
                # Create a summary for this cluster
                try:
                    cluster_nodes = [nodes[i] for i in cluster if 0 <= i < len(nodes)]
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
                        if 0 <= i < len(nodes):
                            node_id = nodes[i].node_id
                            self.move_node(node_id, parent_id)
                            
                    new_parent_ids.append(parent_id)
                except Exception as e:
                    print(f"Error processing cluster {cluster_idx}: {str(e)}")
                    # Add unprocessed nodes to result
                    for i in cluster:
                        if 0 <= i < len(nodes):
                            new_parent_ids.append(nodes[i].node_id)
                    
            return new_parent_ids
        except Exception as e:
            print(f"Error in auto_organize: {str(e)}")
            return node_ids or []

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
            
            # Simple concatenation for summary (can be replaced with more sophisticated methods)
            summary = "\n\n".join(all_content[:5])
            if len(all_content) > 5:
                summary += f"\n\n... and {len(all_content) - 5} more items"
                
            # Update the node with the new summary
            self.update_node(node_id, content=summary)
            
            return summary
        except Exception as e:
            print(f"Error in summarize_node_with_children: {str(e)}")
            return None

    def get_stats(self) -> Dict:
        """
        Get statistics about the current state of the memory system.
        
        Returns:
            A dictionary with counts and hierarchy statistics.
        """
        try:
            # Get basic counts
            node_count = self.node_collection.count()
            relationship_count = self.relationship_collection.count()
            
            # Count root nodes
            root_node_ids = []
            try:
                # Get all nodes and filter for root nodes
                all_nodes = self.node_collection.get(
                    include=[IncludeEnum.metadatas]
                )
                
                if ('ids' in all_nodes and 
                    isinstance(all_nodes['ids'], list) and 
                    'metadatas' in all_nodes and 
                    isinstance(all_nodes['metadatas'], list)):
                    
                    for i, metadata in enumerate(all_nodes['metadatas']):
                        if i < len(all_nodes['ids']) and 'parent_id' not in metadata:
                            root_node_ids.append(all_nodes['ids'][i])
            except Exception as e:
                print(f"Error getting root nodes: {str(e)}")
            
            root_count = len(root_node_ids)
            
            # Calculate max depth (expensive operation, sample-based)
            max_depth = 0
            if root_count > 0:
                # Sample up to 5 root nodes to estimate max depth
                sample_size = min(5, root_count)
                for i in range(sample_size):
                    if i < len(root_node_ids):
                        depth = self._get_max_depth(root_node_ids[i])
                        max_depth = max(max_depth, depth)
            
            return {
                "node_count": node_count,
                "relationship_count": relationship_count,
                "root_node_count": root_count,
                "estimated_max_depth": max_depth,
                "cached_nodes": len(self.node_cache)
            }
        except Exception as e:
            print(f"Error in get_stats: {str(e)}")
            return {
                "node_count": 0,
                "relationship_count": 0,
                "root_node_count": 0,
                "estimated_max_depth": 0,
                "cached_nodes": len(self.node_cache),
                "error": str(e)
            }

    def clear_cache(self):
        """Clear the node cache to free memory."""
        self.node_cache.clear()

    # Helper methods
    def _direct_recall(self, query: str, n_results: int, 
                      context_filter: Optional[Dict], 
                      current_date: datetime) -> List[Dict]:
        """Perform a direct similarity search without hierarchical context."""
        try:
            query_embedding = self._generate_embedding(query)
            
            # Handle the case where no nodes exist yet
            if self.node_collection.count() == 0:
                return []
                
            # Perform the search
            results = self.node_collection.query(
                query_embeddings=[query_embedding],
                n_results=min(n_results, self.node_collection.count()),
                where=context_filter
            )
            
            # Format the results
            formatted_results = []
            
            if ('distances' in results and 
                isinstance(results['distances'], list) and 
                len(results['distances']) > 0 and
                'documents' in results and 
                isinstance(results['documents'], list) and 
                len(results['documents']) > 0 and
                'metadatas' in results and 
                isinstance(results['metadatas'], list) and 
                len(results['metadatas']) > 0):
                
                # Extract the results
                distances = results['distances'][0]
                documents = results['documents'][0]
                metadatas = results['metadatas'][0]
                
                # Process each result
                for i in range(min(len(distances), len(documents), len(metadatas))):
                    # Convert distance to similarity
                    similarity = max(0, min(1, 1 - distances[i]))
                    
                    # Create a result entry
                    result = {
                        'content': documents[i],
                        'metadata': metadatas[i],
                        'similarity': similarity,
                        'memory_type': 'hierarchical'
                    }
                    
                    # Compute rank score
                    result["rank_score"] = self._compute_node_score(
                        metadatas[i], 
                        similarity, 
                        current_date
                    )
                    
                    formatted_results.append(result)
                    
                    # Update access metadata if possible
                    if 'node_id' in metadatas[i]:
                        self._update_node_access(metadatas[i]['node_id'], current_date)
            
            return formatted_results
                
        except Exception as e:
            print(f"Error in direct recall: {str(e)}")
            return []

    def _compute_node_score(self, metadata, similarity, current_date):
        """Compute a score for a node based on recency, similarity, and access count."""
        try:
            # Calculate recency component
            recency = self._compute_recency(metadata, current_date)
            
            # Calculate frequency component (log to dampen effect of very high counts)
            access_count = metadata.get("access_count", 1)
            if not isinstance(access_count, (int, float)):
                access_count = 1
            frequency = np.log1p(access_count)
            
            # Combine components
            return (
                recency * 0.3 +  # Recency component
                similarity * 0.5 +  # Similarity component
                frequency * 0.2  # Frequency component
            )
        except Exception as e:
            print(f"Error in compute_node_score: {str(e)}")
            return similarity  # Default to just similarity if there's an error

    def _compute_recency(self, metadata, current_date):
        """Compute the recency score for a node."""
        try:
            # Get timestamp
            timestamp_str = metadata.get("last_access_timestamp", metadata.get("timestamp"))
            if not timestamp_str or not isinstance(timestamp_str, str):
                return 0.5  # Default value
                
            # Parse timestamp
            last_access = datetime.fromisoformat(timestamp_str)
            
            # Calculate time difference in hours
            time_diff = current_date - last_access
            hours_diff = time_diff.total_seconds() / 3600
            
            # Apply exponential decay
            decay_factor = getattr(self.semantic_memory, 'decay_factor', 0.99)
            recency = decay_factor ** hours_diff
            
            return recency
        except Exception as e:
            print(f"Error in compute_recency: {str(e)}")
            return 0.5  # Default value if calculation fails

    def _generate_embedding(self, content: str) -> List[float]:
        """Generate an embedding for the given content."""
        if not content or not isinstance(content, str):
            return []
            
        try:
            # Access the encoder from semantic memory
            prefix = getattr(self.semantic_memory, 'embeddings_store_prefix', '')
            kwargs = getattr(self.semantic_memory, 'embedding_kwargs', {})
            
            if prefix:
                embedding = self.semantic_memory.encoder.encode(
                    prefix + content,
                    **kwargs
                ).tolist()
            else:
                embedding = self.semantic_memory.encoder.encode(
                    content,
                    **kwargs
                ).tolist()
                
            return embedding
        except Exception as e:
            print(f"Error generating embedding: {str(e)}")
            return []  # Return empty list on error

    def _combine_embeddings(self, embedding1: List[float], embedding2: List[float]) -> List[float]:
        """Combine two embeddings (simple average)."""
        try:
            # Check for valid inputs
            if not embedding1 or not embedding2:
                return embedding1 or embedding2 or []
                
            if not isinstance(embedding1, list) or not isinstance(embedding2, list):
                return []
                
            # Convert to numpy arrays
            emb1 = np.array(embedding1, dtype=np.float32)
            emb2 = np.array(embedding2, dtype=np.float32)
            
            # Check for valid shapes
            if emb1.size == 0 or emb2.size == 0:
                return embedding1 or embedding2 or []
                
            # Average the embeddings
            combined = (emb1 + emb2) / 2
            
            # Normalize the combined embedding
            norm = np.linalg.norm(combined)
            if norm > 0:
                combined = combined / norm
                
            return combined.tolist()
        except Exception as e:
            print(f"Error combining embeddings: {str(e)}")
            return embedding1 or embedding2 or []  # Return an input on error

    def _update_node_metadata(self, node_id: str, metadata_updates: Dict) -> bool:
        """Update specific metadata fields for a node."""
        if not node_id or not metadata_updates:
            return False
            
        try:
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
        except Exception as e:
            print(f"Error updating node metadata: {str(e)}")
            return False

    def _update_node_access(self, node_id: str, access_time: datetime) -> bool:
        """Update the access metadata for a node."""
        if not node_id:
            return False
            
        try:
            node = self.get_node(node_id)
            if not node:
                return False
                
            # Update access count and timestamp
            access_count = int(node.metadata.get("access_count", 0)) + 1
            timestamp = access_time.isoformat()
            
            # Update in metadata
            node.metadata["access_count"] = access_count
            node.metadata["last_access_timestamp"] = timestamp
            
            # Update in collection
            self.node_collection.update(
                ids=[node_id],
                metadatas=[node.metadata]
            )
            
            # Update in node object
            node.access_count = access_count
            node.last_access_timestamp = timestamp
            
            # Update in cache
            self.node_cache[node_id] = node
            
            return True
        except Exception as e:
            print(f"Error updating node access: {str(e)}")
            return False

    def _delete_node_relationships(self, node_id: str) -> bool:
        """Delete all relationships involving a node."""
        if not node_id:
            return False
            
        try:
            # Find relationships where node is source
            source_results = None
            try:
                source_results = self.relationship_collection.get(
                    where={"source_id": node_id}
                )
            except Exception as e:
                print(f"Error getting source relationships: {str(e)}")
                
            # Find relationships where node is target
            target_results = None
            try:
                target_results = self.relationship_collection.get(
                    where={"target_id": node_id}
                )
            except Exception as e:
                print(f"Error getting target relationships: {str(e)}")
                
            # Delete relationships
            if (source_results and 'ids' in source_results and 
                isinstance(source_results['ids'], list) and source_results['ids']):
                self.relationship_collection.delete(ids=source_results['ids'])
                
            if (target_results and 'ids' in target_results and 
                isinstance(target_results['ids'], list) and target_results['ids']):
                self.relationship_collection.delete(ids=target_results['ids'])
                
            return True
                
        except Exception as e:
            print(f"Error deleting relationships for node {node_id}: {str(e)}")
            return False

    def _get_parent_chain(self, node: Optional[HierarchicalMemoryNode], max_depth: int = 2) -> List[Dict]:
        """Get the chain of parents up to a specified depth."""
        if not node or not node.parent_id or max_depth <= 0:
            return []
            
        try:
            ancestors = self.get_ancestors(node.node_id, max_depth=max_depth)
            
            return [
                {
                    "node_id": ancestor.node_id,
                    "content": ancestor.content,
                    "level": i + 1  # 1 = immediate parent, 2 = grandparent, etc.
                }
                for i, ancestor in enumerate(ancestors)
            ]
        except Exception as e:
            print(f"Error getting parent chain: {str(e)}")
            return []

    def _get_children_summary(self, node: Optional[HierarchicalMemoryNode], max_depth: int = 1) -> Dict:
        """Get a summary of children for a node."""
        if not node:
            return {"count": 0, "children": []}
            
        try:
            if not node.child_ids:
                return {"count": 0, "children": []}
                
            children = self.get_children(node.node_id)
            
            # Get grandchildren count if needed
            grandchildren_count = 0
            if max_depth > 1 and children:
                for child in children:
                    try:
                        if child and child.node_id:
                            child_children = self.get_children(child.node_id)
                            grandchildren_count += len(child_children)
                    except Exception:
                        continue
            
            return {
                "count": len(children),
                "grandchildren_count": grandchildren_count,
                "children": [
                    {
                        "node_id": child.node_id,
                        "content": child.content
                    }
                    for child in children if child
                ]
            }
        except Exception as e:
            print(f"Error getting children summary: {str(e)}")
            return {"count": 0, "children": [], "error": str(e)}

    def _adjust_score_with_hierarchy(self, result: Dict, query: str) -> None:
        """Adjust the ranking score based on hierarchical context."""
        if not result or not query:
            return
            
        try:
            # Get base score
            base_score = result.get('rank_score', result.get('similarity', 0.5))
            
            # Generate query embedding only once
            query_embedding = None
            
            # Initialize context scores
            parent_context_score = 0
            child_context_score = 0
            
            # Process parent context if available
            if (self.config.include_parent_context and 
                'hierarchy' in result and 
                'parents' in result['hierarchy'] and 
                isinstance(result['hierarchy']['parents'], list) and 
                result['hierarchy']['parents']):
                
                # Get parent embeddings
                parent_embeddings = []
                for parent in result['hierarchy']['parents']:
                    try:
                        if not isinstance(parent, dict) or 'node_id' not in parent:
                            continue
                            
                        parent_node = self.get_node(parent['node_id'])
                        if not parent_node or not parent_node.embedding:
                            continue
                            
                        if isinstance(parent_node.embedding, list) and parent_node.embedding:
                            parent_embeddings.append(parent_node.embedding)
                    except Exception:
                        continue
                
                # Calculate parent context score
                if parent_embeddings:
                    try:
                        # Convert to numpy array for processing
                        parent_embeddings_array = np.array(parent_embeddings, dtype=np.float32)
                        
                        # Check for valid shape before processing
                        if len(parent_embeddings_array.shape) == 2 and parent_embeddings_array.shape[0] > 0:
                            # Average embeddings
                            avg_parent_embedding = np.mean(parent_embeddings_array, axis=0).tolist()
                            
                            # Generate query embedding if needed
                            if query_embedding is None:
                                query_embedding = self._generate_embedding(query)
                                
                            # Calculate similarity and score
                            if query_embedding and avg_parent_embedding:
                                parent_similarity = max(0, min(1, 1 - self._cosine_distance(avg_parent_embedding, query_embedding)))
                                parent_context_score = parent_similarity * self.config.parent_weight
                    except Exception as e:
                        print(f"Error processing parent embeddings: {str(e)}")
            
            # Process child context if available
            if (self.config.include_child_context and 
                'hierarchy' in result and 
                'children' in result['hierarchy'] and 
                isinstance(result['hierarchy']['children'], dict) and
                'children' in result['hierarchy']['children'] and
                isinstance(result['hierarchy']['children']['children'], list) and
                result['hierarchy']['children']['children']):
                
                # Get child embeddings
                child_embeddings = []
                for child in result['hierarchy']['children']['children']:
                    try:
                        if not isinstance(child, dict) or 'node_id' not in child:
                            continue
                            
                        child_node = self.get_node(child['node_id'])
                        if not child_node or not child_node.embedding:
                            continue
                            
                        if isinstance(child_node.embedding, list) and child_node.embedding:
                            child_embeddings.append(child_node.embedding)
                    except Exception:
                        continue
                
                # Calculate child context score
                if child_embeddings:
                    try:
                        # Convert to numpy array for processing
                        child_embeddings_array = np.array(child_embeddings, dtype=np.float32)
                        
                        # Check for valid shape before processing
                        if len(child_embeddings_array.shape) == 2 and child_embeddings_array.shape[0] > 0:
                            # Average embeddings
                            avg_child_embedding = np.mean(child_embeddings_array, axis=0).tolist()
                            
                            # Generate query embedding if needed
                            if query_embedding is None:
                                query_embedding = self._generate_embedding(query)
                                
                            # Calculate similarity and score
                            if query_embedding and avg_child_embedding:
                                child_similarity = max(0, min(1, 1 - self._cosine_distance(avg_child_embedding, query_embedding)))
                                child_context_score = child_similarity * self.config.child_weight
                    except Exception as e:
                        print(f"Error processing child embeddings: {str(e)}")
            
            # Update score
            result['rank_score'] = base_score + parent_context_score + child_context_score
            
        except Exception as e:
            print(f"Error adjusting score with hierarchy: {str(e)}")
            # Use base score if there was an error
            result['rank_score'] = result.get('rank_score', result.get('similarity', 0.5))

    def _get_related_nodes(self, node: Optional[HierarchicalMemoryNode], query: str, 
                           excluded_ids: Set[str]) -> List[Tuple[HierarchicalMemoryNode, float]]:
        """Get related nodes (siblings, children, etc.) that are relevant to the query."""
        # Initialize empty list
        related_nodes = []
        
        # Safety check
        if not node or not query or not isinstance(excluded_ids, set):
            return related_nodes
            
        try:
            # Generate query embedding
            query_embedding = self._generate_embedding(query)
            if not query_embedding:
                return related_nodes
                
            # Set minimum threshold
            min_threshold = self.minimum_similarity_threshold
            
            # Get children
            try:
                children = self.get_children(node.node_id)
                for child in children:
                    # Skip if node is in excluded set
                    if not child or not child.node_id or child.node_id in excluded_ids:
                        continue
                        
                    # Check for valid embedding
                    if (child.embedding and 
                        isinstance(child.embedding, list) and 
                        len(child.embedding) > 0):
                        
                        # Calculate similarity
                        similarity = max(0, min(1, 1 - self._cosine_distance(child.embedding, query_embedding)))
                        
                        # Add if above threshold
                        if similarity > min_threshold:
                            related_nodes.append((child, similarity))
            except Exception as e:
                print(f"Error getting children: {str(e)}")
            
            # Get siblings if parent exists
            if node.parent_id:
                try:
                    parent = self.get_node(node.parent_id)
                    if parent:
                        siblings = self.get_children(parent.node_id)
                        for sibling in siblings:
                            # Skip if same as current node or in excluded set
                            if (not sibling or 
                                not sibling.node_id or 
                                sibling.node_id == node.node_id or 
                                sibling.node_id in excluded_ids):
                                continue
                                
                            # Check for valid embedding
                            if (sibling.embedding and 
                                isinstance(sibling.embedding, list) and 
                                len(sibling.embedding) > 0):
                                
                                # Calculate similarity
                                similarity = max(0, min(1, 1 - self._cosine_distance(sibling.embedding, query_embedding)))
                                
                                # Add if above threshold
                                if similarity > min_threshold:
                                    related_nodes.append((sibling, similarity))
                except Exception as e:
                    print(f"Error getting siblings: {str(e)}")
            
            # Sort by similarity (descending)
            related_nodes.sort(key=lambda x: x[1], reverse=True)
            
            return related_nodes
        except Exception as e:
            print(f"Error getting related nodes: {str(e)}")
            return []

    def _node_to_result(self, node: Optional[HierarchicalMemoryNode], query: str, 
                        similarity: float, current_date: datetime) -> Dict:
        """Convert a node to a result dictionary."""
        # Handle missing node
        if not node:
            return {
                'content': "Node not found",
                'metadata': {"node_id": "unknown", "type": "missing_node"},
                'similarity': 0.0,
                'memory_type': 'hierarchical',
                'rank_score': 0.0
            }
            
        try:
            # Create result dictionary
            result = {
                'content': node.content,
                'metadata': node.metadata,
                'similarity': similarity,
                'memory_type': 'hierarchical',
                'rank_score': self._compute_node_score(node.metadata, similarity, current_date)
            }
            
            return result
        except Exception as e:
            print(f"Error converting node to result: {str(e)}")
            # Return minimal valid result on error
            return {
                'content': node.content if node else "Error",
                'metadata': {"node_id": node.node_id if node else "unknown"},
                'similarity': similarity,
                'memory_type': 'hierarchical',
                'rank_score': similarity
            }

    def _is_descendant(self, ancestor_id: str, potential_descendant_id: str, 
                      visited_ids: set = None) -> bool:
        """Check if a node is a descendant of another node."""
        # Initialize visited set on first call to prevent infinite recursion
        if visited_ids is None:
            visited_ids = set()
            
        # Direct equality check
        if ancestor_id == potential_descendant_id:
            return True
            
        # Check for cycles
        if potential_descendant_id in visited_ids:
            return False
            
        visited_ids.add(potential_descendant_id)
            
        # Get the potential descendant node
        node = self.get_node(potential_descendant_id)
        
        # Not a descendant if node doesn't exist or has no parent
        if not node or node.parent_id is None:
            return False
            
        # Direct parent check
        if node.parent_id == ancestor_id:
            return True
            
        # Recursive check up the ancestry chain
        return self._is_descendant(ancestor_id, node.parent_id, visited_ids)

    def _get_max_depth(self, node_id: str, current_depth: int = 0, visited_ids: set = None) -> int:
        """Get the maximum depth of a node's hierarchy."""
        # Initialize visited set on first call to prevent infinite recursion
        if visited_ids is None:
            visited_ids = set()
            
        # Check for cycles or invalid ID
        if not node_id or node_id in visited_ids:
            return current_depth
            
        visited_ids.add(node_id)
        
        try:
            # Get the node
            node = self.get_node(node_id)
            if not node or not node.child_ids or len(node.child_ids) == 0:
                return current_depth
                
            # Track maximum depth encountered
            max_child_depth = current_depth
            
            # Process each child
            for child_id in node.child_ids:
                # Skip already visited nodes
                if child_id in visited_ids:
                    continue
                    
                # Recursively get depth
                child_depth = self._get_max_depth(child_id, current_depth + 1, visited_ids)
                max_child_depth = max(max_child_depth, child_depth)
                
            return max_child_depth
        except Exception as e:
            print(f"Error in get_max_depth: {str(e)}")
            return current_depth

    def _generate_cluster_summary(self, nodes: List[HierarchicalMemoryNode]) -> str:
        """Generate a summary for a cluster of nodes."""
        if not nodes:
            return "Empty cluster"
            
        try:
            # Extract content from nodes
            all_content = [node.content for node in nodes if node and node.content]
            
            # If no content, return default
            if not all_content:
                return "Cluster with no content"
                
            # Create a simple summary listing first few items
            max_items = min(3, len(all_content))
            common_themes = "Cluster containing information about: " + ", ".join(all_content[:max_items])
            
            # Indicate if there are more items
            if len(all_content) > max_items:
                common_themes += f", and {len(all_content) - max_items} more items"
                
            return common_themes
        except Exception as e:
            print(f"Error generating cluster summary: {str(e)}")
            return "Cluster summary (error occurred)"

    def _cosine_distance(self, emb1: List[float], emb2: List[float]) -> float:
        """Calculate cosine distance between two embeddings."""
        # Validate inputs
        if (emb1 is None or emb2 is None or
            not isinstance(emb1, (list, np.ndarray)) or 
            not isinstance(emb2, (list, np.ndarray)) or 
            len(emb1) == 0 or len(emb2) == 0):
            return 1.0
            
        try:
            # Convert to numpy arrays with explicit dtype
            a = np.array(emb1, dtype=np.float32)
            b = np.array(emb2, dtype=np.float32)
            
            # Check array shapes
            if a.size == 0 or b.size == 0 or a.shape != b.shape:
                return 1.0
                
            # Calculate cosine similarity safely
            norm_a = np.linalg.norm(a)
            norm_b = np.linalg.norm(b)
            
            if norm_a == 0 or norm_b == 0:
                return 1.0
                
            dot_product = np.dot(a, b)
            cosine_similarity = dot_product / (norm_a * norm_b)
            
            # Clip to valid range [-1, 1] to handle numerical issues
            cosine_similarity = np.clip(cosine_similarity, -1.0, 1.0)
            
            # Convert to distance (1 - similarity)
            return float(1.0 - cosine_similarity)
        except Exception as e:
            print(f"Error calculating cosine distance: {str(e)}")
            return 1.0  # Maximum distance on error