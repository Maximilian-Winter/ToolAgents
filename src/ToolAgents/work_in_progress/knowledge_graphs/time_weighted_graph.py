import json
from datetime import datetime, timedelta
from typing import Optional, Dict, Any

import networkx as nx

from ToolAgents.work_in_progress.knowledge_graphs.knowledge_graph import GeneralizedKnowledgeGraph


class TimeDecayWeightedKnowledgeGraph(GeneralizedKnowledgeGraph):
    """
    A knowledge graph implementation with time-based decay of edge weights,
    query-based weight reinforcement, and connectivity-based weight adjustment.
    """

    def __init__(self):
        """
        Initialize the TimeDecayWeightedKnowledgeGraph.
        """
        super().__init__()
        self.last_update = datetime.now()

    def add_relationship(self, first_entity_id: str, relationship_type: str, second_entity_id: str,
                         attributes: Optional[Dict[str, Any]] = None) -> str:
        """
        Add a relationship (edge) between two entities in the graph.

        Args:
            first_entity_id (str): The ID of the first entity.
            relationship_type (str): The type of the relationship.
            second_entity_id (str): The ID of the second entity.
            attributes (Optional[Dict[str, Any]]): Additional attributes for the relationship.

        Returns:
            str: A message confirming the addition of the relationship.
        """
        self.graph.add_edge(first_entity_id, second_entity_id,
                            relationship_type=relationship_type,
                            weight=1.0,
                            query_count=0,
                            last_accessed=datetime.now(),
                            **attributes if attributes else {})
        return f"Relationship '{relationship_type}' added successfully between entities {first_entity_id} and {second_entity_id}"

    def update_weights(self, decay_factor: float = 0.9, time_interval: timedelta = timedelta(hours=1)):
        """
        Update the weights of all edges in the graph based on time decay, query count, and connectivity.

        Args:
            decay_factor (float): The factor by which weights decay over time. Default is 0.9.
            time_interval (timedelta): The time interval for weight decay calculation. Default is 1 hour.
        """
        current_time = datetime.now()

        for u, v, data in self.graph.edges(data=True):
            # Calculate time decay
            time_since_last_access = (current_time - data['last_accessed']).total_seconds()
            time_decay = decay_factor ** (time_since_last_access / time_interval.total_seconds())

            # Apply time decay
            data['weight'] *= time_decay

            # Increase weight based on query count
            data['weight'] += (data['query_count'] * 0.1)

            # Increase weight based on connectivity
            connectivity = len(list(self.graph.neighbors(u))) + len(list(self.graph.neighbors(v)))
            data['weight'] *= (1.0 + (connectivity * 0.01))

            # Ensure weight doesn't exceed a maximum value (e.g., 10.0)
            data['weight'] = min(data['weight'], 10.0)

            # Reset query count and update last accessed time
            data['query_count'] = 0
            data['last_accessed'] = current_time

        self.last_update = current_time

    def query_relationships(self, entity_id: str, relationship_type: Optional[str] = None) -> str:
        """
        Query relationships of an entity in the graph.

        Args:
            entity_id (str): The ID of the entity to query relationships for.
            relationship_type (Optional[str]): The type of relationship to filter by. If None, all relationships are returned.

        Returns:
            str: A JSON string containing the query results.
        """
        relationships = []
        current_time = datetime.now()
        for neighbor in self.graph.neighbors(entity_id):
            edge_data = self.graph[entity_id][neighbor]
            if relationship_type is None or edge_data['relationship_type'] == relationship_type:
                # Increment query count and update last accessed time
                edge_data['query_count'] += 1
                edge_data['last_accessed'] = current_time

                relationships.append({
                    'related_entity': neighbor,
                    'relationship_type': edge_data['relationship_type'],
                    'weight': edge_data['weight'],
                    'attributes': {k: v for k, v in edge_data.items() if
                                   k not in ['relationship_type', 'weight', 'query_count', 'last_accessed']}
                })

        return json.dumps(relationships, indent=2)

    def find_weighted_path(self, start_entity_id: str, end_entity_id: str, max_depth: int = 5) -> str:
        """
        Find a weighted path between two entities in the graph.

        Args:
            start_entity_id (str): The ID of the starting entity.
            end_entity_id (str): The ID of the ending entity.
            max_depth (int): The maximum depth of the path to search for. Default is 5.

        Returns:
            str: A string describing the path found or a message if no path is found.
        """
        try:
            path = nx.dijkstra_path(self.graph, start_entity_id, end_entity_id, weight=lambda u, v, d: 1 / d['weight'])
            if len(path) > max_depth:
                return f"Path found but exceeds maximum depth of {max_depth}"

            result = f"Weighted path from {start_entity_id} to {end_entity_id}:\n"
            total_weight = 0
            current_time = datetime.now()
            for i in range(len(path) - 1):
                edge_data = self.graph[path[i]][path[i + 1]]
                weight = edge_data['weight']
                total_weight += weight
                # Increment query count and update last accessed time for each edge in the path
                edge_data['query_count'] += 1
                edge_data['last_accessed'] = current_time
                result += f"{path[i]} --({edge_data['relationship_type']}, weight: {weight:.2f})--> {path[i + 1]}\n"
            result += f"Total path weight: {total_weight:.2f}"
            return result
        except nx.NetworkXNoPath:
            return f"No path found between {start_entity_id} and {end_entity_id}"

    def prune_graph(self, weight_threshold: float = 0.5):
        """
        Remove edges from the graph whose weight is below a specified threshold.

        Args:
            weight_threshold (float): The threshold below which edges will be removed. Default is 0.5.
        """
        edges_to_remove = [(u, v) for u, v, data in self.graph.edges(data=True) if data['weight'] < weight_threshold]
        self.graph.remove_edges_from(edges_to_remove)

    def get_most_significant_relationships(self, top_n: int = 10) -> str:
        """
        Get the most significant relationships in the graph based on their weights.

        Args:
            top_n (int): The number of top relationships to return. Default is 10.

        Returns:
            str: A string describing the top N most significant relationships.
        """
        sorted_edges = sorted(self.graph.edges(data=True), key=lambda x: x[2]['weight'], reverse=True)[:top_n]
        result = "Most significant relationships:\n"
        for u, v, data in sorted_edges:
            result += f"{u} --({data['relationship_type']}, weight: {data['weight']:.2f})--> {v}\n"
        return result

    def save_to_file(self, filename: str) -> None:
        """
        Save the TimeDecayWeightedKnowledgeGraph to a JSON file.

        Args:
            filename (str): The name of the file to save to.
        """
        data = {
            "graph": nx.node_link_data(self.graph),
            "entity_counters": self.entity_counters,
            "last_update": self.last_update.isoformat()
        }
        with open(filename, 'w') as f:
            json.dump(data, f)

    @classmethod
    def load_from_file(cls, filename: str) -> 'TimeDecayWeightedKnowledgeGraph':
        """
        Load a TimeDecayWeightedKnowledgeGraph from a JSON file.

        Args:
            filename (str): The name of the file to load from.

        Returns:
            TimeDecayWeightedKnowledgeGraph: The loaded knowledge graph.
        """
        with open(filename, 'r') as f:
            data = json.load(f)

        graph = cls()
        graph.graph = nx.node_link_graph(data['graph'])
        graph.entity_counters = data['entity_counters']
        graph.last_update = datetime.fromisoformat(data['last_update'])
        return graph
