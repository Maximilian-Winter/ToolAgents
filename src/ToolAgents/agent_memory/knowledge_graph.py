from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional

import graphviz
from pydantic import BaseModel, Field
import networkx as nx
import json
import os


class Entity(BaseModel):
    """
    Represents an entity in the knowledge graph.
    """
    entity_id: str = Field(default="",
                           description="The entity id. Gets automatically set when added to the knowledge graph.")
    entity_type: str = Field(..., description="The type of entity")
    attributes: Dict[str, Any] = Field(..., description="The entity attributes")


class EntityQuery(BaseModel):
    """
    Represents an entity query.
    """
    entity_type: Optional[str] = Field(None, description="The type of entity to query")
    attribute_filter: Optional[Dict[str, Any]] = Field(None, description="The attribute filter")


class GeneralizedKnowledgeGraph:
    def __init__(self):
        self.graph = nx.Graph()
        self.entity_counters = {}

    def generate_entity_id(self, entity_type: str) -> str:
        """
        Generates a unique entity ID for a given entity type.

        Args:
            entity_type (str): The type of the entity.

        Returns:
            str: A unique entity ID.
        """
        if entity_type in self.entity_counters:
            self.entity_counters[entity_type] += 1
        else:
            self.entity_counters[entity_type] = 1
        return f"{entity_type}-{self.entity_counters[entity_type]}"

    def add_entity(self, entity: Entity) -> str:
        """
        Adds an entity to the knowledge graph.

        Args:
            entity (Entity): The entity to add.

        Returns:
            str: The entity id of the added entity.
        """
        if not entity.entity_id:
            entity.entity_id = self.generate_entity_id(entity.entity_type)
        self.graph.add_node(entity.entity_id, entity_type=entity.entity_type, **entity.attributes)
        return entity.entity_id

    def query_entities(self, entity_query: EntityQuery) -> str:
        """
        Query entities of the knowledge graph.

        Args:
            entity_query (EntityQuery): The entity query parameters.

        Returns:
            str: A formatted string containing the query results.
        """
        matching_entities: List[Dict[str, Any]] = []
        for node, data in self.graph.nodes(data=True):
            if (entity_query.entity_type is None or
                    data.get('entity_type').lower() == entity_query.entity_type.lower()):
                if entity_query.attribute_filter is None or all(
                        data.get(k) == v for k, v in entity_query.attribute_filter.items()
                ):
                    matching_entities.append({'id': node, **data})

        if not matching_entities:
            return f"No entities found matching the query"

        result = "Matching entities:\n"
        for entity in matching_entities:
            result += f"- ID: {entity['id']}\n"
            result += f"  Type: {entity['entity_type']}\n"
            for key, value in entity.items():
                if key not in ['id', 'entity_type']:
                    result += f"  {key}: {value}\n"
            result += "\n"

        return result

    def add_relationship(self, first_entity_id: str, relationship_type: str, second_entity_id: str,
                         attributes: Optional[Dict[str, Any]] = None) -> str:
        """
        Adds a relationship between two entities to the knowledge graph.

        Args:
            first_entity_id (str): The id of the first entity.
            relationship_type (str): The type of the relationship.
            second_entity_id (str): The id of the second entity.
            attributes (Optional[Dict[str, Any]]): Additional attributes for the relationship.

        Returns:
            str: A message confirming the addition of the relationship.
        """
        self.graph.add_edge(first_entity_id, second_entity_id,
                            relationship_type=relationship_type,
                            **attributes if attributes else {})
        return f"Relationship '{relationship_type}' added successfully between entities {first_entity_id} and {second_entity_id}"

    def query_relationships(self, entity_id: str, relationship_type: Optional[str] = None) -> str:
        """
        Query relationships of an entity in the knowledge graph.

        Args:
            entity_id (str): The id of the entity to query relationships for.
            relationship_type (Optional[str]): The type of relationship to filter by.

        Returns:
            str: A JSON string containing the query results.
        """
        relationships = []
        for neighbor in self.graph.neighbors(entity_id):
            edge_data = self.graph[entity_id][neighbor]
            if relationship_type is None or edge_data['relationship_type'] == relationship_type:
                relationships.append({
                    'related_entity': neighbor,
                    'relationship_type': edge_data['relationship_type'],
                    'attributes': {k: v for k, v in edge_data.items() if k != 'relationship_type'}
                })

        return json.dumps(relationships, indent=2)

    def find_path(self, start_entity_id: str, end_entity_id: str, max_depth: int = 5) -> str:
        """
        Finds a path between two entities in the knowledge graph.

        Args:
            start_entity_id (str): The id of the starting entity.
            end_entity_id (str): The id of the ending entity.
            max_depth (int): The maximum depth to search for a path.

        Returns:
            str: A string describing the path found or a message if no path is found.
        """
        try:
            path = nx.shortest_path(self.graph, start_entity_id, end_entity_id)
            if len(path) > max_depth:
                return f"Path found but exceeds maximum depth of {max_depth}"

            result = f"Path from {start_entity_id} to {end_entity_id}:\n"
            for i in range(len(path) - 1):
                edge_data = self.graph[path[i]][path[i + 1]]
                result += f"{path[i]} --({edge_data['relationship_type']})--> {path[i + 1]}\n"
            return result
        except nx.NetworkXNoPath:
            return f"No path found between {start_entity_id} and {end_entity_id}"

    def get_entity_details(self, entity_id: str) -> str:
        """
        Retrieves detailed information about a specific entity.

        Args:
            entity_id (str): The id of the entity to retrieve details for.

        Returns:
            str: A string containing the detailed information of the entity.
        """
        try:
            entity_data = self.graph.nodes[entity_id]
            result = f"Details for entity {entity_id}:\n"
            for key, value in entity_data.items():
                result += f"{key}: {value}\n"
            return result
        except KeyError:
            return f"No entity found with ID {entity_id}"

    def get_nearby_entities(self, entity_id: str, entity_type: Optional[str] = None, max_distance: int = 2) -> str:
        """
        Finds entities that are near a specified entity in the knowledge graph.

        Args:
            entity_id (str): The id of the entity to search from.
            entity_type (Optional[str]): The type of entity to filter by.
            max_distance (int): The maximum distance (in graph edges) to search.

        Returns:
            str: A string listing the nearby entities found.
        """
        subgraph = nx.ego_graph(self.graph, entity_id, radius=max_distance)
        nearby_entities = []

        for node in subgraph.nodes():
            if node != entity_id and (entity_type is None or subgraph.nodes[node]['entity_type'] == entity_type):
                nearby_entities.append((node, subgraph.nodes[node]))

        if not nearby_entities:
            return f"No nearby entities found within {max_distance} steps of {entity_id}"

        result = f"Entities near {entity_id} (within {max_distance} steps):\n"
        for entity_id, entity_data in nearby_entities:
            result += f"- {entity_id}: {entity_data.get('name', 'Unnamed entity')} ({entity_data['entity_type']})\n"
        return result

    def save_to_file(self, filename: str) -> None:
        """
        Save the GeneralizedKnowledgeGraph to a JSON file.

        Args:
            filename (str): The name of the file to save to.
        """
        data = {
            "graph": nx.node_link_data(self.graph),
            "entity_counters": self.entity_counters
        }
        with open(filename, 'w') as f:
            json.dump(data, f)

    @classmethod
    def load_from_file(cls, filename: str) -> 'GeneralizedKnowledgeGraph':
        """
        Load a GeneralizedKnowledgeGraph from a JSON file.

        Args:
            filename (str): The name of the file to load from.

        Returns:
            GeneralizedKnowledgeGraph: The loaded knowledge graph.
        """
        with open(filename, 'r') as f:
            data = json.load(f)

        gkg = cls()
        gkg.graph = nx.node_link_graph(data['graph'])
        gkg.entity_counters = data['entity_counters']
        return gkg

    def visualize(self, output_file: str = "knowledge_graph", format: str = "png") -> None:
        """
        Visualize the knowledge graph using Graphviz.
        Args:
            output_file: Name of the output file (without extension)
            format: Output format (e.g., "png", "pdf", "svg")
        """
        dot = graphviz.Digraph(comment="Knowledge Graph")

        # Add nodes
        for node, data in self.graph.nodes(data=True):
            label = f"{node}\n{json.dumps(data, indent=2)}"
            dot.node(node, label)

        # Add edges
        for u, v, data in self.graph.edges(data=True):
            label = data.get("relationship", "")
            dot.edge(u, v, label=label)

        # Render the graph
        dot.render(output_file, format=format, cleanup=True)
        print(f"Graph visualization saved as {output_file}.{format}")


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


# Example usage
if __name__ == "__main__":
    kg = GeneralizedKnowledgeGraph()

    # Adding entities
    person1 = Entity(entity_type="Person", attributes={"name": "Alice", "age": 30})
    person2 = Entity(entity_type="Person", attributes={"name": "Bob", "age": 35})
    location = Entity(entity_type="Location", attributes={"name": "Central Park", "city": "New York"})

    alice_id = kg.add_entity(person1)
    bob_id = kg.add_entity(person2)
    park_id = kg.add_entity(location)

    # Adding relationships
    kg.add_relationship(alice_id, "friend_of", bob_id)
    kg.add_relationship(alice_id, "visited", park_id)

    # Querying entities
    print(kg.query_entities(EntityQuery(entity_type="Person")))

    # Querying relationships
    print(kg.query_relationships(alice_id))

    # Finding path
    print(kg.find_path(bob_id, park_id))

    # Getting entity details
    print(kg.get_entity_details(alice_id))

    # Getting nearby entities
    print(kg.get_nearby_entities(alice_id))

    # Saving and loading
    kg.save_to_file("knowledge_graph.json")
    loaded_kg = GeneralizedKnowledgeGraph.load_from_file("knowledge_graph.json")
    print(loaded_kg.get_entity_details(alice_id))
