from types import NoneType

import graphviz
from pydantic import BaseModel, Field
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from typing import List, Set, Dict, Any, Optional, Tuple
import pandas as pd
from collections import defaultdict
import community
from networkx.algorithms.shortest_paths.weighted import dijkstra_path
import matplotlib.pyplot as plt
import yaml
import plotly.graph_objects as go
from pyvis.network import Network
import json
import networkx as nx
import holoviews as hv
from holoviews import opts
hv.extension('bokeh')
from ToolAgents import FunctionTool


class Entity(BaseModel):
    """
    Represents an entity in the knowledge graph.
    """
    entity_id: str = Field(default_factory=str,
                           description="The entity id. Gets automatically set when added to the knowledge graph.")
    entity_type: str = Field(..., description="The type of entity")
    attributes: Dict[str, Any] = Field(..., description="The entity attributes")


class EntityQuery(BaseModel):
    """
    Represents an entity query.
    """
    entity_type: Optional[str] = Field(default_factory=NoneType, description="The type of entity to query")
    attribute_filter: Optional[Dict[str, Any]] = Field(default_factory=NoneType, description="The attribute filter")


class KnowledgeGraph:
    def __init__(self):
        self.graph = nx.Graph()
        self.entity_counters = {}
        self.embeddings = {}
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

    @staticmethod
    def entity_to_string(entity) -> str:
        """
        Converts an entity to a string.
        :param entity:
        :return:
        """
        result = f"Id: {entity['id']}\n"
        result += f"Type: {entity['entity_type']}\n"
        result += "Attributes:\n"
        for key, value in entity.items():
            if key not in ['id', 'entity_type']:
                result += f"  {key}: {value}\n"
        result += "\n"
        return result

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
        Add an entity to the knowledge graph.

        Args:
            entity (Entity): The entity to add.

        Returns:
            str: The entity ID.
        """
        if not entity.entity_id:
            entity.entity_id = self.generate_entity_id(entity.entity_type)
        self.graph.add_node(entity.entity_id, entity_type=entity.entity_type, **entity.attributes)

        entity_text = KnowledgeGraph.entity_to_string(self.graph.nodes[entity.entity_id])
        self.embeddings[entity.entity_id] = self.embedding_model.encode(entity_text)

        return entity.entity_id

    def update_entity(self, entity_id: str, new_attributes: Dict[str, Any]) -> str:
        """
        Update an existing entity's attributes.

        Args:
            entity_id (str): The ID of the entity to update.
            new_attributes (Dict[str, Any]): The new attributes to update or add to the entity.

        Returns:
            str: A message confirming the update or an error message if the entity was not found.
        """
        if entity_id not in self.graph:
            return f"Error: No entity found with ID {entity_id}"

        entity_data = self.graph.nodes[entity_id]
        entity_data.update(new_attributes)

        entity_text = KnowledgeGraph.entity_to_string(entity_data)
        self.embeddings[entity_id] = self.embedding_model.encode(entity_text)

        return f"Entity {entity_id} updated successfully"

    def delete_entity(self, entity_id: str) -> str:
        """
        Delete an entity from the knowledge graph.

        Args:
            entity_id (str): The ID of the entity to delete.

        Returns:
            str: A message confirming the deletion or an error message if the entity was not found.
        """
        if entity_id not in self.graph:
            return f"Error: No entity found with ID {entity_id}"

        self.graph.remove_node(entity_id)
        del self.embeddings[entity_id]

        return f"Entity {entity_id} deleted successfully"

    def query_entities(self, entity_query: EntityQuery) -> str:
        """
        Query entities in the knowledge graph based on type and attributes.

        Args:
            entity_query (EntityQuery): The entity query parameters.

        Returns:
            str: A formatted string containing the query results.
        """
        matching_entities = []
        for node, data in self.graph.nodes(data=True):
            if (entity_query.entity_type is None or
                    data.get('entity_type').lower() == entity_query.entity_type.lower()):
                if entity_query.attribute_filter is None or all(
                        data.get(k) == v for k, v in entity_query.attribute_filter.items()
                ):
                    matching_entities.append({'id': node, **data})

        if not matching_entities:
            return "No matching entities found."

        result = "Matching entities:\n"
        for entity in matching_entities:
            result += f"- ID: {entity['id']}\n"
            result += f"  Type: {entity['entity_type']}\n"
            result += "  Attributes:\n"
            for key, value in entity.items():
                if key not in ['id', 'entity_type']:
                    result += f"    {key}: {value}\n"
            result += "\n"

        return result

    def add_relationship(self, first_entity_id: str, relationship_type: str, second_entity_id: str,
                         attributes: Optional[Dict[str, Any]] = None) -> str:
        """
        Add a relationship between two entities in the knowledge graph.

        Args:
            first_entity_id (str): The ID of the first entity.
            relationship_type (str): The type of the relationship.
            second_entity_id (str): The ID of the second entity.
            attributes (Optional[Dict[str, Any]]): Additional attributes for the relationship.

        Returns:
            str: A message confirming the addition of the relationship.
        """
        if first_entity_id not in self.graph:
            return f"Error: Entity with ID {first_entity_id} not found"
        if second_entity_id not in self.graph:
            return f"Error: Entity with ID {second_entity_id} not found"

        self.graph.add_edge(first_entity_id, second_entity_id,
                            relationship_type=relationship_type,
                            **attributes if attributes else {})
        return f"Relationship '{relationship_type}' added successfully between entities {first_entity_id} and {second_entity_id}"

    def query_relationships(self, entity_id: str, relationship_type: Optional[str] = None) -> str:
        """
        Query relationships of an entity in the knowledge graph.

        Args:
            entity_id (str): The ID of the entity to query relationships for.
            relationship_type (Optional[str]): The type of relationship to filter by.

        Returns:
            str: A formatted string containing the query results.
        """
        if entity_id not in self.graph:
            return f"Error: No entity found with ID {entity_id}"

        relationships = []
        for neighbor in self.graph.neighbors(entity_id):
            edge_data = self.graph[entity_id][neighbor]
            if relationship_type is None or edge_data['relationship_type'] == relationship_type:
                relationships.append({
                    'related_entity': neighbor,
                    'relationship_type': edge_data['relationship_type'],
                    'attributes': {k: v for k, v in edge_data.items() if k != 'relationship_type'}
                })

        if not relationships:
            return f"No relationships found for entity {entity_id}"

        result = f"Relationships for entity {entity_id}:\n"
        for rel in relationships:
            result += f"- Related Entity: {rel['related_entity']}\n"
            result += f"  Relationship Type: {rel['relationship_type']}\n"
            if rel['attributes']:
                result += "  Attributes:\n"
                for key, value in rel['attributes'].items():
                    result += f"    {key}: {value}\n"
            result += "\n"

        return result

    def semantic_search(self, query: str, top_k: int = 5) -> str:
        """
        Perform semantic search on the knowledge graph using embeddings.

        Args:
            query (str): The search query.
            top_k (int): The number of top results to return.

        Returns:
            str: A formatted string containing the top-k matching entities and their similarities.
        """
        query_embedding = self.embedding_model.encode(query)

        similarities = {}
        for entity_id, embedding in self.embeddings.items():
            similarity = cosine_similarity([query_embedding], [embedding])[0][0]
            similarities[entity_id] = similarity

        top_entities = sorted(similarities.items(), key=lambda x: x[1], reverse=True)[:top_k]

        result = f"Top {top_k} entities semantically similar to '{query}':\n"
        for entity_id, similarity in top_entities:
            entity_data = self.graph.nodes[entity_id]
            result += f"- Entity ID: {entity_id}\n"
            result += f"  Similarity: {similarity:.4f}\n"
            result += f"  Type: {entity_data['entity_type']}\n"
            result += "  Attributes:\n"
            for key, value in entity_data.items():
                if key != 'entity_type':
                    result += f"    {key}: {value}\n"
            result += "\n"

        return result

    def find_path(self, start_entity_id: str, end_entity_id: str, max_depth: int = 5) -> str:
        """
        Find a path between two entities in the knowledge graph.

        Args:
            start_entity_id (str): The ID of the starting entity.
            end_entity_id (str): The ID of the ending entity.
            max_depth (int): The maximum depth to search for a path.

        Returns:
            str: A description of the path found or an error message if no path is found.
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
        except nx.exception.NodeNotFound:
            return f"Error: Either source {start_entity_id} or target {end_entity_id} is not in the graph"

    def get_entity_details(self, entity_id: str) -> str:
        """
        Retrieve detailed information about a specific entity.

        Args:
            entity_id (str): The ID of the entity to retrieve details for.

        Returns:
            str: A formatted string containing the detailed information of the entity.
        """
        if entity_id not in self.graph:
            return f"Error: No entity found with ID {entity_id}"

        entity_data = self.graph.nodes[entity_id]
        result = f"Details for entity {entity_id}:\n"
        result += f"Type: {entity_data['entity_type']}\n"
        result += "Attributes:\n"
        for key, value in entity_data.items():
            if key != 'entity_type':
                result += f"  {key}: {value}\n"
        return result

    def get_nearby_entities(self, entity_id: str, entity_type: Optional[str] = None, max_distance: int = 2) -> str:
        """
        Find entities that are near a specified entity in the knowledge graph.

        Args:
            entity_id (str): The ID of the entity to search from.
            entity_type (Optional[str]): The type of entity to filter by.
            max_distance (int): The maximum distance (in graph edges) to search.

        Returns:
            str: A formatted string listing the nearby entities found.
        """
        if entity_id not in self.graph:
            return f"Error: No entity found with ID {entity_id}"

        subgraph = nx.ego_graph(self.graph, entity_id, radius=max_distance)
        nearby_entities = []

        for node in subgraph.nodes():
            if node != entity_id and (entity_type is None or subgraph.nodes[node]['entity_type'] == entity_type):
                nearby_entities.append({
                    "id": node,
                    "name": subgraph.nodes[node].get('name', 'Unnamed entity'),
                    "type": subgraph.nodes[node]['entity_type']
                })

        if not nearby_entities:
            return f"No nearby entities found within {max_distance} steps of {entity_id}"

        result = f"Entities near {entity_id} (within {max_distance} steps):\n"
        for entity in nearby_entities:
            result += f"- ID: {entity['id']}\n"
            result += f"  Name: {entity['name']}\n"
            result += f"  Type: {entity['type']}\n\n"

        return result

    def save_to_file(self, filename: str) -> None:
        """
        Save the EnhancedGeneralizedKnowledgeGraph to a JSON file.

        Args:
            filename (str): The name of the file to save to.
        """
        data = {
            "graph": nx.node_link_data(self.graph),
            "entity_counters": self.entity_counters,
            "embeddings": {k: v.tolist() for k, v in self.embeddings.items()}
        }
        with open(filename, 'w') as f:
            json.dump(data, f)

    @classmethod
    def load_from_file(cls, filename: str) -> 'KnowledgeGraph':
        """
        Load an EnhancedGeneralizedKnowledgeGraph from a JSON file.

        Args:
            filename (str): The name of the file to load from.

        Returns:
            EnhancedGeneralizedKnowledgeGraph: The loaded knowledge graph.
        """
        with open(filename, 'r') as f:
            data = json.load(f)

        gkg = cls()
        gkg.graph = nx.node_link_graph(data['graph'])
        gkg.entity_counters = data['entity_counters']
        gkg.embeddings = {k: np.array(v) for k, v in data['embeddings'].items()}
        return gkg

    def visualize(self, output_file: str = "knowledge_graph", format: str = "png") -> None:
        """
        Visualize the knowledge graph using Graphviz.

        Args:
            output_file (str): Name of the output file (without extension)
            format (str): Output format (e.g., "png", "pdf", "svg")
        """
        dot = graphviz.Digraph(comment="Knowledge Graph")

        # Add nodes
        for node, data in self.graph.nodes(data=True):
            label = f"{node}\n{json.dumps(data, indent=2)}"
            dot.node(node, label)

        # Add edges
        for u, v, data in self.graph.edges(data=True):
            label = data.get("relationship_type", "")
            dot.edge(u, v, label=label)

        # Render the graph
        dot.render(output_file, format=format, cleanup=True)
        print(f"Graph visualization saved as {output_file}.{format}")

    def get_tools(self):
        return [FunctionTool(self.add_entity), FunctionTool(self.update_entity), FunctionTool(self.delete_entity),
                FunctionTool(self.query_entities), FunctionTool(self.add_relationship), FunctionTool(self.query_relationships),
                FunctionTool(self.semantic_search), FunctionTool(self.get_entity_details)]

    def get_subgraph(self, entity_ids: List[str]) -> 'KnowledgeGraph':
        """
        Create a new knowledge graph containing only the specified entities and their relationships.

        Args:
            entity_ids (List[str]): List of entity IDs to include in the subgraph

        Returns:
            KnowledgeGraph: A new knowledge graph containing the specified entities
        """
        subgraph = KnowledgeGraph()
        node_subset = [node for node in entity_ids if node in self.graph]
        subgraph.graph = self.graph.subgraph(node_subset).copy()
        subgraph.entity_counters = {k: v for k, v in self.entity_counters.items()
                                    if any(k in node_id for node_id in node_subset)}
        subgraph.embeddings = {k: v for k, v in self.embeddings.items()
                               if k in node_subset}
        return subgraph

    def get_connected_components(self) -> List[Set[str]]:
        """
        Get all connected components in the knowledge graph.

        Returns:
            List[Set[str]]: List of sets containing entity IDs in each connected component
        """
        return list(nx.connected_components(self.graph))

    def detect_communities(self, resolution: float = 1.0) -> Dict[str, int]:
        """
        Detect communities in the knowledge graph using the Louvain method.

        Args:
            resolution (float): Resolution parameter for community detection

        Returns:
            Dict[str, int]: Dictionary mapping entity IDs to their community numbers
        """
        return community.best_partition(self.graph, resolution=resolution)

    def get_central_entities(self, method: str = 'betweenness', top_k: int = 10) -> List[Tuple[str, float]]:
        """
        Find the most central entities in the graph using various centrality metrics.

        Args:
            method (str): Centrality metric to use ('betweenness', 'eigenvector', 'pagerank', 'degree')
            top_k (int): Number of top central entities to return

        Returns:
            List[Tuple[str, float]]: List of (entity_id, centrality_score) tuples
        """
        centrality_funcs = {
            'betweenness': nx.betweenness_centrality,
            'eigenvector': nx.eigenvector_centrality_numpy,
            'pagerank': nx.pagerank,
            'degree': nx.degree_centrality
        }

        if method not in centrality_funcs:
            raise ValueError(f"Unknown centrality method: {method}")

        centrality = centrality_funcs[method](self.graph)
        return sorted(centrality.items(), key=lambda x: x[1], reverse=True)[:top_k]

    def find_shortest_path_weighted(self, start_entity_id: str, end_entity_id: str,
                                    weight_attribute: str = 'weight') -> Tuple[List[str], float]:
        """
        Find the shortest path between two entities considering edge weights.

        Args:
            start_entity_id (str): Starting entity ID
            end_entity_id (str): Target entity ID
            weight_attribute (str): Edge attribute to use as weight

        Returns:
            Tuple[List[str], float]: Tuple containing (path as list of entity IDs, total path weight)
        """
        try:
            path = dijkstra_path(self.graph, start_entity_id, end_entity_id, weight=weight_attribute)
            total_weight = sum(self.graph[path[i]][path[i+1]][weight_attribute]
                               for i in range(len(path)-1))
            return path, total_weight
        except nx.NetworkXNoPath:
            return [], float('inf')

    def get_entity_clusters(self, n_clusters: int = 5) -> Dict[int, List[str]]:
        """
        Cluster entities based on their embeddings using K-means.

        Args:
            n_clusters (int): Number of clusters to create

        Returns:
            Dict[int, List[str]]: Dictionary mapping cluster IDs to lists of entity IDs
        """
        from sklearn.cluster import KMeans

        embeddings_matrix = np.vstack(list(self.embeddings.values()))
        entity_ids = list(self.embeddings.keys())

        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        clusters = kmeans.fit_predict(embeddings_matrix)

        cluster_dict = defaultdict(list)
        for entity_id, cluster in zip(entity_ids, clusters):
            cluster_dict[int(cluster)].append(entity_id)

        return dict(cluster_dict)

    def export_to_csv(self, nodes_file: str, edges_file: str) -> None:
        """
        Export the knowledge graph to CSV files (one for nodes, one for edges).

        Args:
            nodes_file (str): Path to save nodes CSV file
            edges_file (str): Path to save edges CSV file
        """
        # Export nodes
        nodes_data = []
        for node, data in self.graph.nodes(data=True):
            node_data = {'entity_id': node, **data}
            nodes_data.append(node_data)
        pd.DataFrame(nodes_data).to_csv(nodes_file, index=False)

        # Export edges
        edges_data = []
        for u, v, data in self.graph.edges(data=True):
            edge_data = {'source': u, 'target': v, **data}
            edges_data.append(edge_data)
        pd.DataFrame(edges_data).to_csv(edges_file, index=False)

    def export_to_yaml(self, filename: str) -> None:
        """
        Export the knowledge graph to a YAML file.

        Args:
            filename (str): Path to save YAML file
        """
        data = {
            'nodes': {
                node: dict(data) for node, data in self.graph.nodes(data=True)
            },
            'edges': [
                {
                    'source': u,
                    'target': v,
                    **data
                } for u, v, data in self.graph.edges(data=True)
            ],
            'entity_counters': self.entity_counters
        }

        with open(filename, 'w') as f:
            yaml.dump(data, f, default_flow_style=False)

    def plot_graph_metrics(self, output_file: str = "graph_metrics.png") -> None:
        """
        Generate and save a plot of various graph metrics.

        Args:
            output_file (str): Path to save the metrics plot
        """
        metrics = {
            'Degree Distribution': dict(nx.degree(self.graph)),
            'Clustering Coefficients': nx.clustering(self.graph),
            'Betweenness Centrality': nx.betweenness_centrality(self.graph)
        }

        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        fig.suptitle('Graph Metrics Analysis')

        for ax, (metric_name, values) in zip(axes, metrics.items()):
            ax.hist(list(values.values()), bins=20)
            ax.set_title(metric_name)
            ax.set_xlabel('Value')
            ax.set_ylabel('Frequency')

        plt.tight_layout()
        plt.savefig(output_file)
        plt.close()

    def get_entity_statistics(self) -> Dict[str, Any]:
        """
        Calculate and return various statistics about the knowledge graph.

        Returns:
            Dict[str, Any]: Dictionary containing various graph statistics
        """
        stats = {
            'num_entities': self.graph.number_of_nodes(),
            'num_relationships': self.graph.number_of_edges(),
            'entity_types': defaultdict(int),
            'relationship_types': defaultdict(int),
            'avg_degree': sum(dict(self.graph.degree()).values()) / self.graph.number_of_nodes(),
            'density': nx.density(self.graph),
            'clustering_coefficient': nx.average_clustering(self.graph),
            'connected_components': nx.number_connected_components(self.graph)
        }

        # Count entity types
        for _, data in self.graph.nodes(data=True):
            stats['entity_types'][data['entity_type']] += 1

        # Count relationship types
        for _, _, data in self.graph.edges(data=True):
            stats['relationship_types'][data['relationship_type']] += 1

        return dict(stats)

    def find_similar_entity_groups(self, min_similarity: float = 0.8) -> List[Set[str]]:
        """
        Find groups of entities that are semantically similar based on their embeddings.

        Args:
            min_similarity (float): Minimum cosine similarity threshold

        Returns:
            List[Set[str]]: List of sets containing similar entity IDs
        """
        similar_groups = []
        processed_entities = set()

        embeddings_matrix = np.vstack(list(self.embeddings.values()))
        similarities = cosine_similarity(embeddings_matrix)
        entity_ids = list(self.embeddings.keys())

        for i, entity_id in enumerate(entity_ids):
            if entity_id in processed_entities:
                continue

            similar_entities = {entity_id}
            for j, other_id in enumerate(entity_ids):
                if i != j and similarities[i][j] >= min_similarity:
                    similar_entities.add(other_id)

            if len(similar_entities) > 1:
                similar_groups.append(similar_entities)
                processed_entities.update(similar_entities)

        return similar_groups

    def export_to_graphml(self, filename: str) -> None:
        """
        Export the knowledge graph to GraphML format.

        Args:
            filename (str): Path to save the GraphML file
        """
        # Convert numpy arrays to lists for serialization
        graph_copy = self.graph.copy()
        for node, data in graph_copy.nodes(data=True):
            for key, value in data.items():
                if isinstance(value, np.ndarray):
                    data[key] = value.tolist()

        # Write to GraphML
        nx.write_graphml(graph_copy, filename)

    def import_from_graphml(self, filename: str) -> None:

        self.graph = nx.read_graphml(filename)

    def visualize_interactive_plotly(self, layout: str = 'spring',
                                     title: str = 'Interactive Knowledge Graph',
                                     node_size_factor: float = 10,
                                     save_html: Optional[str] = None) -> go.Figure:
        """
        Create an interactive visualization using Plotly.

        Args:
            layout (str): Layout algorithm ('spring', 'circular', 'random', 'shell')
            title (str): Title of the visualization
            node_size_factor (float): Factor to scale node sizes
            save_html (Optional[str]): If provided, save the visualization to this HTML file

        Returns:
            go.Figure: Plotly figure object
        """
        # Calculate layout
        if layout == 'spring':
            pos = nx.spring_layout(self.graph)
        elif layout == 'circular':
            pos = nx.circular_layout(self.graph)
        elif layout == 'random':
            pos = nx.random_layout(self.graph)
        elif layout == 'shell':
            pos = nx.shell_layout(self.graph)
        else:
            raise ValueError(f"Unknown layout: {layout}")

        # Prepare node traces
        node_x = []
        node_y = []
        node_text = []
        node_colors = []
        node_sizes = []

        for node in self.graph.nodes():
            x, y = pos[node]
            node_x.append(x)
            node_y.append(y)

            # Create node text for hover
            node_info = self.graph.nodes[node]
            text = f"ID: {node}<br>"
            text += "<br>".join(f"{k}: {v}" for k, v in node_info.items())
            node_text.append(text)

            # Node color based on entity type
            node_colors.append(hash(node_info['entity_type']) % 20)
            node_sizes.append(self.graph.degree(node) * node_size_factor)

        node_trace = go.Scatter(
            x=node_x, y=node_y,
            mode='markers',
            hoverinfo='text',
            text=node_text,
            marker=dict(
                size=node_sizes,
                color=node_colors,
                colorscale='Viridis',
                line=dict(color='white', width=0.5)
            )
        )

        # Prepare edge traces
        edge_x = []
        edge_y = []
        edge_text = []

        for edge in self.graph.edges():
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])

            # Edge text for hover
            edge_data = self.graph.edges[edge]
            text = f"Type: {edge_data.get('relationship_type', 'Unknown')}"
            edge_text.append(text)

        edge_trace = go.Scatter(
            x=edge_x, y=edge_y,
            line=dict(width=0.5, color='#888'),
            hoverinfo='text',
            text=edge_text,
            mode='lines'
        )

        # Create figure
        fig = go.Figure(
            data=[edge_trace, node_trace],
            layout=go.Layout(
                title=title,
                showlegend=False,
                hovermode='closest',
                margin=dict(b=20,l=5,r=5,t=40),
                xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)
            )
        )

        if save_html:
            fig.write_html(save_html)

        return fig

    def visualize_interactive_pyvis(self, output_file: str = "knowledge_graph.html",
                                    height: str = "750px", width: str = "100%",
                                    bgcolor: str = "#ffffff", font_color: str = "#000000") -> None:
        """
        Create an interactive visualization using PyVis.

        Args:
            output_file (str): Path to save the HTML file
            height (str): Height of the visualization
            width (str): Width of the visualization
            bgcolor (str): Background color
            font_color (str): Font color
        """
        # Create PyVis network
        net = Network(height=height, width=width, bgcolor=bgcolor, font_color=font_color)

        # Add nodes
        for node, node_data in self.graph.nodes(data=True):
            label = f"{node}\n{node_data.get('entity_type', '')}"
            title = "\n".join(f"{k}: {v}" for k, v in node_data.items())
            color = "#{:06x}".format(hash(node_data['entity_type']) % 0xFFFFFF)
            net.add_node(node, label=label, title=title, color=color)

        # Add edges
        for source, target, edge_data in self.graph.edges(data=True):
            title = "\n".join(f"{k}: {v}" for k, v in edge_data.items())
            net.add_edge(source, target, title=title)

        # Configure physics
        net.set_options("""
        var options = {
          "physics": {
            "forceAtlas2Based": {
              "gravitationalConstant": -50,
              "centralGravity": 0.01,
              "springLength": 100,
              "springConstant": 0.08
            },
            "maxVelocity": 50,
            "minVelocity": 0.1,
            "solver": "forceAtlas2Based",
            "timestep": 0.35
          }
        }
        """)

        net.save_graph(output_file)

    def visualize_interactive_holoviews(self, title: str = "Knowledge Graph Visualization") -> hv.Graph:
        """
        Create an interactive visualization using HoloViews with Bokeh backend.

        Args:
            title (str): Title of the visualization

        Returns:
            hv.Graph: HoloViews graph object
        """
        # Prepare node data
        node_data = pd.DataFrame([
            {'id': node, 'entity_type': data['entity_type'], **data}
            for node, data in self.graph.nodes(data=True)
        ])

        # Prepare edge data
        edge_data = pd.DataFrame([
            {'source': source, 'target': target, **data}
            for source, target, data in self.graph.edges(data=True)
        ])

        # Create graph
        graph = hv.Graph(
            (edge_data, node_data),
            vdims=['entity_type']
        )

        # Style the visualization
        graph.opts(
            opts.Graph(
                tools=['hover', 'tap', 'box_select', 'lasso_select'],
                node_size=10,
                node_color='entity_type',
                cmap='Category20',
                width=800,
                height=600,
                title=title,
                edge_line_width=1,
                edge_alpha=0.5,
                node_alpha=0.8,
                xaxis=None,
                yaxis=None
            )
        )

        return graph

    def create_timeline_visualization(self,
                                      date_attribute: str = 'timestamp',
                                      save_html: Optional[str] = None) -> go.Figure:
        """
        Create an interactive timeline visualization of entities and relationships.

        Args:
            date_attribute (str): Name of the attribute containing timestamp information
            save_html (Optional[str]): If provided, save the visualization to this HTML file

        Returns:
            go.Figure: Plotly figure object
        """
        # Collect timeline data
        timeline_data = []

        for node, data in self.graph.nodes(data=True):
            if date_attribute in data:
                timeline_data.append({
                    'id': node,
                    'type': 'entity',
                    'entity_type': data['entity_type'],
                    'date': data[date_attribute],
                    'description': f"Entity: {node} ({data['entity_type']})"
                })

        for source, target, data in self.graph.edges(data=True):
            if date_attribute in data:
                timeline_data.append({
                    'id': f"{source}-{target}",
                    'type': 'relationship',
                    'relationship_type': data.get('relationship_type', 'Unknown'),
                    'date': data[date_attribute],
                    'description': f"Relationship: {source} -> {target}"
                })

        if not timeline_data:
            raise ValueError(f"No {date_attribute} attribute found in entities or relationships")

        df = pd.DataFrame(timeline_data)

        # Create figure
        fig = go.Figure()

        # Add entities
        entity_df = df[df['type'] == 'entity']
        fig.add_trace(go.Scatter(
            x=entity_df['date'],
            y=entity_df['entity_type'],
            mode='markers',
            name='Entities',
            text=entity_df['description'],
            marker=dict(size=10, symbol='circle')
        ))

        # Add relationships
        rel_df = df[df['type'] == 'relationship']
        fig.add_trace(go.Scatter(
            x=rel_df['date'],
            y=rel_df['relationship_type'],
            mode='markers',
            name='Relationships',
            text=rel_df['description'],
            marker=dict(size=10, symbol='diamond')
        ))

        # Update layout
        fig.update_layout(
            title="Knowledge Graph Timeline",
            xaxis_title="Date",
            yaxis_title="Type",
            hovermode='closest',
            showlegend=True
        )

        if save_html:
            fig.write_html(save_html)

        return fig

    def create_entity_relationship_sunburst(self, save_html: Optional[str] = None) -> go.Figure:
        """
        Create a sunburst visualization showing entity types and their relationships.

        Args:
            save_html (Optional[str]): If provided, save the visualization to this HTML file

        Returns:
            go.Figure: Plotly figure object
        """
        # Collect hierarchy data
        hierarchy_data = []

        # Add entity types as first level
        entity_types = set(nx.get_node_attributes(self.graph, 'entity_type').values())
        for entity_type in entity_types:
            hierarchy_data.append({
                'id': entity_type,
                'parent': '',
                'value': sum(1 for _, data in self.graph.nodes(data=True)
                             if data['entity_type'] == entity_type)
            })

            # Add relationships as second level
            for _, _, edge_data in self.graph.edges(data=True):
                rel_type = edge_data.get('relationship_type', 'Unknown')
                hierarchy_data.append({
                    'id': f"{entity_type}-{rel_type}",
                    'parent': entity_type,
                    'value': 1
                })

        # Create sunburst
        fig = go.Figure(go.Sunburst(
            ids=[item['id'] for item in hierarchy_data],
            parents=[item['parent'] for item in hierarchy_data],
            values=[item['value'] for item in hierarchy_data],
            branchvalues='total'
        ))

        fig.update_layout(
            title="Entity Types and Relationships Hierarchy",
            width=800,
            height=800
        )

        if save_html:
            fig.write_html(save_html)

        return fig