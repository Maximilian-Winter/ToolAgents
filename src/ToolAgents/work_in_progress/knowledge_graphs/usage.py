from extended_knowledge_graph import KnowledgeGraph, Entity, EntityQuery
from typing import Dict, Any
import json


def create_movie_knowledge_graph() -> KnowledgeGraph:
    """
    Creates a knowledge graph for a movie recommendation system.
    """
    kg = KnowledgeGraph()

    # Add movies
    movies = [
        {
            "title": "The Matrix",
            "year": 1999,
            "genre": ["Science Fiction", "Action"],
            "rating": 8.7
        },
        {
            "title": "Inception",
            "year": 2010,
            "genre": ["Science Fiction", "Action", "Thriller"],
            "rating": 8.8
        },
        {
            "title": "The Dark Knight",
            "year": 2008,
            "genre": ["Action", "Crime", "Drama"],
            "rating": 9.0
        }
    ]

    movie_ids = {}
    for movie in movies:
        entity = Entity(
            entity_type="Movie",
            attributes={
                "title": movie["title"],
                "year": movie["year"],
                "genre": json.dumps(movie["genre"]),  # Convert list to string for storage
                "rating": movie["rating"]
            }
        )
        movie_ids[movie["title"]] = kg.add_entity(entity)

    # Add directors
    directors = [
        {
            "name": "Wachowski Sisters",
            "movies": ["The Matrix"],
            "style": "Visual Innovation"
        },
        {
            "name": "Christopher Nolan",
            "movies": ["Inception", "The Dark Knight"],
            "style": "Complex Narratives"
        }
    ]

    director_ids = {}
    for director in directors:
        entity = Entity(
            entity_type="Director",
            attributes={
                "name": director["name"],
                "style": director["style"]
            }
        )
        director_id = kg.add_entity(entity)
        director_ids[director["name"]] = director_id

        # Add relationships between directors and their movies
        for movie_title in director["movies"]:
            kg.add_relationship(
                director_id,
                "DIRECTED",
                movie_ids[movie_title],
                {"year": movies[next(i for i, m in enumerate(movies) if m["title"] == movie_title)]["year"]}
            )

    # Add actors
    actors = [
        {
            "name": "Keanu Reeves",
            "movies": ["The Matrix"],
            "awards": 2
        },
        {
            "name": "Leonardo DiCaprio",
            "movies": ["Inception"],
            "awards": 7
        },
        {
            "name": "Christian Bale",
            "movies": ["The Dark Knight"],
            "awards": 4
        }
    ]

    actor_ids = {}
    for actor in actors:
        entity = Entity(
            entity_type="Actor",
            attributes={
                "name": actor["name"],
                "awards": actor["awards"]
            }
        )
        actor_id = kg.add_entity(entity)
        actor_ids[actor["name"]] = actor_id

        # Add relationships between actors and their movies
        for movie_title in actor["movies"]:
            kg.add_relationship(
                actor_id,
                "ACTED_IN",
                movie_ids[movie_title],
                {"role": "Lead Actor"}
            )

    return kg


def demonstrate_knowledge_graph_features(kg: KnowledgeGraph):
    """
    Demonstrates various features of the knowledge graph.
    """
    print("1. Query all movies:")
    movie_query = EntityQuery(entity_type="Movie")
    print(kg.query_entities(movie_query))

    print("\n2. Semantic search for action movies:")
    print(kg.semantic_search("action movie with crime"))

    print("\n3. Find relationships for Christopher Nolan:")
    director_query = EntityQuery(
        entity_type="Director",
        attribute_filter={"name": "Christopher Nolan"}
    )
    director_result = kg.query_entities(director_query)
    director_id = director_result.split()[3]  # Extract ID from query result
    print(kg.query_relationships(director_id))

    print("\n4. Get central entities:")
    central_entities = kg.get_central_entities(method='betweenness', top_k=3)
    for entity_id, centrality in central_entities:
        print(f"Entity: {entity_id}, Centrality: {centrality:.4f}")
        print(kg.get_entity_details(entity_id))

    print("\n5. Get graph statistics:")
    stats = kg.get_entity_statistics()
    print(json.dumps(stats, indent=2))

    # Generate visualizations
    print("\n6. Generating visualizations...")
    kg.visualize(output_file="movie_knowledge_graph")
    kg.visualize_interactive_plotly(save_html="movie_graph_interactive.html")
    kg.plot_graph_metrics(output_file="movie_graph_metrics.png")


def main():
    # Create the knowledge graph
    kg = create_movie_knowledge_graph()

    # Demonstrate features
    demonstrate_knowledge_graph_features(kg)

    # Save the knowledge graph
    kg.save_to_file("movie_knowledge_graph.json")

    # Export to different formats
    kg.export_to_csv("movie_nodes.csv", "movie_edges.csv")
    kg.export_to_yaml("movie_graph.yaml")
    kg.export_to_graphml("movie_graph.graphml")


if __name__ == "__main__":
    main()