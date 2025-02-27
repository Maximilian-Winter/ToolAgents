"""
Example demonstrating the use of HierarchicalSemanticMemory.

This script shows how to create and use a hierarchical memory system,
including organizing memories into parent-child relationships and
retrieving them with hierarchical context.
"""

import os
import sys
from datetime import datetime
from pprint import pprint

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from ToolAgents.agent_memory import (
    HierarchicalSemanticMemory,
    HierarchicalMemoryConfig,
    HierarchicalRelationshipType,
)


def create_knowledge_hierarchy():
    """Create a sample knowledge hierarchy about programming."""

    # Initialize the hierarchical memory
    memory = HierarchicalSemanticMemory()

    # Create top-level category nodes
    programming = memory.store(
        "Programming: The art and science of instructing computers to perform specific tasks through code."
    )
    print(f"Created 'Programming' node: {programming}")

    web_dev = memory.store(
        "Web Development: Creating websites and web applications using various technologies.",
        parent_id=programming,
    )
    print(f"Created 'Web Development' node: {web_dev}")

    machine_learning = memory.store(
        "Machine Learning: Teaching computers to learn patterns from data and make decisions.",
        parent_id=programming,
    )
    print(f"Created 'Machine Learning' node: {machine_learning}")

    data_science = memory.store(
        "Data Science: Extracting knowledge and insights from structured and unstructured data.",
        parent_id=programming,
    )
    print(f"Created 'Data Science' node: {data_science}")

    # Add second-level nodes under Web Development
    frontend = memory.store(
        "Frontend: The client-side of web development focusing on user interfaces and experiences.",
        parent_id=web_dev,
    )
    print(f"Created 'Frontend' node: {frontend}")

    backend = memory.store(
        "Backend: The server-side of web development handling business logic and data storage.",
        parent_id=web_dev,
    )
    print(f"Created 'Backend' node: {backend}")

    # Add third-level nodes under Frontend
    memory.store(
        "HTML: Hypertext Markup Language is the standard markup language for documents designed to be displayed in a web browser.",
        parent_id=frontend,
    )
    memory.store(
        "CSS: Cascading Style Sheets is a style sheet language used for describing the presentation of a document written in HTML.",
        parent_id=frontend,
    )
    memory.store(
        "JavaScript: A programming language that enables interactive web pages and is an essential part of web applications.",
        parent_id=frontend,
    )
    memory.store(
        "React: A JavaScript library for building user interfaces, particularly single-page applications.",
        parent_id=frontend,
    )
    memory.store(
        "Vue.js: A progressive JavaScript framework for building user interfaces and single-page applications.",
        parent_id=frontend,
    )

    # Add third-level nodes under Backend
    memory.store(
        "Node.js: A JavaScript runtime built on Chrome's V8 JavaScript engine for building fast and scalable network applications.",
        parent_id=backend,
    )
    memory.store(
        "Django: A high-level Python web framework that encourages rapid development and clean, pragmatic design.",
        parent_id=backend,
    )
    memory.store(
        "Flask: A lightweight WSGI web application framework in Python designed to make getting started quick and easy.",
        parent_id=backend,
    )
    memory.store(
        "Database: Systems used to store, manage, and retrieve data efficiently for applications.",
        parent_id=backend,
    )

    # Add second-level nodes under Machine Learning
    supervised = memory.store(
        "Supervised Learning: Training algorithm on labeled data to make predictions.",
        parent_id=machine_learning,
    )
    unsupervised = memory.store(
        "Unsupervised Learning: Finding patterns in unlabeled data.",
        parent_id=machine_learning,
    )
    reinforcement = memory.store(
        "Reinforcement Learning: Training agents to make sequences of decisions by rewarding desired behaviors.",
        parent_id=machine_learning,
    )

    # Add third-level nodes under Supervised Learning
    memory.store(
        "Regression: Predicting continuous values based on input features.",
        parent_id=supervised,
    )
    memory.store(
        "Classification: Assigning categories to input data based on training examples.",
        parent_id=supervised,
    )
    memory.store(
        "Random Forests: An ensemble learning method for classification and regression using multiple decision trees.",
        parent_id=supervised,
    )

    # Add nodes under Data Science with different relationship types
    visualization = memory.store(
        "Data Visualization: The graphical representation of data to identify patterns and trends.",
        parent_id=data_science,
        relationship_type=HierarchicalRelationshipType.THEMATIC,
    )

    memory.store(
        "Data Cleaning: The process of detecting and correcting corrupt or inaccurate records from a dataset.",
        parent_id=data_science,
        relationship_type=HierarchicalRelationshipType.TEMPORAL,
    )

    # Create connections across categories (e.g., relating data science to machine learning)
    memory.create_relationship(
        data_science, machine_learning, HierarchicalRelationshipType.THEMATIC
    )

    # Store some related but more specific entries
    memory.store(
        "Matplotlib: A plotting library for Python used for data visualization.",
        parent_id=visualization,
    )
    memory.store(
        "D3.js: A JavaScript library for producing dynamic, interactive data visualizations in web browsers.",
        parent_id=visualization,
    )

    return memory


def test_queries(memory):
    """Test various queries against the hierarchical memory."""

    print("\n----- Basic Query -----")
    results = memory.recall("How to create interactive websites", n_results=3)
    print_results(results)

    print("\n----- Query with Parent Context -----")
    results = memory.recall("Python web frameworks", n_results=3)
    print_results(results)

    print("\n----- Query with Child Context -----")
    results = memory.recall("web development", n_results=2)
    print_results(results)

    print("\n----- Cross-Hierarchy Query -----")
    results = memory.recall("visualizing data in web applications", n_results=3)
    print_results(results)

    print("\n----- Memory Stats -----")
    stats = memory.get_stats()
    pprint(stats)


def test_memory_operations(memory):
    """Test various memory operations."""

    print("\n----- Testing Node Movement -----")
    # First, retrieve ids of relevant nodes
    results = memory.recall("visualization", n_results=1)
    if not results:
        print("No visualization node found. Creating one for testing...")
        visualization_id = memory.store(
            "Data Visualization: Creating visual representations of data."
        )
    else:
        visualization_id = results[0]["metadata"]["node_id"]

    results = memory.recall("web development", n_results=1)
    if not results:
        print("No web development node found. Creating one for testing...")
        web_dev_id = memory.store(
            "Web Development: Creating websites and web applications."
        )
    else:
        web_dev_id = results[0]["metadata"]["node_id"]

    # Move data visualization to be under web development
    print(
        f"Moving visualization node {visualization_id} under web development {web_dev_id}"
    )
    memory.move_node(visualization_id, web_dev_id)

    # Check the new structure
    print("\nAfter moving visualization to web development:")
    web_dev_node = memory.get_node(web_dev_id)
    if web_dev_node:
        children = memory.get_children(web_dev_id)
        print(f"Web Development now has {len(children)} children")
        for child in children:
            print(f"- {child.content}")

        print("\n----- Testing Summarization -----")
        # Summarize the web development node with its new children
        summary = memory.summarize_node_with_children(web_dev_id)
        if summary:
            print(f"Summarized content: {summary[:200]}...")
        else:
            print("No summary generated.")
    else:
        print("Web development node not found after move operation.")


def auto_organize_test():
    """Test the auto-organization capabilities."""

    print("\n----- Auto-Organize Test -----")
    memory = HierarchicalSemanticMemory(
        HierarchicalMemoryConfig(auto_organize_threshold=5)
    )

    # Create a bunch of flat programming language nodes
    pl_nodes = []
    pl_nodes.append(
        memory.store(
            "Python: a high-level, interpreted programming language known for its readability."
        )
    )
    pl_nodes.append(
        memory.store(
            "JavaScript: a programming language that enables interactive web pages."
        )
    )
    pl_nodes.append(
        memory.store(
            "Java: a general-purpose programming language designed to be class-based and object-oriented."
        )
    )
    pl_nodes.append(
        memory.store(
            "C++: a high-level, general-purpose programming language with imperative and object-oriented features."
        )
    )
    pl_nodes.append(
        memory.store(
            "Go: a statically typed, compiled programming language designed at Google."
        )
    )
    pl_nodes.append(
        memory.store(
            "Ruby: a dynamic, open source programming language focusing on simplicity and productivity."
        )
    )
    pl_nodes.append(
        memory.store(
            "Rust: a multi-paradigm programming language focused on performance and safety."
        )
    )

    # Let's add some databases as well to see if they get grouped separately
    db_nodes = []
    db_nodes.append(
        memory.store("MySQL: an open-source relational database management system.")
    )
    db_nodes.append(
        memory.store(
            "PostgreSQL: a powerful, open source object-relational database system."
        )
    )
    db_nodes.append(
        memory.store(
            "MongoDB: a cross-platform document-oriented NoSQL database program."
        )
    )
    db_nodes.append(
        memory.store(
            "Redis: an in-memory data structure store used as a database, cache, and message broker."
        )
    )
    db_nodes.append(
        memory.store(
            "SQLite: a C-language library that implements a small, fast, self-contained SQL database engine."
        )
    )

    # Auto-organize the flat nodes
    all_nodes = pl_nodes + db_nodes
    top_level_nodes = memory.auto_organize(all_nodes)

    print(f"Auto-organization created {len(top_level_nodes)} top-level categories")

    # Print the resulting hierarchy
    for parent_id in top_level_nodes:
        parent = memory.get_node(parent_id)
        print(f"\nTop-level category: {parent.content}")

        children = memory.get_children(parent_id)
        print(f"Contains {len(children)} items:")
        for child in children:
            print(f"  - {child.content}")


def print_results(results):
    """Print the recall results in a readable format."""
    for i, result in enumerate(results):
        print(f"\nResult {i+1}: {result['content'][:100]}...")
        print(
            f"Similarity: {result['similarity']:.4f}, Rank Score: {result.get('rank_score', 0):.4f}"
        )

        if "hierarchy" in result:
            if result["hierarchy"]["parents"]:
                print("Parent context:")
                for parent in result["hierarchy"]["parents"]:
                    print(f"  - Level {parent['level']}: {parent['content'][:50]}...")

            if result["hierarchy"]["children"]["count"] > 0:
                print(f"Has {result['hierarchy']['children']['count']} children")
                for child in result["hierarchy"]["children"]["children"][
                    :2
                ]:  # Show just first two
                    print(f"  - {child['content'][:50]}...")
                if result["hierarchy"]["children"]["count"] > 2:
                    print(
                        f"  - ... and {result['hierarchy']['children']['count'] - 2} more"
                    )


if __name__ == "__main__":
    # Create the knowledge hierarchy
    memory = create_knowledge_hierarchy()

    # Test various queries
    test_queries(memory)

    # Test memory operations
    test_memory_operations(memory)

    # Test auto-organization
    auto_organize_test()
