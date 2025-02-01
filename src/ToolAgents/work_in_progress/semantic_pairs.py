from sentence_transformers import SentenceTransformer
import numpy as np
from typing import List, Dict, Tuple


def cosine_similarity(vec_a: np.ndarray, vec_b: np.ndarray) -> float:
    """Calculate cosine similarity between two vectors."""
    return np.dot(vec_a, vec_b) / (np.linalg.norm(vec_a) * np.linalg.norm(vec_b))


def filter_and_pair_messages(chat_history: List[Dict[str, str]]) -> List[Tuple[Dict[str, str], Dict[str, str]]]:
    """
    Filter out system messages and create pairs of user-assistant messages.

    Args:
        chat_history: List of message dictionaries with 'role' and 'content' keys

    Returns:
        List of message pairs (tuples)
    """
    # Filter out system messages
    filtered_messages = [msg for msg in chat_history if msg['role'] != 'system']

    # Create pairs of messages
    message_pairs = []
    for i in range(0, len(filtered_messages) - 1, 2):
        if i + 1 < len(filtered_messages):
            message_pairs.append((filtered_messages[i], filtered_messages[i + 1]))

    return message_pairs


def embed_message_pairs(message_pairs: List[Tuple[Dict[str, str], Dict[str, str]]],
                        model: SentenceTransformer) -> np.ndarray:
    """
    Compute embeddings for message pairs by concatenating their contents.

    Args:
        message_pairs: List of message pair tuples
        model: SentenceTransformer model instance

    Returns:
        Array of embeddings for message pairs
    """
    pair_texts = [
        f"{pair[0]['content']} [SEP] {pair[1]['content']}"
        for pair in message_pairs
    ]
    return model.encode(pair_texts, convert_to_numpy=True)


def semantic_split_episodes(message_pairs: List[Tuple[Dict[str, str], Dict[str, str]]],
                            embeddings: np.ndarray,
                            similarity_threshold: float = 0.8) -> List[List[Tuple[Dict[str, str], Dict[str, str]]]]:
    """
    Split message pairs into episodes based on semantic similarity.

    Args:
        message_pairs: List of message pair tuples
        embeddings: Array of embeddings for message pairs
        similarity_threshold: Threshold for creating new episodes

    Returns:
        List of episodes, where each episode is a list of message pairs
    """
    if not message_pairs:
        return []

    episodes = []
    current_episode = [message_pairs[0]]

    for i in range(1, len(message_pairs)):
        sim = cosine_similarity(embeddings[i], embeddings[i - 1])
        if sim >= similarity_threshold:
            current_episode.append(message_pairs[i])
        else:
            episodes.append(current_episode)
            current_episode = [message_pairs[i]]

    if current_episode:
        episodes.append(current_episode)

    return episodes


if __name__ == "__main__":
    # Example chat history
    chat_history = [
        {"role": "system", "content": "Welcome! How can I help you today?"},
        {"role": "user", "content": "Hi! I'm having trouble with my account."},
        {"role": "assistant", "content": "Could you describe the problem more specifically?"},
        {"role": "user", "content": "Sure, I'm getting a 404 error when I log in."},
        {"role": "assistant", "content": "Let's troubleshoot that login error."},
        {"role": "user", "content": "Actually, I also have a billing issue to discuss."},
        {"role": "assistant", "content": "I'll help you with the billing concern."}
    ]

    # Load the model
    model_name = "sentence-transformers/all-MiniLM-L6-v2"
    model = SentenceTransformer(model_name)

    # Process the chat history
    message_pairs = filter_and_pair_messages(chat_history)
    embeddings = embed_message_pairs(message_pairs, model)
    episodes = semantic_split_episodes(message_pairs, embeddings, similarity_threshold=0.4)

    # Print the results
    for idx, episode in enumerate(episodes):
        print(f"\nEpisode {idx + 1}:")
        for pair in episode:
            print(f"\nPair:")
            print(f"  {pair[0]['role']}: {pair[0]['content']}")
            print(f"  {pair[1]['role']}: {pair[1]['content']}")
        print("-" * 50)