from sentence_transformers import SentenceTransformer
import numpy as np


def cosine_similarity(vec_a, vec_b):
    return np.dot(vec_a, vec_b) / (np.linalg.norm(vec_a) * np.linalg.norm(vec_b))


def embed_chat_history(chat_history, model):
    texts = [msg["content"] for msg in chat_history]
    return model.encode(texts, convert_to_numpy=True)


def semantic_split_episodes(chat_history, embeddings, similarity_threshold=0.8):
    if not chat_history:
        return []

    episodes = []
    current_episode = [chat_history[0]]

    for i in range(1, len(chat_history)):
        sim = cosine_similarity(embeddings[i], embeddings[i - 1])
        if sim >= similarity_threshold:
            current_episode.append(chat_history[i])
        else:
            episodes.append(current_episode)
            current_episode = [chat_history[i]]

    # Don’t forget to add the last episode
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
        {"role": "assistant", "content": "Let's see... That might be related to your security settings."},
        {"role": "assistant", "content": "Another angle is if your account isn't fully verified yet."},
        {"role": "user", "content": "Actually, I also have a billing issue I need to discuss."},
    ]

    # Load a model
    model_name = "sentence-transformers/all-MiniLM-L6-v2"
    model = SentenceTransformer(model_name)

    # Compute embeddings
    embeddings = embed_chat_history(chat_history, model)

    # Perform semantic splitting
    episodes = semantic_split_episodes(chat_history, embeddings, similarity_threshold=0.4)

    # Print the results
    for idx, episode in enumerate(episodes):
        print(f"Episode {idx + 1}:")
        for msg in episode:
            print(f"  {msg['role']}: {msg['content']}")
        print("-" * 50)