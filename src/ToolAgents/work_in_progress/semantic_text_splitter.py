import re

try:
    import spacy
    from sentence_transformers import SentenceTransformer, util
except ImportError:
    raise ImportError(
        "Please install required dependencies: "
        "`pip install spacy sentence-transformers` and download a SpaCy model, e.g., "
        "`python -m spacy download en_core_web_sm`."
    )


class SemanticTextSplitter:
    """
    A class for splitting text into semantically coherent chunks.
    It uses SpaCy for sentence segmentation and a Sentence-BERT model
    for comparing embeddings of adjacent sentences.

    Parameters:
    -----------
    model_name : str
        The Hugging Face or Sentence-Transformers model name to use for embeddings.
        E.g., 'sentence-transformers/all-MiniLM-L6-v2'.
    chunk_size : int
        Maximum size of each final chunk (in terms of characters).
    chunk_overlap : int
        Number of characters to overlap when final chunking is performed.
    similarity_threshold : float
        Cosine similarity threshold. If consecutive sentences have a similarity above
        this threshold, they are merged into the same semantic chunk.
    spacy_model : str, optional
        The SpaCy language model to load, e.g. 'en_core_web_sm'. Defaults to 'en_core_web_sm'.
    """

    def __init__(
            self,
            model_name,
            chunk_size,
            chunk_overlap,
            similarity_threshold=0.5,
            spacy_model="en_core_web_sm",
    ):
        # Validate parameters
        if chunk_size <= 0:
            raise ValueError("chunk_size must be a positive integer.")
        if chunk_overlap < 0:
            raise ValueError("chunk_overlap cannot be negative.")
        if chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap must be strictly less than chunk_size.")
        if similarity_threshold < 0.0 or similarity_threshold > 1.0:
            raise ValueError("similarity_threshold must be between 0.0 and 1.0.")

        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.similarity_threshold = similarity_threshold

        # Load the SpaCy model for sentence segmentation
        self.nlp = spacy.load(spacy_model)

        # Load a Sentence-BERT model from sentence-transformers
        self.embed_model = SentenceTransformer(model_name)

    def split_text(self, text):
        """
        Splits the given text into semantic chunks.

        Steps:
        1. Sentence segmentation with SpaCy.
        2. Embed each sentence with Sentence-BERT.
        3. Merge sentences into cohesive chunks if their similarity >= self.similarity_threshold.
        4. Finally, chunk them again (if needed) based on self.chunk_size and self.chunk_overlap
           to respect maximum chunk length.

        Parameters:
        -----------
        text : str
            The text to be split into chunks.

        Returns:
        --------
        list of str
            A list of semantically coherent chunks, each chunk having length <= self.chunk_size.
        """
        if not text.strip():
            return []

        # Step 1: Sentence segmentation
        sentences = self._split_into_sentences(text)

        # Step 2: Get embeddings of each sentence
        embeddings = self._embed_sentences(sentences)

        # Step 3: Semantic grouping of sentences
        semantically_grouped_sentences = self._semantic_grouping(sentences, embeddings)

        # Step 4: Enforce chunk_size with overlap
        final_chunks = []
        for chunk_text in semantically_grouped_sentences:
            # If a single chunk is too long, break it down further
            if len(chunk_text) > self.chunk_size:
                final_chunks.extend(self._split_into_fixed_size(chunk_text))
            else:
                final_chunks.append(chunk_text)

        return final_chunks

    def _split_into_sentences(self, text):
        """
        Uses SpaCy to split the text into sentences.

        Parameters:
        -----------
        text : str
            The text to split into sentences.

        Returns:
        --------
        list of str
            A list of sentences.
        """
        doc = self.nlp(text)
        sentences = [sent.text.strip() for sent in doc.sents if sent.text.strip()]
        return sentences

    def _embed_sentences(self, sentences):
        """
        Computes embeddings for each sentence using Sentence-BERT.

        Parameters:
        -----------
        sentences : list of str
            The list of sentences to embed.

        Returns:
        --------
        torch.Tensor
            A tensor of shape (num_sentences, embedding_dim).
        """
        embeddings = self.embed_model.encode(sentences, convert_to_tensor=True)
        return embeddings

    def _semantic_grouping(self, sentences, embeddings):
        """
        Groups consecutive sentences into chunks based on a cosine similarity threshold.

        Parameters:
        -----------
        sentences : list of str
            The original sentences.
        embeddings : torch.Tensor
            Embeddings of each sentence, parallel in order with 'sentences'.

        Returns:
        --------
        list of str
            List of grouped text chunks before final size-based splitting.
        """
        chunks = []
        current_chunk_sentences = [sentences[0]]

        for i in range(1, len(sentences)):
            # Compute similarity between current sentence and previous sentence
            similarity = float(util.cos_sim(embeddings[i], embeddings[i - 1]))

            if similarity >= self.similarity_threshold:
                # If the sentences are semantically similar, merge them
                current_chunk_sentences.append(sentences[i])
            else:
                # Start a new chunk
                chunks.append(" ".join(current_chunk_sentences))
                current_chunk_sentences = [sentences[i]]

        # Append the last group
        if current_chunk_sentences:
            chunks.append(" ".join(current_chunk_sentences))

        return chunks

    def _split_into_fixed_size(self, text):
        """
        Splits `text` into chunks of size `self.chunk_size`, with overlap of `self.chunk_overlap`.

        Parameters:
        -----------
        text : str
            The text to split.

        Returns:
        --------
        list of str
            The list of chunks of size up to `self.chunk_size`, using overlap.
        """
        size = self.chunk_size
        overlap = self.chunk_overlap
        step = size - overlap  # guaranteed to be > 0 because overlap < size

        chunks = []
        i = 0
        while i < len(text):
            end = i + size
            chunks.append(text[i:end])
            i += step

        # Optionally, if the last chunk is too small (< overlap), you might merge it back:
        # (Uncomment if you prefer that logic, or adjust to your use case)
        if len(chunks) > 1 and len(chunks[-1]) < overlap:
            chunks[-2] += chunks[-1]
            chunks.pop()

        return chunks


# --- Example usage (for illustration) ---
if __name__ == "__main__":
    sample_text = (
        "Artificial intelligence (AI) refers to the simulation of human intelligence in machines "
        "that are programmed to think like humans and mimic their actions. "
        "The term may also be applied to any machine that exhibits traits associated with a human mind "
        "such as learning and problem-solving. "
        "AI is a broad field. "
        "Modern AI often revolves around deep learning, natural language processing, and machine vision. "
        "AI is being applied in a wide range of fields, from healthcare to finance. "
    )

    # Initialize the semantic text splitter
    semantic_splitter = SemanticTextSplitter(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        chunk_size=100,
        chunk_overlap=0,
        similarity_threshold=0.6,
    )

    # Perform semantic chunking
    semantic_chunks = semantic_splitter.split_text(sample_text)

    print("Semantic Chunks:")
    for idx, ch in enumerate(semantic_chunks, 1):
        print(f"Chunk {idx} (length={len(ch)}):\n{ch}\n")
