import abc
import re

class TextSplitter(abc.ABC):

    @abc.abstractmethod
    def get_chunks(self, text):
        pass

class SimpleTextSplitter(TextSplitter):
    """
    A class for splitting text into consecutive overlapping (or non-overlapping) chunks.

    Parameters:
    -----------
    text : str
        The string to be chunked.
    chunk_size : int
        The size (length) of each chunk.
    overlap : int, optional
        Number of characters of overlap between consecutive chunks. Defaults to 0.

    Raises:
    -------
    ValueError
        If chunk_size <= 0, if overlap < 0, or if overlap >= chunk_size.
    """

    def __init__(self, chunk_size, overlap=0):
        if chunk_size <= 0:
            raise ValueError("chunk_size must be a positive integer.")
        if overlap < 0:
            raise ValueError("overlap cannot be negative.")
        if overlap >= chunk_size:
            raise ValueError("overlap must be strictly less than chunk_size.")
        self.chunk_size = chunk_size
        self.overlap = overlap

    def get_chunks(self, text):
        """
        Splits the text into chunks of size `chunk_size`, overlapping by `overlap` characters.

        Args:
            text (str): The text to be chunked.
        Returns:
        --------
        list of str
           A list of chunked text segments.
        """
        chunks = []
        start = 0
        while start < len(text):
            end = start + self.chunk_size
            chunks.append(text[start:end])
            start = end - self.overlap
        return chunks


class RecursiveCharacterTextSplitter(TextSplitter):
    """
    A class that recursively splits text by a hierarchy of separators
    and eventually falls back to fixed-size splitting with overlap
    once all separators have been exhausted.

    Parameters:
    -----------
    separators : list of str
        Ordered list of separators to progressively split the text.
        If an element is the empty string (""), the text is chunked into individual characters.
    chunk_size : int
        Maximum size of each final chunk.
    chunk_overlap : int
        Number of characters to overlap when creating the final chunks.
    length_function : callable, optional
        A function that returns the length of a piece of text. Defaults to `len`.
    keep_separator : bool, optional
        Whether to keep the separator attached to each piece. If True,
        the separator is included in the piece that precedes it.

    Raises:
    -------
    ValueError
        If chunk_size <= 0, if chunk_overlap < 0, or if chunk_overlap >= chunk_size.
    """

    def __init__(self, separators, chunk_size, chunk_overlap, length_function=len, keep_separator=False):
        if chunk_size <= 0:
            raise ValueError("chunk_size must be a positive integer.")
        if chunk_overlap < 0:
            raise ValueError("chunk_overlap cannot be negative.")
        if chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap must be strictly less than chunk_size.")
        self.separators = separators
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.length_function = length_function
        self.keep_separator = keep_separator

    def get_chunks(self, text):
        """
        Recursively splits `text` by the `depth`-th separator in `self.separators`.
        If we run out of separators, we split the text into fixed-size chunks.

        Parameters:
        -----------
        text : str
            The text to split.
        depth : int
            Current level of recursion, indicating which separator is in use.

        Returns:
        --------
        list of str
            A list of pieces after applying the splitting logic.
        """
        return self._split_text(text)

    def _split_text(self, text, depth=0):

        if len(text) <= self.chunk_size:
            return [text]

        # Base case: if all separators are exhausted, use fixed-size splitting.
        if depth == len(self.separators):
            return self._split_into_fixed_size(text)

        current_separator = self.separators[depth]
        if current_separator == "":
            return list(text)

        if self.keep_separator:
            # Use regex to split while capturing the separator.
            pieces = re.split(f"({re.escape(current_separator)})", text)
            merged_pieces = []
            for i in range(0, len(pieces), 2):
                piece = pieces[i]
                if i + 1 < len(pieces) and pieces[i + 1] == current_separator:
                    piece += pieces[i + 1]
                merged_pieces.append(piece)
            pieces = merged_pieces
        else:
            pieces = text.split(current_separator)

        refined_pieces = []
        for piece in pieces:
            if self.length_function(piece) > self.chunk_size:
                refined_pieces.extend(self._split_text(piece, depth + 1))
            else:
                refined_pieces.append(piece)

        if depth == 0 and self.separators[depth] not in text:
            return refined_pieces

        # At the top level, merge pieces into valid chunks.
        if depth == 0:
            return self._merge_pieces(refined_pieces)
        return refined_pieces

    def _split_into_fixed_size(self, text):
        """
        Splits `text` into chunks of size `self.chunk_size`, overlapping by `self.chunk_overlap`.

        Parameters:
        -----------
        text : str
            The text to split.

        Returns:
        --------
        list of str
            The list of chunks of size up to `self.chunk_size`, using overlap.
        """
        step = self.chunk_size - self.chunk_overlap  # guaranteed > 0
        if not text:
            return []
        chunks = [text[i: i + self.chunk_size] for i in range(0, len(text), step)]
        if len(chunks) > 1 and len(chunks[-1]) < self.chunk_overlap:
            chunks[-2] += chunks[-1]
            chunks.pop()
        return chunks

    def _merge_pieces(self, pieces):
        """
        Merges pieces into chunks of size <= self.chunk_size. When a chunk
        hits exactly self.chunk_size, subsequent overlaps are handled.

        Parameters:
        -----------
        pieces : list of str
            Pieces of text that need to be merged into chunks.

        Returns:
        --------
        list of str
            The merged list of chunks, each of length <= self.chunk_size.
        """
        if not pieces:
            return []
        merged = []
        current_chunk = pieces[0]
        for piece in pieces[1:]:
            if self.length_function(current_chunk + piece) <= self.chunk_size:
                current_chunk += piece
            else:
                merged.append(current_chunk)
                if len(current_chunk) == self.chunk_size:
                    # Maintain overlap if the current chunk is exactly full.
                    current_chunk = current_chunk[-self.chunk_overlap:] + piece
                else:
                    current_chunk = piece
        merged.append(current_chunk)
        return merged
