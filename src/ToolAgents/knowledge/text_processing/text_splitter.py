import abc
import re


class TextSplitter(abc.ABC):

    @abc.abstractmethod
    def get_chunks(self, text):
        pass


class NonTextSplitter(TextSplitter):
    """
    A class that does not split text at all, returning the entire text as a single chunk.

    Parameters:
    -----------
    text : str
        The string to be chunked.
    """

    def get_chunks(self, text):
        return [text]


class SimpleTextSplitter(TextSplitter):
    """
    A class for splitting text into consecutive overlapping (or non-overlapping) chunks.
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
    and eventually falls back to fixed-size splitting with overlap.
    """

    def __init__(
        self,
        separators,
        chunk_size,
        chunk_overlap,
        length_function=len,
        keep_separator=False,
    ):
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
        return self._split_text(text)

    def _split_text(self, text, depth=0):
        if len(text) <= self.chunk_size:
            return [text]

        if depth == len(self.separators):
            return self._split_into_fixed_size(text)

        current_separator = self.separators[depth]
        if current_separator == "":
            return list(text)

        if self.keep_separator:
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

        if depth == 0:
            return self._merge_pieces(refined_pieces)
        return refined_pieces

    def _split_into_fixed_size(self, text):
        step = self.chunk_size - self.chunk_overlap
        if not text:
            return []
        chunks = [text[i : i + self.chunk_size] for i in range(0, len(text), step)]
        if len(chunks) > 1 and len(chunks[-1]) < self.chunk_overlap:
            chunks[-2] += chunks[-1]
            chunks.pop()
        return chunks

    def _merge_pieces(self, pieces):
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
                    current_chunk = current_chunk[-self.chunk_overlap :] + piece
                else:
                    current_chunk = piece
        merged.append(current_chunk)
        return merged
