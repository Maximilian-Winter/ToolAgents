import dataclasses
import typing
import uuid
from typing import Optional

from ToolAgents.knowledge.text_processing.text_splitter import TextSplitter


@dataclasses.dataclass
class DocumentChunk:
    id: str
    parent_doc_id: str
    content: str
    chunk_index: int
    size_in_characters: int

@dataclasses.dataclass
class Document:
    id: str
    document_chunks: list[DocumentChunk] = dataclasses.field(default_factory=list[DocumentChunk])
    metadata: Optional[typing.Dict[str, typing.Any]] = None

class DocumentGenerator:
    def __init__(self, text_splitter: TextSplitter = None):
        self.text_splitter = text_splitter

    def generate_document(self, text: str, metadata: dict = None) -> Document:
        chunk_index = 0
        if self.text_splitter is not None:
            chunks = self.text_splitter.get_chunks(text)
        else:
            chunks = [text]

        document = Document(id=str(uuid.uuid4()), metadata=metadata)
        for chunk in chunks:
            document_chunk = DocumentChunk(id=str(uuid.uuid4()), parent_doc_id=document.id, content=chunk, chunk_index=chunk_index, size_in_characters=len(chunk))
            document.document_chunks.append(document_chunk)
            chunk_index = chunk_index + 1
        return document