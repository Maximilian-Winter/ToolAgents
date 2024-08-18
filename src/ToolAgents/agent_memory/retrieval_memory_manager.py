import json

from .retrieval_memory import RetrievalMemory
from .. import FunctionTool


class RetrievalMemoryManager:
    def __init__(self, retrieval_memory: RetrievalMemory):
        self.retrieval_memory = retrieval_memory

        def archival_memory_search(query: str, page: int = 0):
            """
            Search archival memory using semantic (embedding-based) search.
            Args:
                query (str): Search query.
                page (int, optional): Page number. Defaults to 0.
            """
            return self.retrieve_memories(query, page=page)

        def archival_memory_insert(memory: str, importance: float):
            """
            Add to archival memory.
            Args:
                memory (str): The archival memory to insert.
                importance (float): The importance of the archival memory.
            """
            return self.add_memory_to_retrieval(
                memory, importance
            )
        self.tools = [FunctionTool(archival_memory_insert), FunctionTool(archival_memory_search)]

    def get_tools(self):
        return self.tools

    def add_memory_to_retrieval(self, description: str, importance: float = 1.0) -> str:
        """
        Adds a memory with a given description and importance to the memory stream.
        """
        self.retrieval_memory.add_memory(description, importance=importance)
        return f"Information added to archival memory."

    def retrieve_memories(
        self, query: str, max_results: int = 5, page: int = 1, page_size: int = 5
    ) -> str:
        """
        Retrieves memories from the memory stream based on a query.
        """
        memories = self.retrieval_memory.retrieve_memories(query, max_results)
        # Calculate start and end indices for slicing the memories list for pagination
        start_index = (page - 1) * page_size
        end_index = start_index + page_size

        # Slice the list to get the paginated results
        paginated_memories = memories[start_index:end_index]
        formatted_memories = ""
        for memory in paginated_memories:
            formatted_memories += (
                f'{memory["creation_timestamp"]}: {memory["memory"]}\n'
            )

        if formatted_memories != "":
            formatted_memories += f"\n\nPage {page} of {len(memories) // page_size + 1}"
        return (
            formatted_memories
            if formatted_memories
            else "No archival memories found matching the query."
        )
