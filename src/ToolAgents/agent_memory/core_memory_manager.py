import datetime
import json
from enum import Enum

from ToolAgents import FunctionTool


def create_enum(enum_name, enum_values):
    return Enum(enum_name, {value: value for value in enum_values})


class CoreMemoryManager:
    def __init__(self, core_memory_keys: list[str], core_memory: dict):
        self.core_memory = core_memory
        self.last_modified = "Never"
        CoreMemoryKey = create_enum("CoreMemoryKey", core_memory_keys)

        def append_core_memory(key: CoreMemoryKey, content: str):
            """
            Appends content to a key section.
            Args:
                key (CoreMemoryKey): The key section to append the content to.
                content (str): The content to append to the key section.
            """
            if key in self.core_memory:
                self.core_memory[key] += content
                self.last_modified = datetime.datetime.now().strftime("%d/%m/%Y, %H:%M:%S")
                return f"Content appended successfully to key: {key}"
            return f"Key not found in Core memory. Key: {key}."

        def replace_in_core_memory(key: CoreMemoryKey, old_content: str, new_content: str) -> str:
            """
            Replaces existing content in a key section with new content.

            Args:
                key (CoreMemoryKey): The key section in which the content gets replace.
                old_content (str): The old content to replace.
                new_content (str): The new content to replace with.
            """

            if key in self.core_memory:
                if old_content in self.core_memory[key]:
                    self.core_memory[key].replace(old_content, new_content)
                    self.last_modified = datetime.datetime.now().strftime("%d/%m/%Y, %H:%M:%S")
                    return f"Old content replaced. Key: {key}."
                else:
                    return f"Old content is not present in key: {key}."

            else:
                return f"Key not found in Core memory. Key: {key}."

        self.tools = [FunctionTool(append_core_memory), FunctionTool(replace_in_core_memory)]

    def build_core_memory_context(self):
        context = ""
        for key, item in self.core_memory.items():
            context += f"""<{key}>\n"""
            context += f"""  {self.format_multiline_description(item, 2)}\n"""
            context += f"</{key}>\n"

        return context

    def format_multiline_description(self, description: str, indent_level: int) -> str:
        """
        Format a multiline description with proper indentation.

        Args:
            description (str): Multiline description.
            indent_level (int): Indentation level.

        Returns:
            str: Formatted multiline description.
        """
        indent = " " * indent_level
        return description.replace("\n", "\n" + indent)

    def load(self, file_path):
        with open(file_path, "r", encoding="utf-8") as file:
            json_data = json.load(file)
        self.core_memory = json_data["core_memory"]
        self.last_modified = json_data["last_modified"]

    def save(self, filepath):
        json_data = {"core_memory": self.core_memory, "last_modified": self.last_modified}
        with open(filepath, "w", encoding="utf-8") as file:
            json.dump(json_data, file, indent=4)

    def get_tools(self):
        return self.tools
