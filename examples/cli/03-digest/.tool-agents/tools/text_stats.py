"""Tools the digest workflow can call.

A workspace tool module may define TOOLS, a create_tools() function, or simply
bind FunctionTool objects at module level. The plugin name is the file stem, so
these are referenced as {"plugin": "text_stats", "tool_name": "CountWords"}.
"""

from pydantic import BaseModel, Field

from ToolAgents import FunctionTool


class CountWords(BaseModel):
    """Count the words in a passage of text."""

    text: str = Field(..., description="The text to measure.")

    def run(self) -> int:
        return len(self.text.split())


class ReadingMinutes(BaseModel):
    """Estimate reading time in minutes for a passage of text."""

    text: str = Field(..., description="The text to measure.")
    words_per_minute: int = Field(
        238, description="Average adult silent reading speed."
    )

    def run(self) -> float:
        return round(len(self.text.split()) / max(self.words_per_minute, 1), 2)


TOOLS = [FunctionTool(CountWords), FunctionTool(ReadingMinutes)]
