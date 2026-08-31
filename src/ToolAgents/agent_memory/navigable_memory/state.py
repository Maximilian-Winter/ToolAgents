"""Navigation state for NavigableMemory."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

@dataclass
class LocationState:
    """Tracks the agent's current position and movement history."""
    current_path: Optional[str] = None
    current_title: str = "None"
    current_content: str = ""
    history: List[str] = field(default_factory=list)
    max_history: int = 50

    def move_to(self, path: str, title: str, content: str) -> Optional[str]:
        """Move to a new location. Returns the old path or None."""
        old_path = self.current_path
        self.current_path = path
        self.current_title = title
        self.current_content = content
        self.history.append(path)
        if len(self.history) > self.max_history:
            self.history = self.history[-self.max_history:]
        return old_path

    @property
    def has_location(self) -> bool:
        return self.current_path is not None

    @property
    def visit_count(self) -> int:
        if not self.current_path:
            return 0
        return sum(1 for p in self.history if p == self.current_path)
