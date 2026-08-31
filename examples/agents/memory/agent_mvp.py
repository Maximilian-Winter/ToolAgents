"""
agent_mvp.py — Model-View-Presenter for Agent Context Management

The agent's context window is a VIEW composed of independently
rendered ELEMENTS. Each element pulls from a MODEL (any data source)
and is managed by a PRESENTER that decides what's visible and when
things refresh.

Architecture:
  
  ┌─────────────────────────────────────────────┐
  │              DATA SOURCES (Model)            │
  │                                              │
  │  Database  Files  APIs  Memory  Sub-Agents   │
  └──────┬───────┬──────┬──────┬───────┬────────┘
         │       │      │      │       │
         ▼       ▼      ▼      ▼       ▼
  ┌─────────────────────────────────────────────┐
  │         VIEW ELEMENTS (View Layer)           │
  │                                              │
  │  Each element:                               │
  │    - Has a source (fn, agent, static, query) │
  │    - Renders to text                         │
  │    - Has lifecycle (when to refresh)          │
  │    - Has priority (what gets cut first)       │
  │    - Has token budget                        │
  └──────────────────┬──────────────────────────┘
                     │
                     ▼
  ┌─────────────────────────────────────────────┐
  │         CONTEXT PRESENTER                    │
  │                                              │
  │  Orchestrates:                               │
  │    - Which elements are active               │
  │    - When to refresh each element            │
  │    - Token budget allocation                 │
  │    - Render order and formatting             │
  │    - Event-driven reloads                    │
  └──────────────────┬──────────────────────────┘
                     │
                     ▼
  ┌─────────────────────────────────────────────┐
  │         AGENT CONTEXT (Final Output)         │
  │                                              │
  │  System prompt text composed from all        │
  │  active, rendered elements                   │
  └─────────────────────────────────────────────┘
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import (
    Any, Callable, Optional, Protocol, 
    runtime_checkable
)


# ═══════════════════════════════════════════════════════════════════
# VIEW ELEMENT — The atomic unit of context
# ═══════════════════════════════════════════════════════════════════


class RefreshPolicy(Enum):
    """When should this element's content be regenerated?"""
    STATIC = auto()       # Never refreshes after initial render
    EVERY_TURN = auto()   # Regenerates every turn
    ON_EVENT = auto()     # Regenerates when a specific event fires
    ON_STALE = auto()     # Regenerates after N turns without refresh
    MANUAL = auto()       # Only refreshes when explicitly told to


class Priority(Enum):
    """If context budget is tight, what gets cut first?"""
    CRITICAL = 0    # Never cut — core instructions, safety
    HIGH = 1        # Cut only under extreme pressure
    STANDARD = 2    # Normal priority — main content
    LOW = 3         # First to be cut or summarized
    OPTIONAL = 4    # Nice to have, easily dropped


@dataclass
class ElementMetadata:
    """Tracks the lifecycle state of a view element."""
    last_rendered: float = 0.0
    render_count: int = 0
    last_token_count: int = 0
    turns_since_refresh: int = 0
    is_stale: bool = False
    
    def mark_rendered(self, token_count: int):
        self.last_rendered = time.time()
        self.render_count += 1
        self.last_token_count = token_count
        self.turns_since_refresh = 0
        self.is_stale = False
    
    def tick(self):
        self.turns_since_refresh += 1


@runtime_checkable
class ElementSource(Protocol):
    """
    Anything that can produce text content for a view element.
    
    This is the bridge between Model and View.
    Can be implemented by:
      - A simple function
      - A database query wrapper
      - A sub-agent
      - A file reader
      - An API client
      - A computed/derived value
    """
    def render(self, context: dict[str, Any] | None = None) -> str:
        """Produce the text content for this element."""
        ...


class FunctionSource:
    """Wraps a callable as an ElementSource."""
    
    def __init__(self, fn: Callable[..., str]):
        self.fn = fn
    
    def render(self, context: dict[str, Any] | None = None) -> str:
        if context:
            return self.fn(**context)
        return self.fn()


class StaticSource:
    """A fixed string that never changes."""
    
    def __init__(self, content: str):
        self.content = content
    
    def render(self, context: dict[str, Any] | None = None) -> str:
        return self.content


class QuerySource:
    """Fetches content from a data store via a query function."""
    
    def __init__(
        self, 
        query_fn: Callable[[str], str],
        query: str
    ):
        self.query_fn = query_fn
        self.query = query
    
    def render(self, context: dict[str, Any] | None = None) -> str:
        query = self.query
        # Allow context to override the query dynamically
        if context and "query" in context:
            query = context["query"]
        return self.query_fn(query)


class AgentSource:
    """
    Uses a sub-agent to generate content.
    
    The sub-agent receives a prompt and returns structured text.
    This is where cheap models do structured work.
    """
    
    def __init__(
        self,
        agent_fn: Callable[[str, dict], str],
        prompt_template: str
    ):
        self.agent_fn = agent_fn
        self.prompt_template = prompt_template
    
    def render(self, context: dict[str, Any] | None = None) -> str:
        prompt = self.prompt_template
        if context:
            prompt = self.prompt_template.format(**context)
        return self.agent_fn(prompt, context or {})


class CompositeSource:
    """
    Combines multiple sources into one element.
    Useful when one element needs data from several places.
    """
    
    def __init__(
        self,
        sources: list[tuple[str, ElementSource]],
        template: str = "{content}",
        separator: str = "\n"
    ):
        self.sources = sources
        self.template = template
        self.separator = separator
    
    def render(self, context: dict[str, Any] | None = None) -> str:
        parts = {}
        for name, source in self.sources:
            parts[name] = source.render(context)
        
        if "{" in self.template:
            return self.template.format(**parts)
        
        return self.separator.join(parts.values())


@dataclass
class ViewElement:
    """
    One piece of the agent's context.
    
    The atomic unit of the View layer. Each element:
      - Has a name (for reference and debugging)
      - Has a source that produces text
      - Has a position (rendering order)
      - Has a refresh policy (when to regenerate)
      - Has a priority (what to cut under pressure)
      - Has optional formatting (prefix/suffix)
      - Tracks its own lifecycle metadata
    """
    name: str
    source: ElementSource
    position: int = 0
    refresh: RefreshPolicy = RefreshPolicy.EVERY_TURN
    priority: Priority = Priority.STANDARD
    enabled: bool = True
    
    # Formatting
    prefix: str = ""
    suffix: str = ""
    
    # Budget
    max_tokens: int | None = None  # None = no limit
    
    # Staleness
    stale_after: int = 5  # turns before considered stale
    
    # Event triggers
    refresh_events: set[str] = field(default_factory=set)
    
    # Lifecycle tracking
    metadata: ElementMetadata = field(default_factory=ElementMetadata)
    
    # Cached content
    _cached_content: str = ""
    
    def should_refresh(self, fired_events: set[str] | None = None) -> bool:
        """Determine if this element needs to re-render."""
        if not self.enabled:
            return False
        
        if self.refresh == RefreshPolicy.STATIC:
            return self.metadata.render_count == 0
        
        if self.refresh == RefreshPolicy.EVERY_TURN:
            return True
        
        if self.refresh == RefreshPolicy.ON_EVENT:
            if fired_events and self.refresh_events & fired_events:
                return True
            return False
        
        if self.refresh == RefreshPolicy.ON_STALE:
            return self.metadata.turns_since_refresh >= self.stale_after
        
        if self.refresh == RefreshPolicy.MANUAL:
            return False
        
        return True
    
    def render(
        self, 
        context: dict[str, Any] | None = None,
        force: bool = False,
        fired_events: set[str] | None = None
    ) -> str:
        """Render this element, using cache if appropriate."""
        if not self.enabled:
            return ""
        
        if force or self.should_refresh(fired_events):
            self._cached_content = self.source.render(context)
            
            # Estimate tokens (rough: 1 token ≈ 4 chars)
            estimated_tokens = len(self._cached_content) // 4
            self.metadata.mark_rendered(estimated_tokens)
        
        if not self._cached_content:
            return ""
        
        parts = []
        if self.prefix:
            parts.append(self.prefix)
        parts.append(self._cached_content)
        if self.suffix:
            parts.append(self.suffix)
        
        return "\n".join(parts)
    
    def force_refresh(self, context: dict[str, Any] | None = None):
        """Force a re-render regardless of policy."""
        self.render(context, force=True)
    
    def tick(self):
        """Advance the lifecycle by one turn."""
        self.metadata.tick()


# ═══════════════════════════════════════════════════════════════════
# CONTEXT PRESENTER — Orchestrates the View
# ═══════════════════════════════════════════════════════════════════


@dataclass 
class ContextBudget:
    """Tracks token budget allocation."""
    total_budget: int
    reserved: int = 0    # tokens reserved for critical elements
    allocated: int = 0   # tokens currently allocated
    
    @property
    def available(self) -> int:
        return self.total_budget - self.reserved - self.allocated
    
    def allocate(self, tokens: int) -> bool:
        if tokens <= self.available:
            self.allocated += tokens
            return True
        return False
    
    def reset(self):
        self.allocated = 0


class ContextPresenter:
    """
    The Presenter in the MVP pattern.
    
    Orchestrates which ViewElements are active, when they refresh,
    how they're ordered, and manages the token budget.
    
    This replaces PromptComposer with a more structured approach
    while remaining compatible with it.
    """
    
    def __init__(
        self,
        token_budget: int = 8000,
        separator: str = "\n\n"
    ):
        self.elements: dict[str, ViewElement] = {}
        self.budget = ContextBudget(total_budget=token_budget)
        self.separator = separator
        self._event_queue: set[str] = set()
        self._global_context: dict[str, Any] = {}
    
    # ── Element Management ──
    
    def add_element(self, element: ViewElement) -> ContextPresenter:
        """Add a view element. Returns self for chaining."""
        self.elements[element.name] = element
        return self
    
    def remove_element(self, name: str):
        """Remove a view element by name."""
        self.elements.pop(name, None)
    
    def enable(self, name: str):
        if name in self.elements:
            self.elements[name].enabled = True
    
    def disable(self, name: str):
        if name in self.elements:
            self.elements[name].enabled = False
    
    def get_element(self, name: str) -> ViewElement | None:
        return self.elements.get(name)
    
    # ── Context Management ──
    
    def set_context(self, key: str, value: Any):
        """Set a global context value available to all elements."""
        self._global_context[key] = value
    
    def update_context(self, context: dict[str, Any]):
        """Update multiple global context values."""
        self._global_context.update(context)
    
    # ── Events ──
    
    def fire_event(self, event: str):
        """Queue an event that may trigger element refreshes."""
        self._event_queue.add(event)
    
    def fire_events(self, events: set[str]):
        """Queue multiple events."""
        self._event_queue |= events
    
    # ── Core Rendering ──
    
    def render(
        self,
        extra_context: dict[str, Any] | None = None
    ) -> str:
        """
        Render all active elements into a single context string.
        
        This is the main output — what goes into the agent's
        system prompt or context injection point.
        """
        # Merge contexts
        context = {**self._global_context}
        if extra_context:
            context.update(extra_context)
        
        # Get active elements sorted by position
        active = sorted(
            [e for e in self.elements.values() if e.enabled],
            key=lambda e: e.position
        )
        
        # Reset budget
        self.budget.reset()
        
        # Render in priority order for budget allocation,
        # but output in position order
        rendered: dict[str, str] = {}
        
        # First pass: render critical and high priority
        for element in sorted(active, key=lambda e: e.priority.value):
            content = element.render(
                context=context,
                fired_events=self._event_queue
            )
            
            if not content:
                continue
            
            estimated_tokens = len(content) // 4
            
            # Check budget
            if element.max_tokens and estimated_tokens > element.max_tokens:
                # Truncate to budget
                char_limit = element.max_tokens * 4
                content = content[:char_limit] + "\n[...truncated]"
                estimated_tokens = element.max_tokens
            
            if self.budget.allocate(estimated_tokens):
                rendered[element.name] = content
            elif element.priority.value <= Priority.HIGH.value:
                # Critical and high priority always get through
                rendered[element.name] = content
        
        # Build output in position order
        output_parts = []
        for element in active:
            if element.name in rendered:
                output_parts.append(rendered[element.name])
        
        # Tick all elements
        for element in self.elements.values():
            element.tick()
        
        # Clear event queue
        self._event_queue.clear()
        
        return self.separator.join(output_parts)
    
    # ── Convenience Builders ──
    
    def add_static(
        self, 
        name: str, 
        content: str,
        position: int = 0,
        priority: Priority = Priority.CRITICAL,
        prefix: str = "",
        suffix: str = ""
    ) -> ContextPresenter:
        """Add a static text element (never refreshes)."""
        return self.add_element(ViewElement(
            name=name,
            source=StaticSource(content),
            position=position,
            refresh=RefreshPolicy.STATIC,
            priority=priority,
            prefix=prefix,
            suffix=suffix
        ))
    
    def add_dynamic(
        self,
        name: str,
        fn: Callable[..., str],
        position: int = 0,
        refresh: RefreshPolicy = RefreshPolicy.EVERY_TURN,
        priority: Priority = Priority.STANDARD,
        prefix: str = "",
        suffix: str = "",
        refresh_events: set[str] | None = None,
        stale_after: int = 5,
        max_tokens: int | None = None
    ) -> ContextPresenter:
        """Add a dynamic function-based element."""
        return self.add_element(ViewElement(
            name=name,
            source=FunctionSource(fn),
            position=position,
            refresh=refresh,
            priority=priority,
            prefix=prefix,
            suffix=suffix,
            refresh_events=refresh_events or set(),
            stale_after=stale_after,
            max_tokens=max_tokens
        ))
    
    def add_query(
        self,
        name: str,
        query_fn: Callable[[str], str],
        query: str,
        position: int = 0,
        refresh: RefreshPolicy = RefreshPolicy.ON_EVENT,
        priority: Priority = Priority.STANDARD,
        prefix: str = "",
        suffix: str = "",
        refresh_events: set[str] | None = None,
        max_tokens: int | None = None
    ) -> ContextPresenter:
        """Add a query-based element."""
        return self.add_element(ViewElement(
            name=name,
            source=QuerySource(query_fn, query),
            position=position,
            refresh=refresh,
            priority=priority,
            prefix=prefix,
            suffix=suffix,
            refresh_events=refresh_events or set(),
            max_tokens=max_tokens
        ))
    
    def add_agent(
        self,
        name: str,
        agent_fn: Callable[[str, dict], str],
        prompt_template: str,
        position: int = 0,
        refresh: RefreshPolicy = RefreshPolicy.ON_EVENT,
        priority: Priority = Priority.STANDARD,
        prefix: str = "",
        suffix: str = "",
        refresh_events: set[str] | None = None,
        max_tokens: int | None = None
    ) -> ContextPresenter:
        """Add a sub-agent-based element."""
        return self.add_element(ViewElement(
            name=name,
            source=AgentSource(agent_fn, prompt_template),
            position=position,
            refresh=refresh,
            priority=priority,
            prefix=prefix,
            suffix=suffix,
            refresh_events=refresh_events or set(),
            max_tokens=max_tokens
        ))


# ═══════════════════════════════════════════════════════════════════
# EXAMPLES — How different domains use the same pattern
# ═══════════════════════════════════════════════════════════════════


def example_game_master():
    """Game Master context using the MVP pattern."""
    
    # === MODEL LAYER (simulated) ===
    
    # These would be real database calls in production
    def get_location_data() -> str:
        return (
            '{"name": "Wudang Monastery", "type": "monastery",\n'
            ' "description": "A remote Taoist monastery on a cliff...",\n'
            ' "npcs": ["Master Chen", "Brother Fang", "Sister Yue"],\n'
            ' "exits": ["mountain_path", "meditation_platform"]}'
        )
    
    def get_player_state() -> str:
        return (
            '{"name": "Li Wei", "hp": "100/100", "gold": "15g 40c",\n'
            ' "inventory": ["jian", "scrolls", "medicinal herbs"],\n'
            ' "active_quests": ["Decode the Taixuan Jing prophecy"]}'
        )
    
    def get_present_npcs() -> str:
        return (
            '{"npcs": [\n'
            '  {"name": "Master Chen", "disposition": "cryptic",\n'
            '   "secret": "Guards the complete Taixuan Jing"},\n'
            '  {"name": "Brother Fang", "disposition": "gruff",\n'
            '   "secret": "Former imperial soldier"}\n'
            ']}'
        )
    
    story_summary = (
        "Li Wei and companions arrived at Wudang seeking the "
        "Taixuan Jing prophecy. Master Chen has revealed only "
        "fragments. Bandit activity increasing on the mountain path."
    )
    
    gm_notes = "Brother Fang knows more about the bandits than he reveals."
    
    # === PRESENTER LAYER ===
    
    presenter = ContextPresenter(token_budget=4000)
    
    # Static: GM instructions (never change)
    presenter.add_static(
        "gm_instructions",
        content=(
            "You are an expert Game Master running a Wuxia RPG.\n"
            "Narrate vividly. NPCs have distinct voices.\n"
            "Always end with a prompt for player action.\n"
            "Use signal_event tool for significant occurrences.\n"
            "Use change_location tool when the player travels."
        ),
        position=0,
        priority=Priority.CRITICAL
    )
    
    # Dynamic: Location data (refreshes on location_change event)
    presenter.add_dynamic(
        "location",
        fn=get_location_data,
        position=10,
        refresh=RefreshPolicy.ON_EVENT,
        refresh_events={"location_change"},
        priority=Priority.HIGH,
        prefix="## Current Location",
        suffix="## End Location"
    )
    
    # Dynamic: NPCs present (refreshes on location_change or npc_event)
    presenter.add_dynamic(
        "present_npcs",
        fn=get_present_npcs,
        position=15,
        refresh=RefreshPolicy.ON_EVENT,
        refresh_events={"location_change", "npc_event"},
        priority=Priority.HIGH,
        prefix="## NPCs Present",
        suffix="## End NPCs"
    )
    
    # Dynamic: Player state (refreshes every turn — it's cheap)
    presenter.add_dynamic(
        "player_state",
        fn=get_player_state,
        position=20,
        refresh=RefreshPolicy.EVERY_TURN,
        priority=Priority.HIGH,
        prefix="## Player State",
        suffix="## End Player State"
    )
    
    # Static-ish: Story summary (refreshes on story_checkpoint)
    presenter.add_static(
        "story_summary",
        content=story_summary,
        position=5,
        priority=Priority.STANDARD,
        prefix="## Story So Far",
        suffix="## End Story"
    )
    
    # Low priority: GM notes (nice to have, can be cut)
    presenter.add_static(
        "gm_notes",
        content=gm_notes,
        position=25,
        priority=Priority.LOW,
        prefix="## GM Notes",
        suffix="## End GM Notes"
    )
    
    # === RENDER ===
    
    # Initial render
    print("=== INITIAL CONTEXT ===")
    print(presenter.render())
    print(f"\nBudget: {presenter.budget.allocated}/{presenter.budget.total_budget} tokens")
    
    # Simulate location change
    print("\n=== AFTER LOCATION CHANGE EVENT ===")
    presenter.fire_event("location_change")
    print(presenter.render())
    
    # Show element status
    print("\n=== ELEMENT STATUS ===")
    for name, elem in presenter.elements.items():
        print(
            f"  {name}: renders={elem.metadata.render_count}, "
            f"stale_turns={elem.metadata.turns_since_refresh}, "
            f"tokens≈{elem.metadata.last_token_count}"
        )


def example_personal_assistant():
    """Personal assistant context using the same MVP pattern."""
    
    def get_user_profile() -> str:
        return (
            '{"name": "Max", "preferences": "likes Cyberpunk Red, '
            'codes in Python/C++", "timezone": "CET"}'
        )
    
    def get_active_reminders() -> str:
        return '{"reminders": ["Dentist Thursday 2pm", "Deploy v2.1 Friday"]}'
    
    def get_current_project() -> str:
        return (
            '{"project": "ToolAgents", "status": "adding MVP context system",\n'
            ' "recent_files": ["agent_mvp.py", "context_presenter.py"],\n'
            ' "blockers": "none"}'
        )
    
    presenter = ContextPresenter(token_budget=4000)
    
    presenter.add_static(
        "instructions",
        content="You are Ada, a personal AI companion with persistent memory.",
        position=0,
        priority=Priority.CRITICAL
    )
    
    presenter.add_dynamic(
        "user_profile",
        fn=get_user_profile,
        position=5,
        refresh=RefreshPolicy.ON_STALE,
        stale_after=10,  # refresh every 10 turns
        priority=Priority.HIGH,
        prefix="## User Profile"
    )
    
    presenter.add_dynamic(
        "reminders",
        fn=get_active_reminders,
        position=10,
        refresh=RefreshPolicy.EVERY_TURN,  # always fresh
        priority=Priority.HIGH,
        prefix="## Active Reminders"
    )
    
    presenter.add_dynamic(
        "current_project",
        fn=get_current_project,
        position=15,
        refresh=RefreshPolicy.ON_EVENT,
        refresh_events={"project_switch", "file_change"},
        priority=Priority.STANDARD,
        prefix="## Current Project Context"
    )
    
    print("=== ASSISTANT CONTEXT ===")
    print(presenter.render())


if __name__ == "__main__":
    print("=" * 64)
    print("  Game Master Example")
    print("=" * 64)
    example_game_master()
    
    print("\n" + "=" * 64)
    print("  Personal Assistant Example")
    print("=" * 64)
    example_personal_assistant()