from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Union

logger = logging.getLogger(__name__)


@dataclass
class PromptModule:
    """A named section of the system prompt.

    Attributes:
        name: Unique identifier for this module (e.g. "instructions", "core_memory").
        position: Sort order when assembling the prompt. Lower values appear first.
            Modules with equal position are ordered by insertion time.
        content: Static string content. Mutually exclusive with content_fn at
            generation time — if both are set, content_fn takes priority.
        content_fn: A callable that returns the current content string. Called
            on every compile(). Use this for modules that reflect mutable state
            (e.g. core memory blocks, dynamic tool documentation).
        enabled: Whether this module is included in the compiled prompt.
        prefix: Optional text prepended before the content (e.g. a section header).
        suffix: Optional text appended after the content (e.g. a closing tag).
        separator: String used to join prefix + content + suffix. Defaults to newline.
    """

    name: str
    position: int = 0
    content: Optional[str] = None
    content_fn: Optional[Callable[[], str]] = None
    enabled: bool = True
    prefix: Optional[str] = None
    suffix: Optional[str] = None
    separator: str = "\n"

    # Internal: insertion order for stable sorting within the same position.
    _insertion_order: int = field(default=0, repr=False, compare=False)

    def get_content(self) -> str:
        """Resolve the current content of this module.

        If content_fn is set, it is called and its return value is used.
        Otherwise, the static content string is used.

        Returns:
            The resolved content string, or empty string if neither is set.
        """
        if self.content_fn is not None:
            try:
                return self.content_fn()
            except Exception as e:
                logger.error("Error in content_fn for module '%s': %s", self.name, e)
                # Fall back to static content if the callable fails.
                return self.content or ""
        return self.content or ""

    def render(self) -> str:
        """Render this module's full output including prefix and suffix.

        Returns:
            The assembled string, or empty string if the module has no content.
        """
        body = self.get_content()
        if not body and not self.prefix and not self.suffix:
            return ""

        parts = []
        if self.prefix:
            parts.append(self.prefix)
        if body:
            parts.append(body)
        if self.suffix:
            parts.append(self.suffix)

        return self.separator.join(parts)


class PromptComposer:
    """Assembles a system prompt from ordered, named modules.

    The composer maintains a registry of PromptModule instances. On each
    call to compile(), it sorts enabled modules by (position, insertion_order)
    and joins their rendered output into a single system prompt string.

    This replaces the monolithic system_prompt string in HarnessConfig,
    allowing runtime addition, removal, and modification of prompt sections.

    Usage:
        composer = PromptComposer()

        # Static module
        composer.add_module("instructions", position=0,
                           content="You are a helpful assistant.")

        # Dynamic module that reads state each turn
        composer.add_module("core_memory", position=10,
                           content_fn=lambda: core_memory_manager.build_context(),
                           prefix="<core_memory>", suffix="</core_memory>")

        # Compile on each turn
        system_prompt = composer.compile()

        # Runtime modifications
        composer.update_module("instructions", content="You are a coding assistant.")
        composer.disable_module("core_memory")
        composer.enable_module("core_memory")
        composer.remove_module("core_memory")
    """

    def __init__(self, module_separator: str = "\n\n") -> None:
        """Initialize the composer.

        Args:
            module_separator: String used to join modules in the final prompt.
                Defaults to double newline for clean visual separation.
        """
        self._modules: Dict[str, PromptModule] = {}
        self._insertion_counter: int = 0
        self.module_separator = module_separator

    # --- Module Management ---

    def add_module(
        self,
        name: str,
        position: int = 0,
        content: Optional[str] = None,
        content_fn: Optional[Callable[[], str]] = None,
        enabled: bool = True,
        prefix: Optional[str] = None,
        suffix: Optional[str] = None,
        separator: str = "\n",
    ) -> PromptModule:
        """Add a new module to the composer.

        Args:
            name: Unique identifier for the module.
            position: Sort order (lower = earlier in prompt).
            content: Static content string.
            content_fn: Callable returning dynamic content.
            enabled: Whether the module is active.
            prefix: Text prepended to the content.
            suffix: Text appended to the content.
            separator: Joiner for prefix + content + suffix.

        Returns:
            The created PromptModule.

        Raises:
            ValueError: If a module with this name already exists.
        """
        if name in self._modules:
            raise ValueError(
                f"Module '{name}' already exists. Use update_module() to modify it, "
                f"or remove_module() first."
            )

        module = PromptModule(
            name=name,
            position=position,
            content=content,
            content_fn=content_fn,
            enabled=enabled,
            prefix=prefix,
            suffix=suffix,
            separator=separator,
        )
        module._insertion_order = self._insertion_counter
        self._insertion_counter += 1
        self._modules[name] = module

        logger.debug("Added prompt module '%s' at position %d", name, position)
        return module

    def add_module_from_instance(self, module: PromptModule) -> PromptModule:
        """Add an existing PromptModule instance.

        Args:
            module: The module to add.

        Returns:
            The module.

        Raises:
            ValueError: If a module with this name already exists.
        """
        if module.name in self._modules:
            raise ValueError(f"Module '{module.name}' already exists.")

        module._insertion_order = self._insertion_counter
        self._insertion_counter += 1
        self._modules[module.name] = module
        return module

    def remove_module(self, name: str) -> Optional[PromptModule]:
        """Remove a module by name.

        Args:
            name: The module name to remove.

        Returns:
            The removed module, or None if not found.
        """
        module = self._modules.pop(name, None)
        if module:
            logger.debug("Removed prompt module '%s'", name)
        return module

    def update_module(
        self,
        name: str,
        content: Optional[str] = ...,
        content_fn: Optional[Callable[[], str]] = ...,
        position: Optional[int] = None,
        prefix: Optional[str] = ...,
        suffix: Optional[str] = ...,
        enabled: Optional[bool] = None,
    ) -> PromptModule:
        """Update an existing module's properties.

        Only the provided arguments are updated; others are left unchanged.
        Use explicit None to clear a field (e.g. content_fn=None to remove
        the dynamic callable and fall back to static content).

        The sentinel value ... (Ellipsis) means "don't change this field".

        Args:
            name: The module name to update.
            content: New static content (or ... to leave unchanged).
            content_fn: New dynamic callable (or ... to leave unchanged).
            position: New position (or None to leave unchanged).
            prefix: New prefix (or ... to leave unchanged).
            suffix: New suffix (or ... to leave unchanged).
            enabled: New enabled state (or None to leave unchanged).

        Returns:
            The updated module.

        Raises:
            KeyError: If the module doesn't exist.
        """
        if name not in self._modules:
            raise KeyError(f"Module '{name}' not found.")

        module = self._modules[name]

        if content is not ...:
            module.content = content
        if content_fn is not ...:
            module.content_fn = content_fn
        if position is not None:
            module.position = position
        if prefix is not ...:
            module.prefix = prefix
        if suffix is not ...:
            module.suffix = suffix
        if enabled is not None:
            module.enabled = enabled

        return module

    def enable_module(self, name: str) -> None:
        """Enable a module so it appears in the compiled prompt.

        Args:
            name: The module name.

        Raises:
            KeyError: If the module doesn't exist.
        """
        if name not in self._modules:
            raise KeyError(f"Module '{name}' not found.")
        self._modules[name].enabled = True

    def disable_module(self, name: str) -> None:
        """Disable a module so it is excluded from the compiled prompt.

        Args:
            name: The module name.

        Raises:
            KeyError: If the module doesn't exist.
        """
        if name not in self._modules:
            raise KeyError(f"Module '{name}' not found.")
        self._modules[name].enabled = False

    def has_module(self, name: str) -> bool:
        """Check if a module exists.

        Args:
            name: The module name.
        """
        return name in self._modules

    def get_module(self, name: str) -> Optional[PromptModule]:
        """Get a module by name.

        Args:
            name: The module name.

        Returns:
            The module, or None if not found.
        """
        return self._modules.get(name)

    # --- Compilation ---

    def compile(self) -> str:
        """Assemble all enabled modules into a single system prompt string.

        Modules are sorted by (position, insertion_order) and their rendered
        output is joined with module_separator. Empty renders are skipped.

        Returns:
            The compiled system prompt string.
        """
        sorted_modules = sorted(
            (m for m in self._modules.values() if m.enabled),
            key=lambda m: (m.position, m._insertion_order),
        )

        parts = []
        for module in sorted_modules:
            rendered = module.render()
            if rendered:
                parts.append(rendered)

        return self.module_separator.join(parts)

    # --- Introspection ---

    @property
    def modules(self) -> Dict[str, PromptModule]:
        """All registered modules (copy)."""
        return dict(self._modules)

    @property
    def enabled_modules(self) -> List[PromptModule]:
        """All currently enabled modules, sorted by position."""
        return sorted(
            (m for m in self._modules.values() if m.enabled),
            key=lambda m: (m.position, m._insertion_order),
        )

    @property
    def module_names(self) -> List[str]:
        """Names of all registered modules."""
        return list(self._modules.keys())

    def __len__(self) -> int:
        """Number of registered modules."""
        return len(self._modules)

    def __contains__(self, name: str) -> bool:
        """Check if a module name is registered."""
        return name in self._modules

    def __repr__(self) -> str:
        enabled = sum(1 for m in self._modules.values() if m.enabled)
        return f"PromptComposer({len(self._modules)} modules, {enabled} enabled)"


# --- Factory helpers ---

def create_prompt_composer(
    system_prompt: str = "You are a helpful assistant.",
    instructions_position: int = 0,
) -> PromptComposer:
    """Create a PromptComposer with a default instructions module.

    This is the simplest migration path from a monolithic system_prompt string
    to the modular composer. The existing prompt becomes the "instructions"
    module at the given position.

    Args:
        system_prompt: The initial system prompt content.
        instructions_position: Position for the instructions module.

    Returns:
        A configured PromptComposer.
    """
    composer = PromptComposer()
    composer.add_module(
        name="instructions",
        position=instructions_position,
        content=system_prompt,
    )
    return composer
