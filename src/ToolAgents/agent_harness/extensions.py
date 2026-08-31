from __future__ import annotations

from pathlib import Path
from typing import Callable, Iterable, Optional, TYPE_CHECKING

from ToolAgents.data_models.messages import ChatMessage, ToolCallResultContent
from ToolAgents.function_tool import FunctionTool

from .prompt_composer import PromptComposer
from .smart_messages import MessageLifecycle, SmartMessageManager

if TYPE_CHECKING:
    from ToolAgents.extensions.manager import ExtensionManager


def create_extension_manager(
    skill_paths: Optional[Iterable] = None,
    scan_defaults: bool = True,
) -> "ExtensionManager":
    """Create and discover the default skill-backed extension manager."""
    from ToolAgents.extensions import ExtensionManager, SkillFolderHandler, ExtensionScanPath

    manager = ExtensionManager()
    manager.register_handler(SkillFolderHandler())

    if scan_defaults:
        cwd = Path.cwd()
        home = Path.home()
        for subdir in [".agents/skills", ".claude/skills"]:
            project_path = cwd / subdir
            if project_path.is_dir():
                manager.add_scan_path(
                    ExtensionScanPath(path=project_path, scope="project", priority=10)
                )
        for subdir in [".agents/skills", ".claude/skills"]:
            user_path = home / subdir
            if user_path.is_dir():
                manager.add_scan_path(
                    ExtensionScanPath(path=user_path, scope="user", priority=0)
                )

    if skill_paths:
        for skill_path in skill_paths:
            manager.add_scan_path(
                ExtensionScanPath(path=Path(skill_path), scope="project", priority=10)
            )

    manager.discover()
    return manager


def add_extension_catalog(
    prompt_composer: PromptComposer,
    extension_manager: Optional["ExtensionManager"],
) -> str:
    """Add the extension catalog module and return the compiled system prompt."""
    if extension_manager is None:
        return prompt_composer.compile()

    catalog = extension_manager.build_catalog()
    if catalog:
        if prompt_composer.has_module("extension_catalog"):
            prompt_composer.update_module("extension_catalog", content=catalog)
        else:
            prompt_composer.add_module(
                name="extension_catalog",
                position=100,
                content=catalog,
            )

    return prompt_composer.compile()


def handle_slash_command(
    user_input: str,
    extension_manager: Optional["ExtensionManager"],
    smart_messages: SmartMessageManager,
    add_tools: Callable[[list[FunctionTool]], None],
) -> Optional[str]:
    """Activate slash-command skills, returning confirmation text when handled."""
    stripped = user_input.strip()
    if not stripped.startswith("/") or extension_manager is None:
        return None

    command = stripped[1:]
    result = extension_manager.try_handle_command(command)
    if result is None:
        return None

    message = ChatMessage.create_system_message(result.content)
    smart_messages.add_message(
        message,
        MessageLifecycle(pinned=result.pin_in_context),
    )
    if result.tools:
        add_tools(result.tools)

    return f"Skill '{command}' activated."


def handle_extension_tool_result(
    message: ChatMessage,
    extension_manager: Optional["ExtensionManager"],
    smart_messages: SmartMessageManager,
    add_tools: Callable[[list[FunctionTool]], None],
) -> None:
    """Apply pending extension activations returned through activate_skill."""
    if extension_manager is None:
        return

    for content in message.content:
        if not (
            isinstance(content, ToolCallResultContent)
            and content.tool_call_name == "activate_skill"
        ):
            continue

        pending = extension_manager._pending_activations
        for activation_name in list(pending.keys()):
            activation = pending[activation_name]
            if activation.content not in content.tool_call_result:
                continue

            if activation.pin_in_context:
                smart_messages.pin_message(message.id)
            if activation.tools:
                add_tools(activation.tools)
            del pending[activation_name]
            break
