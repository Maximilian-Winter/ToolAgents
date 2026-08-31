from __future__ import annotations

from typing import List, Optional

from ToolAgents.function_tool import FunctionTool
from ToolAgents.provider.llm_provider import (
    AsyncChatAPIProvider,
    ChatAPIProvider,
    ProviderSettings,
)

from .config import HarnessConfig
from .extensions import add_extension_catalog, create_extension_manager
from .prompt_composer import PromptComposer, create_prompt_composer
from .smart_messages import SmartMessageManager


def build_context_config(
    max_context_tokens: int,
    total_budget_tokens: Optional[int],
    context_kwargs: dict,
) -> dict:
    config = {
        "max_context_tokens": max_context_tokens,
        **context_kwargs,
    }
    if total_budget_tokens is not None:
        config["total_budget_tokens"] = total_budget_tokens
    return config


def build_harness_config(
    system_prompt: str,
    max_context_tokens: int,
    max_turns: int,
    streaming: bool,
    total_budget_tokens: Optional[int],
    context_kwargs: dict,
    extension_manager=None,
    prompt_composer: Optional[PromptComposer] = None,
) -> tuple[HarnessConfig, PromptComposer]:
    if prompt_composer is None:
        prompt_composer = create_prompt_composer(system_prompt)

    system_prompt = add_extension_catalog(prompt_composer, extension_manager)

    config = HarnessConfig(
        system_prompt=system_prompt,
        max_turns=max_turns,
        streaming=streaming,
        context_manager_config=build_context_config(
            max_context_tokens,
            total_budget_tokens,
            context_kwargs,
        ),
    )
    return config, prompt_composer


def register_initial_tools(
    harness,
    tools: Optional[List[FunctionTool]],
    extension_manager=None,
) -> None:
    if tools:
        harness.add_tools(tools)

    if extension_manager is not None:
        extension_tools = extension_manager.get_tools()
        if extension_tools:
            harness.add_tools(extension_tools)


def create_harness(
    provider: ChatAPIProvider,
    system_prompt: str = "You are a helpful assistant.",
    max_context_tokens: int = 128000,
    max_turns: int = -1,
    streaming: bool = False,
    total_budget_tokens: Optional[int] = None,
    settings: Optional[ProviderSettings] = None,
    tools: Optional[List[FunctionTool]] = None,
    log_output: bool = False,
    extension_manager=None,
    prompt_composer: Optional[PromptComposer] = None,
    smart_message_manager: Optional[SmartMessageManager] = None,
    **context_kwargs,
):
    from .harness import AgentHarness

    config, prompt_composer = build_harness_config(
        system_prompt=system_prompt,
        max_context_tokens=max_context_tokens,
        max_turns=max_turns,
        streaming=streaming,
        total_budget_tokens=total_budget_tokens,
        context_kwargs=context_kwargs,
        extension_manager=extension_manager,
        prompt_composer=prompt_composer,
    )

    harness = AgentHarness(
        provider=provider,
        config=config,
        settings=settings,
        log_output=log_output,
        extension_manager=extension_manager,
        prompt_composer=prompt_composer,
        smart_message_manager=smart_message_manager,
    )
    register_initial_tools(harness, tools, extension_manager)
    return harness


def create_async_harness(
    provider: AsyncChatAPIProvider,
    system_prompt: str = "You are a helpful assistant.",
    max_context_tokens: int = 128000,
    max_turns: int = -1,
    streaming: bool = False,
    total_budget_tokens: Optional[int] = None,
    settings: Optional[ProviderSettings] = None,
    tools: Optional[List[FunctionTool]] = None,
    log_output: bool = False,
    extension_manager=None,
    prompt_composer: Optional[PromptComposer] = None,
    smart_message_manager: Optional[SmartMessageManager] = None,
    **context_kwargs,
):
    from .async_harness import AsyncAgentHarness

    config, prompt_composer = build_harness_config(
        system_prompt=system_prompt,
        max_context_tokens=max_context_tokens,
        max_turns=max_turns,
        streaming=streaming,
        total_budget_tokens=total_budget_tokens,
        context_kwargs=context_kwargs,
        extension_manager=extension_manager,
        prompt_composer=prompt_composer,
    )

    harness = AsyncAgentHarness(
        provider=provider,
        config=config,
        settings=settings,
        log_output=log_output,
        extension_manager=extension_manager,
        prompt_composer=prompt_composer,
        smart_message_manager=smart_message_manager,
    )
    register_initial_tools(harness, tools, extension_manager)
    return harness


def create_harness_with_extensions(
    provider: ChatAPIProvider,
    system_prompt: str = "You are a helpful assistant.",
    skill_paths: Optional[List] = None,
    scan_defaults: bool = True,
    **kwargs,
):
    manager = create_extension_manager(
        skill_paths=skill_paths,
        scan_defaults=scan_defaults,
    )
    return create_harness(
        provider=provider,
        system_prompt=system_prompt,
        extension_manager=manager,
        **kwargs,
    )


def create_async_harness_with_extensions(
    provider: AsyncChatAPIProvider,
    system_prompt: str = "You are a helpful assistant.",
    skill_paths: Optional[List] = None,
    scan_defaults: bool = True,
    **kwargs,
):
    manager = create_extension_manager(
        skill_paths=skill_paths,
        scan_defaults=scan_defaults,
    )
    return create_async_harness(
        provider=provider,
        system_prompt=system_prompt,
        extension_manager=manager,
        **kwargs,
    )
