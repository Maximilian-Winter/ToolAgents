"""Declarative provider and agent configuration for pipelines.

A pipeline JSON document can declare the endpoints it runs against::

    "agents": [
      {
        "name": "writer",
        "provider": {
          "type": "openai",
          "model": "qwen/qwen3.5-9b",
          "base_url": "https://openrouter.ai/api/v1",
          "api_key_env": "OPENROUTER_API_KEY",
          "settings": {"temperature": 0.3, "top_p": 0.9}
        }
      },
      {
        "name": "judge",
        "provider": {"type": "anthropic", "model": "claude-sonnet-4-20250514"}
      }
    ],
    "default_agent": "writer"

Secrets are never serialized. A config names the *environment variable* that
holds the key; the value is read at build time and, if absent, the failure
names the variable so it is obvious what to set.

Provider SDKs are imported lazily, so a pipeline that only uses OpenAI does
not require ``anthropic``, ``groq`` and ``mistralai`` to be installed.
"""

from __future__ import annotations

import importlib
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

__all__ = [
    "AgentConfig",
    "load_env_file",
    "PROVIDER_SPECS",
    "ProviderConfig",
    "ProviderSpec",
    "AgentConfigurationError",
    "build_agents_from_configs",
    "register_provider_spec",
]


class AgentConfigurationError(ValueError):
    """Raised when an agent or provider cannot be built from configuration."""


def load_env_file(path: str | "os.PathLike[str]", required: bool = True) -> bool:
    """Load environment variables from a ``.env`` file.

    Values already present in the environment are left alone, so an exported
    variable always beats a file — the file is a default, not an override.

    Args:
        path: The ``.env`` file to read.
        required: When true, a missing file is an error. Pass false for a
            conventional location that may simply not exist.

    Returns:
        bool: Whether a file was actually read.
    """

    resolved = Path(path).expanduser()
    if not resolved.is_file():
        if required:
            raise AgentConfigurationError(f"Env file does not exist: {resolved}")
        return False

    try:
        from dotenv import load_dotenv
    except ImportError as exc:
        raise AgentConfigurationError(
            f"Reading {resolved} needs the 'python-dotenv' package, which is "
            "not installed. Install it, or set the variables in the "
            "environment directly."
        ) from exc

    load_dotenv(resolved, override=False)
    return True


@dataclass(frozen=True)
class ProviderSpec:
    """How to construct one kind of chat provider from configuration."""

    #: Value used in the JSON ``type`` field.
    name: str
    #: Module holding the provider class, imported on demand.
    module: str
    #: Provider class name within ``module``.
    class_name: str
    #: Environment variable consulted when the config names none.
    default_api_key_env: str
    #: Whether this provider accepts a ``base_url`` argument.
    supports_base_url: bool = True
    #: Whether this provider accepts a ``provider_identifier`` argument.
    supports_provider_identifier: bool = False

    def load_class(self) -> type:
        """Import and return the provider class."""

        try:
            module = importlib.import_module(self.module)
        except ImportError as exc:
            raise AgentConfigurationError(
                f"Provider '{self.name}' requires the '{exc.name}' package, "
                f"which is not installed."
            ) from exc
        return getattr(module, self.class_name)


PROVIDER_SPECS: dict[str, ProviderSpec] = {}


def register_provider_spec(spec: ProviderSpec) -> ProviderSpec:
    """Register a provider kind so pipeline JSON can name it."""

    PROVIDER_SPECS[spec.name] = spec
    return spec


_CHAT_API_MODULE = "ToolAgents.provider.chat_api_provider"

for _spec in (
    ProviderSpec(
        name="openai",
        module=f"{_CHAT_API_MODULE}.open_ai",
        class_name="OpenAIChatAPI",
        default_api_key_env="OPENAI_API_KEY",
        supports_base_url=True,
        supports_provider_identifier=True,
    ),
    ProviderSpec(
        name="anthropic",
        module=f"{_CHAT_API_MODULE}.anthropic",
        class_name="AnthropicChatAPI",
        default_api_key_env="ANTHROPIC_API_KEY",
        supports_base_url=True,
    ),
    ProviderSpec(
        name="groq",
        module=f"{_CHAT_API_MODULE}.groq",
        class_name="GroqChatAPI",
        default_api_key_env="GROQ_API_KEY",
        supports_base_url=True,
    ),
    ProviderSpec(
        name="mistral",
        module=f"{_CHAT_API_MODULE}.mistral",
        class_name="MistralChatAPI",
        default_api_key_env="MISTRAL_API_KEY",
        supports_base_url=True,
    ),
):
    register_provider_spec(_spec)

#: Convenience aliases for OpenAI-compatible gateways. They are ordinary
#: OpenAI providers with a different default base URL.
#: ``alias -> (default base URL, key env var, key is optional)``. The last flag
#: covers local servers, which accept any key or none: requiring one would make
#: the aliases whose whole purpose is local serving unusable out of the box.
OPENAI_COMPATIBLE_DEFAULTS: dict[str, tuple[str, str, bool]] = {
    "openai_compatible": ("", "OPENAI_API_KEY", False),
    "openrouter": ("https://openrouter.ai/api/v1", "OPENROUTER_API_KEY", False),
    "together": ("https://api.together.xyz/v1", "TOGETHER_API_KEY", False),
    "deepseek": ("https://api.deepseek.com/v1", "DEEPSEEK_API_KEY", False),
    "ollama": ("http://localhost:11434/v1", "OLLAMA_API_KEY", True),
    "vllm": ("http://localhost:8000/v1", "VLLM_API_KEY", True),
}

#: Sent when a local endpoint needs no real credential.
PLACEHOLDER_API_KEY = "not-required"


@dataclass
class ProviderConfig:
    """A serializable description of a chat provider endpoint.

    Attributes:
        provider_type: One of :data:`PROVIDER_SPECS` or an OpenAI-compatible
            alias such as ``"openrouter"`` or ``"vllm"``.
        model: Model identifier passed to the provider.
        base_url: Optional endpoint override. This is what makes an
            OpenAI-shaped or Anthropic-shaped API reachable at a custom
            address (a gateway, a proxy, or a self-hosted server).
        api_key_env: Environment variable holding the API key. Defaults to the
            provider's conventional variable.
        provider_identifier: Optional override for the provider identifier
            string, where the provider supports one.
        settings: Sampling settings applied to the provider's defaults. Names
            must already exist on the provider, so typos are caught loudly.
        extra_settings: Additional request settings to *add*, for parameters a
            given endpoint understands but the provider does not declare.
        env_file: Optional ``.env`` file read before the API key is looked up.
            Variables already set in the environment win, so the file supplies
            a default rather than an override.
        timeout: Seconds to wait for a response before giving up. The SDKs
            default to 600 with two retries, so a stalled request can hang for
            half an hour looking like a crash; set this for anything running
            unattended.
        max_retries: How many times to retry a failed request.
    """

    provider_type: str
    model: str
    base_url: str | None = None
    api_key_env: str | None = None
    provider_identifier: str | None = None
    settings: dict[str, Any] = field(default_factory=dict)
    extra_settings: dict[str, Any] = field(default_factory=dict)
    env_file: str | None = None
    timeout: float | None = None
    max_retries: int | None = None

    # -- resolution --------------------------------------------------------

    def resolve_spec(self) -> tuple[ProviderSpec, str | None, str, bool]:
        """Return the spec, base URL, API key env var, and key-optional flag."""

        provider_type = self.provider_type
        alias = OPENAI_COMPATIBLE_DEFAULTS.get(provider_type)
        if alias is not None:
            alias_base_url, alias_key_env, key_optional = alias
            spec = PROVIDER_SPECS["openai"]
            base_url = self.base_url or (alias_base_url or None)
            if base_url is None:
                raise AgentConfigurationError(
                    f"Provider type '{provider_type}' requires an explicit "
                    "'base_url'."
                )
            return spec, base_url, self.api_key_env or alias_key_env, key_optional

        spec = PROVIDER_SPECS.get(provider_type)
        if spec is None:
            known = ", ".join(sorted({*PROVIDER_SPECS, *OPENAI_COMPATIBLE_DEFAULTS}))
            raise AgentConfigurationError(
                f"Unknown provider type: '{provider_type}'. Known types: {known}."
            )
        if self.base_url is not None and not spec.supports_base_url:
            raise AgentConfigurationError(
                f"Provider '{spec.name}' does not support a custom 'base_url'."
            )
        return (
            spec,
            self.base_url,
            self.api_key_env or spec.default_api_key_env,
            False,
        )

    def resolve_api_key(self, api_key_env: str, key_optional: bool = False) -> str:
        """Read the API key from the environment, or fail with a clear message."""

        if self.env_file and api_key_env not in os.environ:
            load_env_file(self.env_file)

        api_key = os.environ.get(api_key_env)
        if not api_key:
            if key_optional:
                return PLACEHOLDER_API_KEY
            raise AgentConfigurationError(
                f"Environment variable '{api_key_env}' is not set, so the "
                f"'{self.provider_type}' provider for model '{self.model}' "
                "cannot be built. Set it, or name a different variable with "
                "'api_key_env'."
            )
        return api_key

    # -- construction ------------------------------------------------------

    def build(self) -> Any:
        """Construct and return the configured chat provider."""

        spec, base_url, api_key_env, key_optional = self.resolve_spec()
        provider_cls = spec.load_class()

        kwargs: dict[str, Any] = {
            "api_key": self.resolve_api_key(api_key_env, key_optional),
            "model": self.model,
        }
        if base_url is not None:
            kwargs["base_url"] = base_url
        if self.timeout is not None:
            kwargs["timeout"] = self.timeout
        if self.max_retries is not None:
            kwargs["max_retries"] = self.max_retries
        if self.provider_identifier is not None:
            if not spec.supports_provider_identifier:
                raise AgentConfigurationError(
                    f"Provider '{spec.name}' does not support "
                    "'provider_identifier'."
                )
            kwargs["provider_identifier"] = self.provider_identifier

        try:
            provider = provider_cls(**kwargs)
        except TypeError as exc:
            raise AgentConfigurationError(
                f"Could not construct provider '{spec.name}': {exc}"
            ) from exc

        self.apply_settings(provider)
        return provider

    def apply_settings(self, provider: Any) -> None:
        """Apply configured sampling settings to ``provider``'s defaults.

        ``ProviderSettings.__setattr__`` silently creates a dead attribute for
        an unknown name, so settings are applied through ``set_value`` and an
        unknown name is reported rather than quietly ignored.
        """

        if not self.settings and not self.extra_settings:
            return

        settings = provider.get_default_settings()

        for name, value in self.settings.items():
            if settings.get_setting(name) is None:
                known = ", ".join(settings.setting_names()) or "<none>"
                raise AgentConfigurationError(
                    f"Provider '{self.provider_type}' has no setting '{name}'. "
                    f"Known settings: {known}. Use 'extra_settings' to send a "
                    "parameter the provider does not declare."
                )
            settings.set_value(name, value)

        for name, value in self.extra_settings.items():
            if settings.get_setting(name) is not None:
                raise AgentConfigurationError(
                    f"'{name}' is already a declared setting on provider "
                    f"'{self.provider_type}'; put it in 'settings', not "
                    "'extra_settings'. Adding it as an extra would silently "
                    "redefine the declared one."
                )
            settings.add_request_setting(name, value)

        provider.set_default_settings(settings)

    # -- serialization -----------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible representation. Never contains a secret."""

        data: dict[str, Any] = {"type": self.provider_type, "model": self.model}
        if self.base_url is not None:
            data["base_url"] = self.base_url
        if self.api_key_env is not None:
            data["api_key_env"] = self.api_key_env
        if self.provider_identifier is not None:
            data["provider_identifier"] = self.provider_identifier
        if self.settings:
            data["settings"] = dict(self.settings)
        if self.extra_settings:
            data["extra_settings"] = dict(self.extra_settings)
        if self.env_file is not None:
            data["env_file"] = self.env_file
        if self.timeout is not None:
            data["timeout"] = self.timeout
        if self.max_retries is not None:
            data["max_retries"] = self.max_retries
        return data

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ProviderConfig":
        """Restore a provider config from its JSON representation."""

        if not isinstance(data, Mapping):
            raise AgentConfigurationError(
                f"Provider config must be an object, got {type(data).__name__}."
            )
        if "api_key" in data:
            raise AgentConfigurationError(
                "Provider config must not contain a literal 'api_key'. Use "
                "'api_key_env' to name the environment variable holding it."
            )

        provider_type = data.get("type", data.get("provider_type"))
        if not provider_type:
            raise AgentConfigurationError("Provider config is missing 'type'.")
        model = data.get("model")
        if not model:
            raise AgentConfigurationError("Provider config is missing 'model'.")

        return cls(
            provider_type=str(provider_type),
            model=str(model),
            base_url=_optional_str(data.get("base_url")),
            api_key_env=_optional_str(data.get("api_key_env")),
            provider_identifier=_optional_str(data.get("provider_identifier")),
            settings=dict(data.get("settings") or {}),
            extra_settings=dict(data.get("extra_settings") or {}),
            env_file=_optional_str(data.get("env_file")),
            timeout=None if data.get("timeout") is None else float(data["timeout"]),
            max_retries=(
                None if data.get("max_retries") is None else int(data["max_retries"])
            ),
        )


@dataclass
class AgentConfig:
    """A named agent declared in pipeline JSON."""

    name: str
    provider: ProviderConfig
    agent_type: str = "chat_tool_agent"

    def build(self) -> Any:
        """Construct the configured agent."""

        if self.agent_type != "chat_tool_agent":
            raise AgentConfigurationError(
                f"Unknown agent type: '{self.agent_type}'. "
                "Supported types: chat_tool_agent."
            )
        from ToolAgents.agents import ChatToolAgent

        return ChatToolAgent(chat_api=self.provider.build())

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible representation."""

        data: dict[str, Any] = {
            "name": self.name,
            "provider": self.provider.to_dict(),
        }
        if self.agent_type != "chat_tool_agent":
            data["agent_type"] = self.agent_type
        return data

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "AgentConfig":
        """Restore an agent config from its JSON representation."""

        if not isinstance(data, Mapping):
            raise AgentConfigurationError(
                f"Agent config must be an object, got {type(data).__name__}."
            )
        name = data.get("name")
        if not name:
            raise AgentConfigurationError("Agent config is missing 'name'.")
        provider_data = data.get("provider")
        if provider_data is None:
            raise AgentConfigurationError(
                f"Agent '{name}' is missing a 'provider' block."
            )
        return cls(
            name=str(name),
            provider=ProviderConfig.from_dict(provider_data),
            agent_type=str(data.get("agent_type", "chat_tool_agent")),
        )


class LazyAgentRegistry(Mapping):
    """Build declared agents on first reference, not on load.

    Building eagerly means a document that merely *declares* an unused
    Anthropic agent fails to load with "ANTHROPIC_API_KEY is not set", even
    when every process references the OpenAI one. Constructing on demand keeps
    the failure attached to the agent actually being used.
    """

    def __init__(self, configs: Sequence[AgentConfig]) -> None:
        self._configs: dict[str, AgentConfig] = {}
        for config in configs:
            if config.name in self._configs:
                raise AgentConfigurationError(
                    f"Duplicate agent name in pipeline configuration: "
                    f"'{config.name}'."
                )
            self._configs[config.name] = config
        self._built: dict[str, Any] = {}

    def __getitem__(self, name: str) -> Any:
        if name not in self._built:
            if name not in self._configs:
                raise KeyError(name)
            self._built[name] = self._configs[name].build()
        return self._built[name]

    def get(self, name: str, default: Any = None) -> Any:
        try:
            return self[name]
        except KeyError:
            return default

    def __iter__(self):
        return iter(self._configs)

    def __len__(self) -> int:
        return len(self._configs)


def build_agents_from_configs(
    configs: Sequence[AgentConfig],
) -> dict[str, Any]:
    """Build every declared agent eagerly, keyed by name.

    Prefer :class:`LazyAgentRegistry` when some declared agents may go unused.
    """

    registry = LazyAgentRegistry(configs)
    return {name: registry[name] for name in registry}


def _optional_str(value: Any) -> str | None:
    """Normalize to a non-empty string, or ``None``.

    An empty ``base_url`` must become ``None`` so the provider default applies,
    rather than being handed to the SDK as a blank endpoint.
    """

    if value is None:
        return None
    text = str(value).strip()
    return text or None
