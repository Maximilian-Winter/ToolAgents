import abc
import copy
import datetime
import enum
import json
from dataclasses import dataclass
from typing import List, Dict, Optional, Generator, Any, Union, AsyncGenerator

from ToolAgents import FunctionTool, ToolRegistry
from pydantic import BaseModel, Field
from typing import Dict, Optional, Any, Union

from ToolAgents.data_models.messages import ChatMessage, StreamingChatMessage
from ToolAgents.utilities.logger_utilities import EasyLogger





class SettingLevel(enum.Enum):
    """Where a setting should be placed in the API request."""
    PROVIDER = "provider"  # Top-level request parameter
    REQUEST = "request"
    METADATA = "metadata"  # Not sent to API, used for tracking


@dataclass
class LLMSetting:
    """
    Represents a single LLM setting with default and neutral values.

    Simplified from the original - just stores the values directly
    instead of wrapping them in dicts.
    """
    name: str
    default_value: Any
    neutral_value: Any
    level: SettingLevel = SettingLevel.PROVIDER
    current_value: Any = None

    def __post_init__(self):
        if self.current_value is None:
            self.current_value = self.default_value

    def get_value(self) -> Any:
        """Get current value."""
        return self.current_value

    def set_value(self, value: Any) -> None:
        """Set current value."""
        self.current_value = value

    def reset(self) -> None:
        """Reset to default value."""
        self.current_value = self.default_value

    def neutralize(self) -> None:
        """Set to neutral (disabled) value."""
        self.current_value = self.neutral_value

    def is_neutral(self) -> bool:
        """Check if currently at neutral value."""
        return self.current_value == self.neutral_value


class ProviderSettings:
    """
    Manages LLM settings with support for different placement levels.

    Settings can be at provider level (top-level params) or in extra_body.
    """

    def __init__(self, settings: Optional[List[LLMSetting]] = None):
        self._settings: Dict[str, LLMSetting] = {}
        if settings:
            for setting in settings:
                self._settings[setting.name] = setting

    def add_setting(self, setting: LLMSetting) -> None:
        """Add a new setting."""
        self._settings[setting.name] = setting

    def add_request_setting(self, name: str, value: Any) -> None:
        """Add a new setting."""
        setting = LLMSetting(name, value, value, level=SettingLevel.REQUEST)
        self._settings[setting.name] = setting

    def add_provider_setting(self, name: str, value: Any) -> None:
        """Add a new setting."""
        setting = LLMSetting(name, value, value, level=SettingLevel.PROVIDER)
        self._settings[setting.name] = setting

    def remove_setting(self, name: str) -> None:
        """Remove a setting by name."""
        if name in self._settings:
            del self._settings[name]

    def get_setting(self, name: str) -> Optional[LLMSetting]:
        """Get a setting object by name."""
        return self._settings.get(name)

    def get_value(self, name: str) -> Any:
        """Get current value of a setting."""
        setting = self._settings.get(name)
        return setting.get_value() if setting else None

    def set_value(self, name: str, value: Any) -> None:
        """Set current value of a setting."""
        if name in self._settings:
            self._settings[name].set_value(value)
        else:
            raise KeyError(name)

    def update(self, **kwargs) -> None:
        """Update multiple settings at once."""
        for name, value in kwargs.items():
            self.set_value(name, value)

    def reset(self, name: str) -> None:
        """Reset a setting to default."""
        if name in self._settings:
            self._settings[name].reset()

    def reset_all(self) -> None:
        """Reset all settings to defaults."""
        for setting in self._settings.values():
            setting.reset()

    def neutralize(self, name: str) -> None:
        """Set a setting to neutral value."""
        if name in self._settings:
            self._settings[name].neutralize()

    def neutralize_all(self) -> None:
        """Set all settings to neutral values."""
        for setting in self._settings.values():
            setting.neutralize()

    def to_dict(
            self,
            include: Optional[List[str]] = None,
            exclude: Optional[List[str]] = None,
            include_neutral: bool = True,
            param_mapping: Optional[Dict[str, str]] = None
    ) -> Dict[str, Any]:
        """
        Convert settings to dict for API request.

        Args:
            include: Only include these setting names
            exclude: Exclude these setting names
            include_neutral: Include settings at neutral value
            param_mapping: Rename parameters (e.g., {'max_tokens': 'max_new_tokens'})
        Returns:
            Dict with provider-level settings at top level and extra_body nested
        """
        result = { "PROVIDER_SETTINGS": {}, "REQUEST_SETTINGS": {}, "METADATA": {} }

        for name, setting in self._settings.items():
            # Filter by include/exclude
            if include and name not in include:
                continue
            if exclude and name in exclude:
                continue

            # Skip neutral values unless explicitly included
            if not include_neutral and setting.is_neutral():
                continue

            # Get the value
            value = setting.get_value()

            # Apply parameter mapping
            output_name = param_mapping.get(name, name) if param_mapping else name

            # Place in appropriate location based on level
            if setting.level == SettingLevel.PROVIDER:
                result["PROVIDER_SETTINGS"][output_name] = value
            elif setting.level == SettingLevel.REQUEST:
                result["REQUEST_SETTINGS"][output_name] = value
            elif setting.level == SettingLevel.METADATA:
                result["METADATA"][output_name] = value

        return result

    def copy(self) -> "ProviderSettings":
        """Create a deep copy of settings."""
        return copy.deepcopy(self)

    def __getitem__(self, name: str) -> Any:
        """Allow dict-like access: settings['temperature']"""
        return self.get_value(name)

    def __setitem__(self, name: str, value: Any) -> None:
        """Allow dict-like assignment: settings['temperature'] = 0.7"""
        self.set_value(name, value)

    def __contains__(self, name: str) -> bool:
        """Check if setting exists: 'temperature' in settings"""
        return name in self._settings

    def __repr__(self) -> str:
        values = {name: setting.get_value() for name, setting in self._settings.items()}
        return f"ProviderSettings({values})"

    def __getattr__(self, name: str) -> Any:
        """Allow attribute access: settings.temperature"""
        if name.startswith('_'):
            raise AttributeError(name)
        if name in self._settings:
            return self.get_value(name)
        raise AttributeError(f"No setting named '{name}'")

    def __setattr__(self, name: str, value: Any) -> None:
        """Allow attribute assignment: settings.temperature = 0.7"""
        if name.startswith('_'):
            object.__setattr__(self, name, value)
        elif hasattr(self, '_settings') and name in self._settings:
            self.set_value(name, value)
        else:
            object.__setattr__(self, name, value)
# Convenience function for creating common settings
def create_standard_settings() -> ProviderSettings:
    """Create standard LLM settings used by most providers."""
    return ProviderSettings([
        LLMSetting("temperature", default_value=1.0, neutral_value=1.0, level=SettingLevel.REQUEST),
        LLMSetting("top_p", default_value=1.0, neutral_value=1.0, level=SettingLevel.REQUEST),
        LLMSetting("top_k", default_value=0, neutral_value=0, level=SettingLevel.REQUEST),
        LLMSetting("max_tokens", default_value=4096, neutral_value=4096, level=SettingLevel.REQUEST),
    ])


def create_openai_settings() -> ProviderSettings:
    """Create OpenAI-specific settings."""
    return ProviderSettings([
        LLMSetting("temperature", default_value=1.0, neutral_value=1.0, level=SettingLevel.REQUEST),
        LLMSetting("top_p", default_value=1.0, neutral_value=1.0, level=SettingLevel.REQUEST),
        LLMSetting("max_tokens", default_value=4096, neutral_value=4096, level=SettingLevel.REQUEST),
        LLMSetting("frequency_penalty", default_value=0.0, neutral_value=0., level=SettingLevel.REQUEST),
        LLMSetting("presence_penalty", default_value=0.0, neutral_value=0.0, level=SettingLevel.REQUEST),
    ])


def create_anthropic_settings() -> ProviderSettings:
    """Create Anthropic-specific settings."""
    return ProviderSettings([
        LLMSetting("temperature", default_value=1.0, neutral_value=1.0, level=SettingLevel.REQUEST),
        LLMSetting("top_p", default_value=1.0, neutral_value=1.0, level=SettingLevel.REQUEST),
        LLMSetting("top_k", default_value=0, neutral_value=0, level=SettingLevel.REQUEST),
        LLMSetting("max_tokens", default_value=4096, neutral_value=4096, level=SettingLevel.REQUEST),
    ])

class ChatAPIProvider(abc.ABC):

    @abc.abstractmethod
    def get_response(
        self,
        messages: List[ChatMessage],
        settings: ProviderSettings = None,
        tools: Optional[List[FunctionTool]] = None,
    ) -> ChatMessage:
        pass

    @abc.abstractmethod
    def get_streaming_response(
        self,
        messages: List[ChatMessage],
        settings: ProviderSettings = None,
        tools: Optional[List[FunctionTool]] = None,
    ) -> Generator[StreamingChatMessage, None, None]:
        pass

    @abc.abstractmethod
    def get_default_settings(self):
        pass

    @abc.abstractmethod
    def set_default_settings(self, settings) -> None:
        pass

    @abc.abstractmethod
    def get_provider_identifier(self) -> str:
        pass


class AsyncChatAPIProvider(abc.ABC):

    @abc.abstractmethod
    async def get_response(
        self,
        messages: List[ChatMessage],
        settings: ProviderSettings = None,
        tools: Optional[List[FunctionTool]] = None,
    ) -> ChatMessage:
        pass

    @abc.abstractmethod
    async def get_streaming_response(
        self,
        messages: List[ChatMessage],
        settings: ProviderSettings = None,
        tools: Optional[List[FunctionTool]] = None,
    ) -> AsyncGenerator[StreamingChatMessage, None]:
        pass

    @abc.abstractmethod
    def get_default_settings(self):
        pass

    @abc.abstractmethod
    def set_default_settings(self, settings) -> None:
        pass

    @abc.abstractmethod
    def get_provider_identifier(self) -> str:
        pass
