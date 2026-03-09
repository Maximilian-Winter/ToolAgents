from __future__ import annotations

from typing import Iterable

from ToolAgents.data_models.chat_history import ChatHistory, Chats
from ToolAgents.data_models.messages import ChatMessage, ChatMessageRole


class AdvancedChatFormatter:
    def __init__(self, role_templates: dict[str, str]):
        self.role_templates = role_templates

    def format_messages(self, messages: Iterable[ChatMessage | dict]) -> str:
        formatted_messages: list[str] = []
        for message in messages:
            if isinstance(message, ChatMessage):
                role = message.role.value
                content = message.get_as_text()
            else:
                role = message['role']
                content = message.get('content', '')
            template = self.role_templates.get(role)
            if template is None:
                formatted_messages.append(content)
            else:
                formatted_messages.append(template.format(content=content))
        return ''.join(formatted_messages)


__all__ = ['AdvancedChatFormatter', 'ChatHistory', 'ChatMessage', 'ChatMessageRole', 'Chats']
