import datetime

import json
import os
from typing import List, Dict, Any

from ToolAgents.utilities.message_template import MessageTemplate


class ChatFormatter:
    def __init__(self, template, role_names: Dict[str, str] = None):
        self.template = template
        self.role_names = role_names or {}

    def format_messages(self, messages):
        formatted_chat = []
        for message in messages:
            role = message['role']
            content = message['content']
            display_name = self.role_names.get(role, role.capitalize())
            formatted_message = self.template.format(role=display_name, content=content)
            formatted_chat.append(formatted_message)
        return '\n'.join(formatted_chat)


class AdvancedChatFormatter:
    def __init__(self, role_templates: dict[str, str]):
        self.role_templates: dict[str, MessageTemplate] = {}
        for key, value in role_templates.items():
            self.role_templates[key] = MessageTemplate.from_string(value)

    def format_messages(self, messages):
        formatted_chat = []
        for message in messages:
            role = message['role']
            content = message['content']
            template = self.role_templates[role]
            formatted_message = template.generate_message_content(content=content)
            formatted_chat.append(formatted_message)
        return ''.join(formatted_chat)


class Message:
    def __init__(self, role: str, content: str, **kwargs):
        self.role = role
        self.content = content
        self.__dict__.update(kwargs)

    def to_dict(self, filter_keys: list[str] = None) -> Dict[str, Any]:
        result = {"role": self.role, "content": self.content}
        if filter_keys is None:
            filter_keys = ["role", "content"]
        else:
            filter_keys.extend(["role", "content"])

        result.update({k: v for k, v in self.__dict__.items() if k not in filter_keys})
        return result

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Message':
        data = data.copy()
        role = data.pop('role')
        content = data.pop('content')
        return cls(role, content, **data)


class ChatHistory:
    def __init__(self):
        self.messages: List[Message] = []

    def add_message(self, message: Message):
        self.messages.append(message)

    def to_list(self) -> List[Dict[str, Any]]:
        return [message.to_dict() for message in self.messages]

    def save_history(self, filename: str) -> None:
        if not os.path.exists(os.path.dirname(filename)):
            os.makedirs(os.path.dirname(filename))

        with open(filename, "w") as f:
            json.dump(self.to_list(), f, indent=2)

    def load_history(self, filename: str) -> None:
        with open(filename, "r") as f:
            loaded_messages = json.load(f)

        for msg_data in loaded_messages:
            self.add_message(Message.from_dict(msg_data))

    def delete_last_messages(self, k: int) -> int:
        if k >= len(self.messages):
            deleted = len(self.messages)
            self.messages.clear()
        else:
            deleted = k
            self.messages = self.messages[:-k]
        return deleted

    def add_list_of_dicts(self, message_list: List[Dict[str, Any]]) -> None:
        msgs = message_list.copy()
        for msg_data in msgs:
            self.add_message(Message.from_dict(msg_data))
