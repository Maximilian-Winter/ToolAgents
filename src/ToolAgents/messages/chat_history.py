import copy
import json
import os
from enum import Enum

from typing import List, Dict, Any, Optional

from ToolAgents.messages.message_template import MessageTemplate


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
    def __init__(self, role_templates: dict[str, str], prompt_layout_template: str = "{system_message}{available_tools}{chat_history}{last_user_message}", available_tools_template: str = None, include_system_message_in_first_user_message: bool = False):
        self.include_system_message_in_first_user_message = include_system_message_in_first_user_message
        self.role_templates: dict[str, MessageTemplate] = {}
        for key, value in role_templates.items():
            self.role_templates[key] = MessageTemplate.from_string(value)
        if prompt_layout_template is None:
            prompt_layout_template = "{system_message}{chat_history}{last_user_message}"
        self.prompt_layout_template = MessageTemplate.from_string(prompt_layout_template)

        if available_tools_template is None:
            available_tools_template = "{tools}"
        self.available_tool_template = MessageTemplate.from_string(available_tools_template)

    def format_messages(self, messages, tools: list = None):
        formatted_chat = []
        system_message = None
        tool_system_message = None
        if tools is not None and len(tools) > 0 and self.available_tool_template is not None and self.prompt_layout_template is not None:
            tools_open_ai = [tool.to_openai_tool() for tool in tools]
            tool_system_message = self.available_tool_template.generate_message_content(tools=tools_open_ai)
        for message in messages:
            role = message['role']
            content = message['content']
            template = self.role_templates[role]

            if self.include_system_message_in_first_user_message and role == "system":
                formatted_message = template.generate_message_content(content=content)
                system_message = formatted_message
            elif self.include_system_message_in_first_user_message and role == "user" and system_message is not None:
                formatted_message = template.generate_message_content(content=system_message + content)
                system_message = None
                formatted_chat.append(formatted_message)
            else:
                if content is None:
                    msg_copy = copy.deepcopy(message)
                    msg_copy.pop('role')
                    msg_copy.pop('content')
                    content = json.dumps(msg_copy)
                if role == "tool":
                    msg_copy = copy.deepcopy(message)
                    msg_copy.pop('role')
                    result = msg_copy.pop('content')
                    msg_copy["result"] = result
                    content = json.dumps(msg_copy)

                formatted_message = template.generate_message_content(content=content)
                formatted_chat.append(formatted_message)
        if self.include_system_message_in_first_user_message:
            last_user_message = ""
            if messages[-1]["role"] == "user":
                last_user_message = formatted_chat.pop(-1)
            if tool_system_message:
                return self.prompt_layout_template.generate_message_content(chat_history=''.join(formatted_chat),
                                                                            available_tools=tool_system_message,
                                                                            last_user_message=last_user_message)
            else:
                return self.prompt_layout_template.generate_message_content(chat_history=''.join(formatted_chat),
                                                                            last_user_message=last_user_message)

        else:
            last_user_message = ""
            sys = ""
            if messages[-1]["role"] == "user":
                last_user_message = formatted_chat.pop(-1)
            if messages[0]["role"] == "system":
                sys = formatted_chat.pop(0)
            if tool_system_message:
                return self.prompt_layout_template.generate_message_content(system_message=sys, chat_history=''.join(formatted_chat), available_tools=tool_system_message, last_user_message=last_user_message)
            else:
                return self.prompt_layout_template.generate_message_content(system_message=sys, chat_history=''.join(formatted_chat),  last_user_message=last_user_message)


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
    def __init__(self, history_folder: str = None) -> None:
        self.history_folder = history_folder
        self.messages: List[Message] = []

    def add_message(self, role: str, message: str):
        self.messages.append(Message(role=role, content=message))

    def add_user_message(self, message: str):
        self.messages.append(Message(role='user', content=message))

    def add_assistant_message(self, message: str, **kwargs):
        self.messages.append(Message(role='assistant', content=message, **kwargs))

    def add_system_message(self, message: str):
        self.messages.append(Message(role='system', content=message))

    def add_tool_message(self, message: str, **kwargs):
        self.messages.append(Message(role='tool', content=message, **kwargs))

    def to_list(self, message_filter_keys: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        return [message.to_dict(message_filter_keys) for message in self.messages]

    def save_history(self, filename: str) -> None:
        if self.history_folder is None:
            with open(filename, "w") as f:
                json.dump(self.to_list(), f, indent=2)
        else:
            if not os.path.exists(self.history_folder):
                os.makedirs(self.history_folder)
                full_filename = os.path.join(self.history_folder, filename)
                with open(full_filename, "w") as f:
                    json.dump(self.to_list(), f, indent=2)

    def load_history(self, filename: str) -> None:
        if self.history_folder is not None:
            full_filename = os.path.join(self.history_folder, filename)
        else:
            full_filename = filename

        try:
            with open(full_filename, "r") as f:
                loaded_messages = json.load(f)

            for msg_data in loaded_messages:
                self.messages.append(Message.from_dict(msg_data))
        except FileNotFoundError:
            print(f"History file not found: {full_filename}")
        except json.JSONDecodeError:
            print(f"Error decoding JSON from file: {full_filename}")

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
            self.messages.append(Message.from_dict(msg_data))
