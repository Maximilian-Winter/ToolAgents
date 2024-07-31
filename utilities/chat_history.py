import datetime
import json
import os
from typing import List, Dict, Any


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


class Message:
    def __init__(self, role: str, content: str, **kwargs):
        self.role = role
        self.content = content

        self.__dict__.update(kwargs)

    def to_dict(self) -> Dict[str, Any]:
        result = {"role": self.role, "content": self.content}
        result.update({k: v for k, v in self.__dict__.items() if k not in ['role', 'content']})
        return result


class ChatHistory:
    def __init__(self, history_folder: str):
        self.messages: List[Message] = []
        self.history_folder = history_folder

    def add_message(self, message: Message):
        self.messages.append(message)

    def edit_message(self, message_id: int, new_content: str) -> bool:
        for message in self.messages:
            if message.id == message_id:
                message.content = new_content
                return True
        return False

    def to_list(self) -> List[Dict[str, Any]]:
        return [message.to_dict() for message in self.messages]

    def save_history(self):
        if not os.path.exists(self.history_folder):
            os.makedirs(self.history_folder)

        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        save_id = f"{timestamp}"
        filename = f"chat_history_{save_id}.json"

        with open(f"{self.history_folder}/{filename}", "w") as f:
            json.dump(self.to_list(), f, indent=2)

    def load_history(self):
        if not os.path.exists(self.history_folder):
            os.makedirs(self.history_folder)
            print("No chat history found. Starting with an empty history.")
            self.messages = []
            return

        history_files = [f for f in os.listdir(self.history_folder) if
                         f.startswith("chat_history_") and f.endswith(".json")]

        if not history_files:
            print("No chat history found. Starting with an empty history.")
            self.messages = []
            return

        # Sort history files based on the timestamp in the filename
        latest_history = sorted(history_files, reverse=True)[0]

        try:
            with open(f"{self.history_folder}/{latest_history}", "r") as f:
                loaded_history = json.load(f)
                self.messages = [Message(msg['role'], msg['content'], id=msg.get('id')) for msg in loaded_history]
            print(f"Loaded the most recent chat history: {latest_history}")

        except (FileNotFoundError, json.JSONDecodeError) as e:
            print(f"Error loading chat history: {e}. Starting with an empty history.")
            self.messages = []

    def delete_last_messages(self, k: int) -> int:
        if k >= len(self.messages):
            deleted = len(self.messages)
            self.messages.clear()
        else:
            deleted = k
            self.messages = self.messages[:-k]
        return deleted

    def delete_message(self, msg_id: int) -> int:
        for msg in self.messages:
            if msg.id == msg_id:
                self.messages.remove(msg)
                return True
        return False
