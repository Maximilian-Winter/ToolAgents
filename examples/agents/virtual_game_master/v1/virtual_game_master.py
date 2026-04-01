import datetime
import os
import json
from typing import Any, Tuple, Generator, Dict, Optional

from dotenv import load_dotenv
from pathlib import Path

from ToolAgents.agent_memory.context_app_state import ContextAppState
from ToolAgents.agents import ChatToolAgent
from ToolAgents.data_models.chat_history import ChatHistory
from ToolAgents.data_models.messages import ChatMessage
from ToolAgents.utilities.message_template import MessageTemplate, ChatFormatter

from command_system import CommandSystem


class VirtualGameMasterConfig:
    def __init__(self):
        self.GAME_SAVE_FOLDER: str = ""
        self.INITIAL_GAME_STATE: str = ""
        self.MAX_MESSAGES: int = 0
        self.KEPT_MESSAGES: int = 0
        self.SYSTEM_MESSAGE_FILE: str = ""
        self.SAVE_SYSTEM_MESSAGE_FILE: str = ""
        self.MAX_TOKENS: int = 0
        self.API_TYPE: str = "openai"
        self.API_KEY: str | None = None
        self.API_URL: str = ""
        self.MODEL: str = ""
        self.TEMPERATURE: float = 0.7
        self.TOP_P: float = 1.0
        self.TOP_K: int = 0
        self.MIN_P: float = 0.0
        self.TFS_Z: float = 1.0
        self.COMMAND_PREFIX: str = "@"
        self.STOP_SEQUENCES: str = "[]"

    @classmethod
    def from_env(cls, env_file: str = ".env") -> "VirtualGameMasterConfig":
        load_dotenv(env_file)
        config = cls()
        config.GAME_SAVE_FOLDER = os.getenv("GAME_SAVE_FOLDER")
        config.INITIAL_GAME_STATE = os.getenv("INITIAL_GAME_STATE")
        config.MAX_MESSAGES = int(os.getenv("MAX_MESSAGES"))
        config.KEPT_MESSAGES = int(os.getenv("KEPT_MESSAGES"))
        config.SYSTEM_MESSAGE_FILE = os.getenv("SYSTEM_MESSAGE_FILE")
        config.SAVE_SYSTEM_MESSAGE_FILE = os.getenv("SAVE_SYSTEM_MESSAGE_FILE")
        config.MAX_TOKENS = int(os.getenv("MAX_TOKENS_PER_RESPONSE"))
        config.API_TYPE = os.getenv("API_TYPE", "openai").lower()
        config.API_KEY = os.getenv("API_KEY", None)
        config.API_URL = os.getenv("API_URL")
        config.MODEL = os.getenv("MODEL")
        config.TEMPERATURE = float(os.getenv("TEMPERATURE", 0.7))
        config.TOP_P = float(os.getenv("TOP_P", 1.0))
        config.TOP_K = int(os.getenv("TOP_K", 0))
        config.MIN_P = float(os.getenv("MIN_P", 0.0))
        config.TFS_Z = float(os.getenv("TFS_Z", 1.0))
        config.COMMAND_PREFIX = os.getenv("COMMAND_PREFIX", "@")
        config.STOP_SEQUENCES = os.getenv("STOP_SEQUENCES", "[]")
        return config

    @classmethod
    def from_json(cls, json_file: str) -> "VirtualGameMasterConfig":
        with open(json_file, "r") as f:
            data = json.load(f)
        config = cls()
        for key, value in data.items():
            if hasattr(config, key):
                setattr(config, key, config._parse_value(key, value))
        return config

    def to_env(self, env_file: str = ".env") -> None:
        with open(env_file, "w") as f:
            for key, value in self.__dict__.items():
                f.write(f"{key}={value}\n")

    def to_json(self, json_file: str) -> None:
        with open(json_file, "w") as f:
            json.dump(self.__dict__, f, indent=2)

    def _parse_value(self, key: str, value: Any) -> Any:
        current_value = getattr(self, key)
        if isinstance(current_value, bool):
            return str(value).lower() in ("true", "1", "yes")
        if current_value is None:
            return value
        return type(current_value)(value)

    def update(self, updates: Dict[str, Any]) -> None:
        for key, value in updates.items():
            if hasattr(self, key):
                setattr(self, key, self._parse_value(key, value))

    def to_dict(self) -> Dict[str, Any]:
        return {key: value for key, value in self.__dict__.items()}


class VirtualGameMaster:
    def __init__(self, config: VirtualGameMasterConfig, tool_agent: ChatToolAgent, debug_mode: bool = False):
        self.config = config
        CommandSystem.command_prefix = self.config.COMMAND_PREFIX
        self.tool_agent = tool_agent
        self.system_message_template = MessageTemplate.from_file(
            config.SYSTEM_MESSAGE_FILE
        )
        self.save_system_message_template = MessageTemplate.from_file(
            config.SAVE_SYSTEM_MESSAGE_FILE
        )

        self.game_state = ContextAppState(config.INITIAL_GAME_STATE)
        self.history = ChatHistory(title="Game")
        self.game_save_folder = config.GAME_SAVE_FOLDER
        self.game_json_filepath = str(Path(self.game_save_folder).joinpath("current_history.json"))
        self.history_offset = 0

        self.debug_mode = debug_mode
        self.max_messages = config.MAX_MESSAGES
        self.kept_messages = config.KEPT_MESSAGES

    def process_input(self, user_input: str, stream: bool) -> Tuple[str, bool] | Tuple[Generator[str, None, None], bool]:
        if user_input.startswith(CommandSystem.command_prefix):
            return CommandSystem.handle_command(self, user_input)

        if stream:
            return self.get_streaming_response(user_input), False
        return self.get_response(user_input), False

    def get_response(self, user_input: str) -> str:
        history = self.pre_response(user_input)
        response = self.tool_agent.get_response(history)
        self.post_response(response.response.strip())
        return response.response.strip()

    def get_streaming_response(self, user_input: str) -> Generator[str, None, None]:
        history = self.pre_response(user_input)
        full_response = ""
        for response_chunk in self.tool_agent.get_streaming_response(history):
            full_response += response_chunk.chunk
            yield response_chunk.chunk
        self.post_response(full_response)

    def pre_response(self, user_input: str) -> list[ChatMessage]:
        self.history.add_user_message(user_input.strip())

        messages = self.history.get_messages()
        active_messages = messages[self.history_offset:]
        system_msg = ChatMessage.create_system_message(self.get_current_system_message())

        result = [system_msg] + active_messages

        if self.debug_mode:
            print(system_msg.get_as_text())

        return result

    def get_current_system_message(self):
        return self.system_message_template.generate_message_content(
            self.game_state.template_fields).strip()

    def format_history(self, history: list[ChatMessage]) -> str:
        template = "{role}: {content}\n\n"
        role_names = {
            "assistant": "Game Master",
            "user": "Player"
        }
        formatter = ChatFormatter(template, role_names)

        if len(history) > 0:
            output = "History:\n"
            output += formatter.format_messages(history)
        else:
            output = "History is empty.\n"

        return output

    def get_complete_history_formatted(self):
        history = self.history.get_messages()
        return self.format_history(history=history)

    def get_current_history_formatted(self):
        history = self.get_currently_used_history()
        return self.format_history(history=history)

    def post_response(self, response: str) -> None:
        if len(response.strip()) > 0:
            self.history.add_assistant_message(response.strip())
            self.history.save_to_json(self.game_json_filepath)

            active_count = len(self.history.get_messages()) - self.history_offset
            if active_count >= self.max_messages:
                self.generate_save_state()

    def edit_message(self, index_or_id, new_content: str) -> bool:
        """Edit a message by list index (int) or UUID (str).

        Args:
            index_or_id: Either an integer index into the message list,
                or a UUID string matching a message's id.
            new_content: The new text content for the message.

        Returns:
            True if the message was found and edited.
        """
        messages = self.history.get_messages()
        target: Optional[ChatMessage] = None

        if isinstance(index_or_id, int):
            if 0 <= index_or_id < len(messages):
                target = messages[index_or_id]
        else:
            for msg in messages:
                if msg.id == index_or_id:
                    target = msg
                    break

        if target is None:
            return False

        # Update the text content — ChatMessage stores content as a list
        # of content objects. Replace the first TextContent found.
        from ToolAgents.data_models.messages import TextContent
        for i, content_item in enumerate(target.content):
            if isinstance(content_item, TextContent):
                target.content[i] = TextContent(content=new_content)
                target.updated_at = datetime.datetime.now()
                self.history.save_to_json(self.game_json_filepath)
                return True

        # No TextContent found — add one
        target.content = [TextContent(content=new_content)]
        target.updated_at = datetime.datetime.now()
        self.history.save_to_json(self.game_json_filepath)
        return True

    def delete_last_message(self) -> bool:
        """Remove the last message from history."""
        messages = self.history.get_messages()
        if messages:
            self.history.remove_last_message()
            self.history.save_to_json(self.game_json_filepath)
            return True
        return False

    def manual_save(self):
        self.generate_save_state()

    def get_currently_used_history(self):
        return self.history.get_messages()[self.history_offset:]

    def get_message_count(self) -> int:
        """Total messages in history."""
        return self.history.get_message_count()

    def get_active_message_count(self) -> int:
        """Messages in the active window (after offset)."""
        return self.history.get_message_count() - self.history_offset

    def generate_save_state(self):
        history = self.get_currently_used_history()

        template = "{role}: {content}\n\n"
        role_names = {
            "assistant": "Game Master",
            "user": "Player"
        }
        formatter = ChatFormatter(template, role_names)
        formatted_chat = formatter.format_messages(history)

        prompt = self.save_system_message_template.generate_message_content(
            template_fields=self.game_state.template_fields,
            CHAT_HISTORY=formatted_chat)

        if self.debug_mode:
            print(prompt)

        prompt_messages = [
            ChatMessage.create_system_message(
                "You are an AI assistant tasked with updating the game state of a text-based role-playing game."
            ),
            ChatMessage.create_user_message(prompt),
        ]
        response_gen = self.tool_agent.get_streaming_response(prompt_messages)

        full_response = ""
        for response_chunk in response_gen:
            full_response += response_chunk.chunk
            print(response_chunk.chunk, end="", flush=True)

        if self.debug_mode:
            print(f"Update game info:\n{full_response}")

        self.game_state.update_from_xml(full_response)
        self.history_offset = len(self.history.get_messages()) - self.kept_messages

        self.save()

    def save(self):
        self.history.save_to_json(self.game_json_filepath)
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"save_state_{timestamp}.json"

        save_data = {
            "config": self.config.to_dict(),
            "template_fields": self.game_state.template_fields,
            "history_offset": self.history_offset,
        }
        with open(f"{self.config.GAME_SAVE_FOLDER}/{filename}", "w") as f:
            json.dump(save_data, f)

    def load(self):
        if os.path.exists(self.game_json_filepath):
            self.history = ChatHistory.load_from_json(self.game_json_filepath)
        else:
            print("No save state found. Starting a new game.")
            return

        save_files = [
            f for f in os.listdir(self.config.GAME_SAVE_FOLDER)
            if f.startswith("save_state_") and f.endswith(".json")
        ]

        if not save_files:
            print("No save state found. Starting a new game.")
            return

        latest_save = sorted(save_files, reverse=True)[0]

        try:
            with open(f"{self.config.GAME_SAVE_FOLDER}/{latest_save}", "r") as f:
                save_data = json.load(f)
            self.game_state.template_fields = save_data.get(
                "template_fields", self.game_state.template_fields
            )
            self.history_offset = save_data.get("history_offset", 0)
            print(f"Loaded the most recent game state: {latest_save}")
        except (FileNotFoundError, json.JSONDecodeError) as e:
            print(f"Error loading save state: {e}. Starting a new game.")
