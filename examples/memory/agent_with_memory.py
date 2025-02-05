import json
import os
from typing import Any

from ToolAgents.agent_memory.context_app_state import ContextAppState
from ToolAgents.agent_memory.semantic_memory.memory import SemanticMemory


class AgentWithMemory:
    def __init__(self, save_dir: str = "./default_agent", initial_state_file: str = None):
        load = True
        if not os.path.isdir(save_dir):
            os.makedirs(save_dir)
            load = False
        self.save_dir = save_dir
        self.has_app_state = False
        self.agent_config_path = os.path.join(self.save_dir, "agent_config.json")
        self.app_state_path = os.path.join(save_dir, "app_state.json")
        self.semantic_memory_path = os.path.join(save_dir, "semantic_memory")
        self.app_state = None
        self.chat_history_index = 0
        self.chat_history = []
        self.semantic_memory = SemanticMemory(self.semantic_memory_path)
        if load:
            self.load_agent()
        else:
            if initial_state_file:
                self.
            self.save_agent()

    def load_agent(self):
        if not os.path.isfile(self.agent_config_path):
            return
        with open(self.agent_config_path, "r") as f:
            loaded_data = json.load(fp=f)
        self.has_app_state = loaded_data["has_app_state"]
        self.chat_history = loaded_data["chat_history"]
        self.chat_history_index = loaded_data["chat_history_index"]
        if self.has_app_state:
            self.app_state = ContextAppState()
            self.app_state.load_json(self.app_state_path)

    def save_agent(self):
        with open(self.agent_config_path, "w") as f:
            save_data = {
                "has_app_state": self.has_app_state,
                "chat_history": self.chat_history,
                "chat_history_index": self.chat_history_index
            }
            json.dump(save_data, fp=f)