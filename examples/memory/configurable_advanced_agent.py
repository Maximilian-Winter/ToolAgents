import dataclasses
import json
import os
from enum import Enum
from typing import Any

from ToolAgents import ToolRegistry, FunctionTool
from ToolAgents.agent_memory.context_app_state import ContextAppState
from ToolAgents.agent_memory.semantic_memory.memory import SemanticMemory
from ToolAgents.interfaces import LLMSamplingSettings
from ToolAgents.interfaces.base_llm_agent import BaseToolAgent

class AgentOutputType(Enum):
    STDOUT = "stdout"
    FILE = "file"
    FUNCTION = "function"

@dataclasses.dataclass
class AgentOutputSettings:
    output_type: AgentOutputType = AgentOutputType.STDOUT
    additional_settings: dict[str, Any] = dataclasses.field(default_factory=dict)

@dataclasses.dataclass
class AgentConfig:
    system_message: str = "You are a helpful assistant."
    save_dir: str = "./default_agent"
    initial_state_file: str = None
    max_chat_history_length: int = -1
    use_semantic_memory: bool = False


class ConfigurableAdvancedAgent:
    def __init__(self, agent: BaseToolAgent, tool_registry: ToolRegistry = None, output_settings: AgentOutputSettings = None, agent_config: AgentConfig = None):
        if agent_config is None:
            agent_config = AgentConfig()

        if output_settings is None:
            self.output_settings = AgentOutputSettings()

        self.tool_registry = tool_registry
        self.agent = agent

        save_dir = agent_config.save_dir
        initial_state_file = agent_config.initial_state_file
        system_message = agent_config.system_message

        self.use_semantic_memory = agent_config.use_semantic_memory
        self.max_chat_history_length = agent_config.max_chat_history_length

        load = True
        if not os.path.isdir(save_dir):
            os.makedirs(save_dir)
            load = False

        self.save_dir = save_dir
        self.has_app_state = False
        self.system_message = system_message
        self.agent_config_path = os.path.join(self.save_dir, "agent_config.json")
        self.app_state_path = os.path.join(save_dir, "app_state.json")
        self.semantic_memory_path = os.path.join(save_dir, "semantic_memory")
        self.app_state = None
        self.chat_history_index = 0
        self.chat_history = []

        if load:
            self.load_agent()
        else:
            if initial_state_file:
                self.has_app_state = True
                self.app_state = ContextAppState(initial_state_file)

            self.save_agent()
        if self.use_semantic_memory:
            self.semantic_memory = SemanticMemory(self.semantic_memory_path)

    def set_tool_registry(self, tool_registry: ToolRegistry):
        self.tool_registry = tool_registry

    def set_tools(self, tools: list[FunctionTool]):
        self.tool_registry = ToolRegistry()
        self.tool_registry.add_tools(tools)

    def remove_all_tools(self):
        self.tool_registry = None

    def chat_with_agent(self, chat_input: str, settings: LLMSamplingSettings = None):
        user = chat_input
        if not self.has_app_state:
            chat_history = [{"role": "system", "content": self.system_message}]
        else:
            chat_history = [{"role": "system", "content": self.system_message.format(app_state=self.app_state.get_app_state_string())}]
        chat_history.extend(self.chat_history[self.chat_history_index:])
        if self.use_semantic_memory:
            results = self.semantic_memory.recall(user, 3)

            additional_context = "\n--- Additional Context From Past Interactions ---\n"
            for r in results:
                additional_context += f"Memories: {r['content']}\n\n---\n\n"

            user += '\n' + additional_context.strip()
        chat_history.append({"role": "user", "content": user})
        result = self.agent.get_response(chat_history, self.tool_registry, settings=settings)

        if len(self.chat_history) > self.max_chat_history_length and self.max_chat_history_length > -1:
            self.chat_history_index = self.chat_history_index + 1
        return result


    def load_agent(self):
        if not os.path.isfile(self.agent_config_path):
            return
        with open(self.agent_config_path, "r") as f:
            loaded_data = json.load(fp=f)
        self.system_message = loaded_data["system_message"]
        self.has_app_state = loaded_data["has_app_state"]
        self.chat_history = loaded_data["chat_history"]
        self.chat_history_index = loaded_data["chat_history_index"]
        self.max_chat_history_length = loaded_data["max_chat_history_length"]
        self.use_semantic_memory = loaded_data["use_semantic_memory"]

        if self.has_app_state:
            self.app_state = ContextAppState()
            self.app_state.load_json(self.app_state_path)

    def save_agent(self):
        with open(self.agent_config_path, "w") as f:
            save_data = {
                "system_message": self.system_message,
                "has_app_state": self.has_app_state,
                "chat_history": self.chat_history,
                "chat_history_index": self.chat_history_index,
                "max_chat_history_length": self.max_chat_history_length,
                "use_semantic_memory": self.use_semantic_memory,
            }
            json.dump(save_data, fp=f)
        if self.has_app_state:
            self.app_state.save_json(self.app_state_path)

