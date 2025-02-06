import dataclasses
import json
import os
from typing import Any

from ToolAgents import ToolRegistry, FunctionTool
from ToolAgents.agent_memory.context_app_state import ContextAppState
from ToolAgents.agent_memory.semantic_memory.memory import SemanticMemory, SemanticMemoryConfig
from ToolAgents.interfaces import LLMSamplingSettings
from ToolAgents.interfaces.base_llm_agent import BaseToolAgent

@dataclasses.dataclass
class AgentConfig:
    system_message: str = "You are a helpful assistant."
    save_dir: str = "./default_agent"
    initial_app_state_file: str = None
    give_agent_edit_tool = False
    max_chat_history_length: int = -1
    use_semantic_chat_history_memory: bool = False
    save_on_creation: bool = False
    semantic_chat_history_config: SemanticMemoryConfig = dataclasses.field(default_factory=SemanticMemoryConfig)


class AdvancedAgent:
    def __init__(self, agent: BaseToolAgent, tool_registry: ToolRegistry = None, agent_config: AgentConfig = None):
        if agent_config is None:
            agent_config = AgentConfig()

        self.tool_registry = tool_registry
        self.agent = agent

        save_dir = agent_config.save_dir
        initial_state_file = agent_config.initial_app_state_file
        system_message = agent_config.system_message

        self.give_agent_edit_tool = agent_config.give_agent_edit_tool
        self.use_semantic_memory = agent_config.use_semantic_chat_history_memory
        self.max_chat_history_length = agent_config.max_chat_history_length

        load = False

        if not os.path.isdir(save_dir) and agent_config.save_on_creation:
            os.makedirs(save_dir)

        self.save_dir = save_dir
        self.has_app_state = False
        self.system_message = system_message
        self.agent_config_path = os.path.join(self.save_dir, "agent_config.json")
        self.app_state_path = os.path.join(save_dir, "app_state.json")
        self.semantic_memory_path = os.path.join(save_dir, "semantic_memory")
        if os.path.exists(save_dir) and os.path.exists(self.agent_config_path) and os.path.exists(self.app_state_path) and os.path.exists(self.semantic_memory_path):
            load = True

        self.app_state = None
        self.chat_history_index = 0
        self.chat_history = []
        self.tool_usage_history = {}
        if load:
            self.load_agent()
        else:
            if initial_state_file:
                self.has_app_state = True
                self.app_state = ContextAppState(initial_state_file)
            if agent_config.save_on_creation:
                self.save_agent()
        if self.use_semantic_memory:
            if not os.path.isdir(self.save_dir):
                os.makedirs(self.save_dir)
            agent_config.semantic_chat_history_config.persist_directory = self.semantic_memory_path
            self.semantic_memory = SemanticMemory(agent_config.semantic_chat_history_config)

        if self.tool_registry is not None and initial_state_file and self.give_agent_edit_tool:
            self.tool_registry.add_tools(self.app_state.get_edit_tools())

    def set_tool_registry(self, tool_registry: ToolRegistry):
        self.tool_registry = tool_registry
        if self.tool_registry is not None and self.give_agent_edit_tool:
            self.tool_registry.add_tools(self.app_state.get_edit_tools())

    def set_tools(self, tools: list[FunctionTool]):
        self.tool_registry = ToolRegistry()
        self.tool_registry.add_tools(tools)
        if self.tool_registry is not None and self.give_agent_edit_tool:
            self.tool_registry.add_tools(self.app_state.get_edit_tools())

    def remove_all_tools(self):
        self.tool_registry = None

    def chat_with_agent(self, chat_input: str, tool_registry: ToolRegistry = None, settings: LLMSamplingSettings = None):
        chat_history = self._before_run(chat_input)
        if tool_registry is not None and self.give_agent_edit_tool:
            tool_registry.add_tools(self.app_state.get_edit_tools())

        if self.tool_registry is None and tool_registry is None and self.give_agent_edit_tool:
            tool_registry = ToolRegistry()
            tool_registry.add_tools(self.app_state.get_edit_tools())

        result = self.agent.get_response(chat_history, self.tool_registry if tool_registry is None else tool_registry, settings=settings)
        self._after_run(chat_input)
        return result

    def stream_chat_with_agent(self, chat_input: str, tool_registry: ToolRegistry = None, settings: LLMSamplingSettings = None):
        chat_history = self._before_run(chat_input)
        if tool_registry is not None and self.give_agent_edit_tool:
            tool_registry.add_tools(self.app_state.get_edit_tools())

        if self.tool_registry is None and tool_registry is None and self.give_agent_edit_tool:
            tool_registry = ToolRegistry()
            tool_registry.add_tools(self.app_state.get_edit_tools())

        result = self.agent.get_streaming_response(chat_history, self.tool_registry if tool_registry is None else tool_registry, settings=settings)
        for tok in result:
            yield tok
        self._after_run(chat_input)

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
        self.tool_usage_history = loaded_data["tool_usage_history"]
        self.give_agent_edit_tool = loaded_data["give_agent_edit_tool"]
        if self.has_app_state:
            self.app_state = ContextAppState()
            self.app_state.load_json(self.app_state_path)

    def save_agent(self):
        if not os.path.isdir(self.save_dir):
            os.makedirs(self.save_dir)
        with open(self.agent_config_path, "w") as f:
            save_data = {
                "system_message": self.system_message,
                "has_app_state": self.has_app_state,
                "chat_history": self.chat_history,
                "chat_history_index": self.chat_history_index,
                "max_chat_history_length": self.max_chat_history_length,
                "use_semantic_memory": self.use_semantic_memory,
                "tool_usage_history": self.tool_usage_history,
                "give_agent_edit_tool": self.give_agent_edit_tool,
            }
            # noinspection PyTypeChecker
            json.dump(save_data, fp=f)
        if self.has_app_state:
            self.app_state.save_json(self.app_state_path)

    def add_to_chat_history(self, messages: list[dict[str, Any]]):
        self.chat_history.extend(messages)

    def add_to_chat_history_from_json(self, file_path: str):
        with open(file_path, "r") as f:
            loaded_data = json.load(fp=f)
        self.chat_history.extend(loaded_data)

    def _before_run(self, chat_input):
        user = chat_input
        if not self.has_app_state:
            chat_history = [{"role": "system", "content": self.system_message}]
        else:
            chat_history = [{"role": "system",
                             "content": self.system_message.format(app_state=self.app_state.get_app_state_string())}]
        chat_history.extend(self.chat_history[self.chat_history_index:])

        if self.use_semantic_memory:
            results = self.semantic_memory.recall(user, 3)
            if len(results) > 0:
                additional_context = "\n--- Additional Context From Past Interactions ---\n"
                for r in results:
                    additional_context += f"Memories: {r['content']}\n\n---\n\n"
                user += '\n' + additional_context.strip()

        chat_history.append({"role": "user", "content": user})
        return chat_history

    def _after_run(self, chat_input):
        # Add all messages to chat history
        self.tool_usage_history[len(self.chat_history)] = 1
        self.chat_history.append({"role": "user", "content": chat_input})
        self.tool_usage_history[len(self.chat_history)] = len(self.agent.last_messages_buffer)
        self.chat_history.extend(self.agent.last_messages_buffer)
        self.process_chat_history()

    def process_chat_history(self, max_chat_history_length: int = None):
        if max_chat_history_length is None:
            max_chat_history_length = self.max_chat_history_length
        if (len(self.chat_history) - self.chat_history_index) > max_chat_history_length and max_chat_history_length > -1:
            while (len(self.chat_history) - self.chat_history_index) > max_chat_history_length:
                msg_count = self.tool_usage_history.get(self.chat_history_index, 1)
                msg_count += self.tool_usage_history.get(self.chat_history_index + 1, 1)
                message = {}
                message2 = {}
                if self.use_semantic_memory and msg_count == 2:
                    message = self.chat_history[self.chat_history_index]
                    message2 = self.chat_history[self.chat_history_index + 1]
                elif self.use_semantic_memory and msg_count > 2:
                    message = self.chat_history[self.chat_history_index]
                    message2 = self.chat_history[self.chat_history_index + (msg_count - 1)]
                if self.use_semantic_memory and msg_count >= 2:
                    memory = f"<{message['role'].capitalize()}> {message['content']} </{message['role'].capitalize()}>\n"
                    memory += f"<{message2['role'].capitalize()}> {message2['content']} </{message2['role'].capitalize()}>"
                    self.semantic_memory.store(memory)
                self.chat_history_index += msg_count