import dataclasses
import json
import os
from typing import Any

from ToolAgents import ToolRegistry, FunctionTool
from ToolAgents.agent_memory.context_app_state import ContextAppState
from ToolAgents.agent_memory.semantic_memory.memory import SemanticMemory, SemanticMemoryConfig, \
    SummarizationExtractPatternStrategy
from ToolAgents.interfaces import LLMSamplingSettings
from ToolAgents.interfaces.base_llm_agent import BaseToolAgent


@dataclasses.dataclass
class AgentConfig:
    """
    Configuration for initializing an agent.

    Attributes:
        system_message (str): The initial system prompt for the agent.
        save_dir (str): Directory where agent-related files will be saved.
        initial_app_state_file (str): Optional file path for the agent's initial application state.
        give_agent_edit_tool: Flag to determine if the agent should be given edit tools.
        max_chat_history_length (int): Maximum length of chat history to retain (-1 for unlimited).
        use_semantic_chat_history_memory (bool): Whether to use semantic memory for chat history.
        save_on_creation (bool): Whether to automatically save the agent upon creation.
        semantic_chat_history_config (SemanticMemoryConfig): Configuration for semantic memory.
    """
    system_message: str = "You are a helpful assistant."
    save_dir: str = "./default_agent"
    initial_app_state_file: str = None
    give_agent_edit_tool = False
    max_chat_history_length: int = -1
    use_semantic_chat_history_memory: bool = False
    save_on_creation: bool = False
    summarize_chat_pairs_before_storing: bool = False
    semantic_chat_history_config: SemanticMemoryConfig = dataclasses.field(default_factory=SemanticMemoryConfig)


class AdvancedAgent:
    """
    An advanced agent that leverages a base language model agent along with tools,
    chat history, and optional semantic memory to have more context-aware conversations.

    This class provides methods to load, save, and manage agent state, as well as
    to interact with the underlying language model.
    """

    def __init__(self, agent: BaseToolAgent, tool_registry: ToolRegistry = None, agent_config: AgentConfig = None, user_name: str = None, assistant_name: str = None):
        """
        Initialize the AdvancedAgent.

        Args:
            agent (BaseToolAgent): The underlying language model agent.
            tool_registry (ToolRegistry, optional): Registry of available tools.
            agent_config (AgentConfig, optional): Configuration parameters for the agent.
        """
        # Use default configuration if none provided
        if agent_config is None:
            agent_config = AgentConfig()

        self.tool_registry = tool_registry
        self.agent = agent
        self.summarization_prompt = None
        # Extract configuration parameters for easier access
        save_dir = agent_config.save_dir
        initial_state_file = agent_config.initial_app_state_file
        system_message = agent_config.system_message

        self.give_agent_edit_tool = agent_config.give_agent_edit_tool
        self.use_semantic_memory = agent_config.use_semantic_chat_history_memory
        self.max_chat_history_length = agent_config.max_chat_history_length
        self.summarize_chat_pairs_before_storing = agent_config.summarize_chat_pairs_before_storing
        if user_name is None:
            self.user_name = "User"
        else:
            self.user_name = user_name

        if assistant_name is None:
            self.assistant_name = "Assistant"
        else:
            self.assistant_name = assistant_name

        load = False  # Flag to indicate whether to load an existing agent state

        # Create the save directory if it doesn't exist and if we want to save on creation
        if not os.path.isdir(save_dir) and agent_config.save_on_creation:
            os.makedirs(save_dir)

        # Set file paths for configuration, app state, and semantic memory
        self.save_dir = save_dir
        self.has_app_state = False
        self.system_message = system_message
        self.agent_config_path = os.path.join(self.save_dir, "agent_config.json")
        self.app_state_path = os.path.join(save_dir, "app_state.json")
        self.semantic_memory_path = os.path.join(save_dir, "semantic_memory")

        # Determine if we should load an existing agent state based on file existence
        if os.path.exists(save_dir) and os.path.exists(self.agent_config_path):
            load = True

        # Initialize chat history and state tracking variables
        self.app_state = None
        self.chat_history_index = 0
        self.chat_history = []
        self.tool_usage_history = {}

        # Load existing agent state if available
        if load:
            self.load_agent()
        else:
            # Otherwise, initialize app state from a provided file (if any)
            if initial_state_file:
                self.has_app_state = True
                self.app_state = ContextAppState(initial_state_file)
            # Save the agent state if configured to do so on creation
            if agent_config.save_on_creation:
                self.save_agent()

        # Initialize semantic memory if enabled
        if self.use_semantic_memory:
            if not os.path.isdir(self.save_dir):
                os.makedirs(self.save_dir)
            # Set the directory where semantic memory will persist
            agent_config.semantic_chat_history_config.persist_directory = self.semantic_memory_path
            self.semantic_memory = SemanticMemory(agent_config.semantic_chat_history_config)

        # If a tool registry and initial app state exist, and edit tools are enabled,
        # add the edit tools to the tool registry.
        if self.tool_registry is not None and initial_state_file and self.give_agent_edit_tool:
            self.tool_registry.add_tools(self.app_state.get_edit_tools())

    def set_tool_registry(self, tool_registry: ToolRegistry):
        """
        Set the tool registry for the agent.

        Args:
            tool_registry (ToolRegistry): The new tool registry to be used.
        """
        self.tool_registry = tool_registry
        # If edit tools are enabled, add them to the registry
        if self.tool_registry is not None and self.give_agent_edit_tool:
            self.tool_registry.add_tools(self.app_state.get_edit_tools())

    def set_tools(self, tools: list[FunctionTool]):
        """
        Initialize and set a list of tools for the agent.

        Args:
            tools (list[FunctionTool]): A list of function tools to add.
        """
        self.tool_registry = ToolRegistry()
        self.tool_registry.add_tools(tools)
        # Add edit tools if enabled
        if self.tool_registry is not None and self.give_agent_edit_tool:
            self.tool_registry.add_tools(self.app_state.get_edit_tools())

    def remove_all_tools(self):
        """
        Remove all tools from the agent by clearing the tool registry.
        """
        self.tool_registry = None

    def chat_with_agent(self, chat_input: str, tool_registry: ToolRegistry = None,
                        settings: LLMSamplingSettings = None):
        """
        Have a conversation with the agent using a given input.

        Args:
            chat_input (str): The user's chat input.
            tool_registry (ToolRegistry, optional): A tool registry to use for this conversation.
            settings (LLMSamplingSettings, optional): Sampling settings for the language model.

        Returns:
            The agent's response as a string.
        """
        # Prepare chat history with context and previous interactions
        chat_history = self._before_run(chat_input)

        # If an external tool registry is provided and edit tools are enabled,
        # add the agent's edit tools to it.
        if tool_registry is not None and self.give_agent_edit_tool:
            tool_registry.add_tools(self.app_state.get_edit_tools())

        # If no tool registry is provided but edit tools are required,
        # create a new one and add the edit tools.
        if self.tool_registry is None and tool_registry is None and self.give_agent_edit_tool:
            tool_registry = ToolRegistry()
            tool_registry.add_tools(self.app_state.get_edit_tools())

        # Get the agent's response based on the prepared chat history and the tool registry.
        result = self.agent.get_response(
            chat_history,
            self.tool_registry if tool_registry is None else tool_registry,
            settings=settings
        )
        # Update chat history and state after the conversation
        self._after_run(chat_input)
        return result

    def stream_chat_with_agent(self, chat_input: str, tool_registry: ToolRegistry = None,
                               settings: LLMSamplingSettings = None):
        """
        Stream a conversation with the agent, yielding tokens as they become available.

        Args:
            chat_input (str): The user's chat input.
            tool_registry (ToolRegistry, optional): A tool registry to use.
            settings (LLMSamplingSettings, optional): Sampling settings for the language model.

        Yields:
            Tokens (str) of the agent's streaming response.
        """
        # Prepare the chat history similarly to the non-streaming method
        chat_history = self._before_run(chat_input)

        if tool_registry is not None and self.give_agent_edit_tool:
            tool_registry.add_tools(self.app_state.get_edit_tools())

        if self.tool_registry is None and tool_registry is None and self.give_agent_edit_tool:
            tool_registry = ToolRegistry()
            tool_registry.add_tools(self.app_state.get_edit_tools())

        # Get the streaming response from the agent
        result = self.agent.get_streaming_response(
            chat_history,
            self.tool_registry if tool_registry is None else tool_registry,
            settings=settings
        )
        # Yield each token as it arrives
        for tok in result:
            yield tok
        # Update state after the streaming conversation is complete
        self._after_run(chat_input)

    def load_agent(self):
        """
        Load the agent's state from disk, including configuration and chat history.
        """
        if not os.path.isfile(self.agent_config_path):
            return
        # Open the configuration file and load JSON data
        with open(self.agent_config_path, "r") as f:
            loaded_data = json.load(fp=f)
        # Restore the agent's configuration and chat history
        self.system_message = loaded_data["system_message"]
        self.has_app_state = loaded_data["has_app_state"]
        self.chat_history = loaded_data["chat_history"]
        self.chat_history_index = loaded_data["chat_history_index"]
        self.max_chat_history_length = loaded_data["max_chat_history_length"]
        self.use_semantic_memory = loaded_data["use_semantic_memory"]
        self.tool_usage_history = loaded_data["tool_usage_history"]
        self.give_agent_edit_tool = loaded_data["give_agent_edit_tool"]
        self.assistant_name = loaded_data["assistant_name"]
        self.user_name = loaded_data["user_name"]
        # If an application state is maintained, load it from the corresponding file
        if self.has_app_state:
            self.app_state = ContextAppState()
            self.app_state.load_json(self.app_state_path)

    def save_agent(self):
        """
        Save the agent's current state and configuration to disk.
        """
        # Ensure the save directory exists
        if not os.path.isdir(self.save_dir):
            os.makedirs(self.save_dir)
        # Write the agent configuration and chat history to a JSON file
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
                "assistant_name": self.assistant_name,
                "user_name": self.user_name,
            }
            json.dump(save_data, fp=f)
        # If the agent has an application state, save it separately
        if self.has_app_state:
            self.app_state.save_json(self.app_state_path)

    def get_current_chat_history(self):
        if not self.has_app_state:
            chat_history = [{"role": "system", "content": self.system_message}]
        else:
            # Format system message with app state if needed
            chat_history = [{"role": "system",
                             "content": self.system_message.format(app_state=self.app_state.get_app_state_string())}]
        # Add any existing chat history beyond the current index
        chat_history.extend(self.chat_history[self.chat_history_index:])
        return chat_history

    def add_to_chat_history(self, messages: list[dict[str, Any]]):
        """
        Append a list of messages to the chat history.

        Args:
            messages (list[dict[str, Any]]): List of message dictionaries to add.
        """
        self.chat_history.extend(messages)

    def add_to_chat_history_from_json(self, file_path: str):
        """
        Load additional chat history from a JSON file and append it.

        Args:
            file_path (str): The file path to the JSON file containing chat messages.
        """
        with open(file_path, "r") as f:
            loaded_data = json.load(fp=f)
        self.chat_history.extend(loaded_data)

    def set_summarization_prompt(self, prompt: tuple[str, str]):
        self.summarization_prompt = prompt

    def _before_run(self, chat_input: str):
        """
        Prepare the chat history before sending a new message to the agent.

        This method constructs the chat history by combining the system message,
        previous messages, and optionally additional semantic memory context.

        Args:
            chat_input (str): The incoming user message.

        Returns:
            list: A list of message dictionaries representing the current conversation context.
        """
        user = chat_input
        # Start with a system message; include app state if available
        if not self.has_app_state:
            chat_history = [{"role": "system", "content": self.system_message}]
        else:
            # Format system message with app state if needed
            chat_history = [{"role": "system",
                             "content": self.system_message.format(app_state=self.app_state.get_app_state_string())}]
        # Add any existing chat history beyond the current index
        chat_history.extend(self.chat_history[self.chat_history_index:])

        # If semantic memory is enabled, retrieve additional context and append it to the user input
        if self.use_semantic_memory:
            results = self.semantic_memory.recall(user, 3)
            if len(results) > 0:
                additional_context = "\n--- Additional Context From Past Interactions ---\n"
                for r in results:
                    additional_context += f"Memories: {r['content']}\n\n---\n\n"
                # Append the additional context to the user input
                user += '\n' + additional_context.strip()

        # Append the (possibly enriched) user message to the chat history
        chat_history.append({"role": "user", "content": user})
        return chat_history

    def _after_run(self, chat_input: str):
        """
        Update the chat history and usage tracking after an agent run.

        This method updates the history with the new user message and the agent's response,
        then triggers chat history processing (e.g., for memory consolidation).

        Args:
            chat_input (str): The original user input that was sent to the agent.
        """
        # Record tool usage for the current position in chat history
        self.tool_usage_history[len(self.chat_history)] = 1
        # Append the user's message to the chat history
        self.chat_history.append({"role": "user", "content": chat_input})
        # Record the length of the agent's response buffer as tool usage
        self.tool_usage_history[len(self.chat_history)] = len(self.agent.last_messages_buffer)
        # Append the agent's response messages to the chat history
        self.chat_history.extend(self.agent.last_messages_buffer)
        # Process the chat history to enforce any maximum length limits
        self.process_chat_history()

    def process_chat_history(self, max_chat_history_length: int = None):
        """
        Process the chat history to ensure it doesn't exceed the maximum allowed length.

        If the history exceeds the maximum length, older messages may be consolidated
        and stored in semantic memory.

        Args:
            max_chat_history_length (int, optional): The maximum number of chat messages to retain.
                Defaults to the instance's max_chat_history_length.
            summarize_with_agent (bool, optional): If True, the message pairs get summarized by the agent before storing them.
        """
        if max_chat_history_length is None:
            max_chat_history_length = self.max_chat_history_length
        # Only process if the chat history length exceeds the allowed limit and the limit is set (not -1)
        if (
                len(self.chat_history) - self.chat_history_index) > max_chat_history_length and max_chat_history_length > -1:
            while (len(self.chat_history) - self.chat_history_index) > max_chat_history_length:
                # Determine how many messages are associated with the current chat block
                msg_count = self.tool_usage_history.get(self.chat_history_index, 1)
                msg_count += self.tool_usage_history.get(self.chat_history_index + 1, 1)
                message = {}
                message2 = {}
                # If semantic memory is enabled, select messages to be consolidated
                if self.use_semantic_memory and msg_count == 2:
                    message = self.chat_history[self.chat_history_index]
                    message2 = self.chat_history[self.chat_history_index + 1]
                elif self.use_semantic_memory and msg_count > 2:
                    message = self.chat_history[self.chat_history_index]
                    message2 = self.chat_history[self.chat_history_index + (msg_count - 1)]
                # If there are at least two messages to consolidate, build a memory string and store it
                if self.use_semantic_memory and msg_count >= 2:
                    memory = f"<{self.user_name}> {message['content']} </{self.user_name}>\n"
                    memory += f"<{self.assistant_name}> {message2['content']} </{self.assistant_name}>"

                    if self.summarize_chat_pairs_before_storing:
                        if self.summarization_prompt is None:
                            prompt = SummarizationExtractPatternStrategy.get_dynamic_prompt("chat")
                        else:
                            prompt = self.summarization_prompt
                        summarization_history = [{"role": "system", "content": prompt[0]},
                                                 {"role": "user", "content": prompt[1] + memory}]
                        settings = self.agent.get_default_settings()
                        settings.neutralize_all_samplers()
                        settings.temperature = 0.0
                        memory = self.agent.get_response(summarization_history, settings=settings)
                        print(memory)
                    self.semantic_memory.store(memory)
                # Move the chat history index forward by the number of messages consolidated
                self.chat_history_index += msg_count
