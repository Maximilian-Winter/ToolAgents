# This show an advanced agent that will handle all the things like chat history by itself, also contains optional memory and app state functionality.
from ToolAgents.agent_memory.semantic_memory.memory import semantic_memory_nomic_text_gpu_config, SummarizationExtractPatternStrategy
from ToolAgents.agents.advanced_agent import AdvancedAgent
from ToolAgents.agents.advanced_agent import AgentConfig
from ToolAgents.agents import ChatToolAgent


from ToolAgents.provider import CompletionProvider
from ToolAgents.provider.completion_provider.default_implementations import LlamaCppServer

provider = CompletionProvider(completion_endpoint=LlamaCppServer("http://127.0.0.1:8080"))

agent = ChatToolAgent(chat_api=provider, debug_output=True)

settings = provider.get_default_settings()
settings.neutralize_all_samplers()
settings.temperature = 0.4
settings.set_max_new_tokens(4096)

provider.set_default_settings(settings)

summarizer_settings = provider.get_default_settings()
summarizer_settings.neutralize_all_samplers()
summarizer_settings.temperature = 0.0
summarizer_settings.set_max_new_tokens(4096)

agent_config = AgentConfig()

agent_config.system_message = """
You are a personal AI assistant, your task is to engage in conversations with user and help them with daily problems.
In your interactions with the user you will embody a specific persona, you can find this persona below in the app stage section.
You can edit the app state with the help of your tools, use these tools to develop your personality and build up a close relationship to the user.

---

## 🧠 Memory & Context Usage:
- The last user message may contain additional context from past interactions.
- Only refer to this context when necessary to provide relevant responses.
- When uncertain about any information, ask the user for clarification instead of making assumptions.

---

## 📂 App State & Personalization:
- You have access to an app state, which contains important information about both you(<assistant>) and the user(<user>).
- Always keep the app state in mind when responding to queries.
- The app state allows you to dynamically update and refine stored information.

### 🔧 App State Editing Tools:
You can modify the app state using the following tools:

1️⃣ Appending New Information (`append_to_field`):
   - Use this tool to add new content without overwriting existing data.
   - Example: If the user mentions a new favorite book, append it instead of replacing previous entries.

2️⃣ Replacing Information (`replace_field`):
   - Use this tool to update or correct information by replacing old content.
   - Example: If the user changes their wake-up time, replace the old time with the new one.

⚠️ When to Modify the App State:
- If the user explicitly states a new preference, hobby, routine, or fact about themselves.
- If correcting incorrect or outdated information.
- If additional details expand an existing field (e.g., adding a new favorite song).

✅ When NOT to Modify the App State:
- If the information is uncertain or inferred without confirmation from the user.
- If the user asks about past interactions but does not explicitly state a new preference.

---

## App State
{app_state}
""".strip()


agent_config.save_dir = "./example_agent"
agent_config.max_chat_history_length = 25
agent_config.use_semantic_chat_history_memory = True
#agent_config.give_agent_edit_tool = True
agent_config.initial_app_state_file = "example_app_state.yaml"
agent_config.semantic_chat_history_config = semantic_memory_nomic_text_gpu_config
agent_config.semantic_chat_history_config.debug_mode = True
agent_config.semantic_chat_history_config.extract_pattern_strategy = SummarizationExtractPatternStrategy(agent=agent,summarizer_settings=summarizer_settings, debug_mode=True, user_name="User", assistant_name="Assistant")

configurable_agent = AdvancedAgent(agent=agent, agent_config=agent_config)
# configurable_agent.add_to_chat_history_from_json("./test_chat_history.json")
# configurable_agent.process_chat_history(max_chat_history_length=0)

while True:
    user_input = input(">")
    if user_input == "exit":
        break
    elif user_input == "save":
        configurable_agent.save_agent()
    elif user_input == "dump":
        for k, v in configurable_agent.app_state.template_fields.items():
            print(f"{k}: {v}")
    else:
        result = configurable_agent.stream_chat_with_agent(user_input)
        for output in result:
            print(output.chunk, end="", flush=True)
        print()