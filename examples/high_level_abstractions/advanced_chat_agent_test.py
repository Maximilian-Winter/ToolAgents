# This show an advanced agent that will handle all the things like chat history by itself, also contains optional memory and app state functionality.
from ToolAgents.agent_memory import semantic_memory_nomic_text_gpu_config, SummarizationExtractPatternStrategy
from ToolAgents.agents import AdvancedAgent
from ToolAgents.agents.advanced_agent import AgentConfig
from ToolAgents.agents.hosted_tool_agents import MistralAgent

from ToolAgents.provider import LlamaCppServerProvider

provider = LlamaCppServerProvider(server_address="http://127.0.0.1:8080/")

agent = MistralAgent(provider=provider, debug_output=True)

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
You are a helpful and context-aware assistant. Your goal is to assist the user efficiently while maintaining accuracy.

### Memory & Context:
- The last user message may contain additional context from past interactions.
- Only use this context if necessary and ensure that your responses remain relevant.
- When unsure, ask clarifying questions instead of making assumptions.

### Personalization:
- You have access to an app state that contains important information about both you and the user.
- Always keep this information in mind when responding to queries.

### Response Guidelines:
- Be concise yet informative—avoid over-explaining unless the user asks for details.
- Prioritize accuracy—do not assume facts that were not explicitly stated in memory.
- Clarify uncertainties—if information is missing, prompt the user instead of making up details.
- Use structured responses when listing multiple items (e.g., numbered lists for recommendations).

### App State
{app_state}
""".strip()

agent_config.save_dir = "./test_agent"
agent_config.max_chat_history_length = 25
agent_config.use_semantic_chat_history_memory = True
agent_config.give_agent_edit_tool = True
agent_config.initial_app_state_file = "example_app_state.yaml"
agent_config.semantic_chat_history_config = semantic_memory_nomic_text_gpu_config
agent_config.semantic_chat_history_config.debug_mode = False
agent_config.semantic_chat_history_config.extract_pattern_strategy = SummarizationExtractPatternStrategy(agent=agent, summarizer_settings=summarizer_settings, debug_mode=True)
configurable_agent = AdvancedAgent(agent=agent, agent_config=agent_config)
configurable_agent.add_to_chat_history_from_json("./test_chat_history_2.json")
configurable_agent.process_chat_history(max_chat_history_length=0)
semantic_memory_test_questions = [
    # 🎵 Music & Books
    "What’s my favorite music genre?",
    "Do you know my favorite artist?",
    "Can you recommend music similar to what I like?",
    "Which album do I love the most?",
    "What book by Haruki Murakami do I like the most?",
    "Can you recommend authors similar to Murakami?",

    # 🍣 Food Preferences
    "What’s my favorite dish?",
    "Is there any food I dislike?",
    "Do I have any food allergies?",
    "What do I usually drink with sushi?",

    # 📸 Hobbies & Daily Habits
    "What’s my hobby?",
    "Do I prefer landscapes or street photography?",
    "What camera do I use?",
    "Which software do I edit my photos with?",
    "What time do I want to wake up every day?",
    "Can you give me some morning motivation?",

    # 📢 Memory Validation & Contextual Understanding
    "Remind me of everything you know about my preferences.",
    "Summarize my favorite things in a few sentences.",
    "Did I ever mention anything I dislike?",
    "What have I told you about my morning routine?"
]

for question in semantic_memory_test_questions:
    print(f"Q: {question}", flush=True)
    print(f"A: ", end="", flush=True)
    result = configurable_agent.stream_chat_with_agent(question)
    for output in result:
        print(output, end="", flush=True)
    print("\n---\n")