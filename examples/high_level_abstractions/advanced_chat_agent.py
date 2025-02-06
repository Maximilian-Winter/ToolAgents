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

agent_config.system_message = "You are an helpful assistant. The last user message will contain additional context based on past interactions, only refer to the additional context if necessary.\n\n\nThe following is your current app state which contains important information for about you and the user, always keep these things in mind!\n\n{app_state}"
agent_config.save_dir = "./example_agent"
agent_config.max_chat_history_length = 25
agent_config.use_semantic_chat_history_memory = True
agent_config.give_agent_edit_tool = True
agent_config.initial_app_state_file = "example_app_state.yaml"
agent_config.semantic_chat_history_config = semantic_memory_nomic_text_gpu_config
agent_config.semantic_chat_history_config.debug_mode = True
agent_config.semantic_chat_history_config.extract_pattern_strategy = SummarizationExtractPatternStrategy(agent=agent,summarizer_settings=summarizer_settings, debug_mode=True)

configurable_agent = AdvancedAgent(agent=agent, agent_config=agent_config)
configurable_agent.add_to_chat_history_from_json("./test_chat_history.json")
configurable_agent.process_chat_history(max_chat_history_length=0)

while True:
    user_input = input(">")
    if user_input == "exit":
        break
    elif user_input == "save":
        configurable_agent.save_agent()
    else:
        result = configurable_agent.stream_chat_with_agent(user_input)
        for output in result:
            print(output, end="", flush=True)
        print()