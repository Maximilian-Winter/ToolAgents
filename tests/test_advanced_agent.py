# This show an advanced agent that will handle all the things like chat history by itself, also contains optional memory and app state functionality.

from ToolAgents.agents import AdvancedAgent
from ToolAgents.agents.advanced_agent import AgentConfig
from ToolAgents.agents.hosted_tool_agents import MistralAgent

from ToolAgents.provider import LlamaCppServerProvider

provider = LlamaCppServerProvider("http://127.0.0.1:8080/")

agent = MistralAgent(provider=provider)

settings = provider.get_default_settings()
settings.neutralize_all_samplers()
settings.temperature = 0.4
settings.set_max_new_tokens(4096)

provider.set_default_settings(settings)

agent_config = AgentConfig()

agent_config.system_message = "You are an helpful assistant.\n\n\nThe following is your current app state which contains important information for the chat, always keep these things in mind!\n\n{app_state}"
agent_config.save_dir = "./example_agent"
agent_config.max_chat_history_length = 10
agent_config.use_semantic_chat_history_memory = True
agent_config.initial_app_state_file = "example_app_state.yaml"

configurable_agent = AdvancedAgent(agent=agent, agent_config=agent_config)

result = configurable_agent.chat_with_agent("Hello! Tell me about you!")
print(result)

result = configurable_agent.chat_with_agent("Hello! Tell me about you!")
print(result)