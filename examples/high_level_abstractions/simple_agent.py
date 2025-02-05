# This show an advanced agent that will handle all the things like chat history by itself.

from ToolAgents.agents import AdvancedAgent
from ToolAgents.agents.hosted_tool_agents import MistralAgent
from ToolAgents.provider import LlamaCppServerProvider

provider = LlamaCppServerProvider(server_address="http://127.0.0.1:8080/")

agent = MistralAgent(provider=provider)

settings = provider.get_default_settings()
settings.neutralize_all_samplers()
settings.temperature = 0.4
settings.set_max_new_tokens(4096)

provider.set_default_settings(settings)

advanced_agent = AdvancedAgent(agent=agent)

while True:
    user_input = input(">")
    if user_input == "exit":
        break
    result = advanced_agent.chat_with_agent(user_input)
    print(result)
