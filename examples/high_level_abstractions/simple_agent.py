# This show an advanced agent that will handle all the things like chat history by itself.

from ToolAgents.agents import AdvancedAgent
from ToolAgents.agents import ChatToolAgent
from ToolAgents.provider import OpenAIChatAPI

provider = OpenAIChatAPI(api_key="token-abc123", base_url="http://127.0.0.1:8080/v1", model="unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit")

agent = ChatToolAgent(chat_api=provider)

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
