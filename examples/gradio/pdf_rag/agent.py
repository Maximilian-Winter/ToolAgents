from ToolAgents.agents import MistralAgent
from ToolAgents.provider import LlamaCppServerProvider

provider = LlamaCppServerProvider("http://127.0.0.1:8080/")

answer_agent = MistralAgent(provider=provider, debug_output=True)

settings = provider.get_default_settings()
settings.neutralize_all_samplers()
settings.temperature = 0.3

settings.set_max_new_tokens(4096)

provider.set_default_settings(settings)
