from ToolAgents.agents import ChatToolAgent
from ToolAgents.provider import CompletionProvider
from ToolAgents.provider.completion_provider.default_implementations import LlamaCppServer

api = CompletionProvider(completion_endpoint=LlamaCppServer("http://127.0.0.1:8080"))

answer_agent = ChatToolAgent(chat_api=api, debug_output=True)

settings = api.get_default_settings()
settings.neutralize_all_samplers()
settings.temperature = 0.3

settings.set_max_new_tokens(4096)

api.set_default_settings(settings)
