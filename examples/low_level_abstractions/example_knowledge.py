import os

from ToolAgents import ToolRegistry
from ToolAgents.agents import ChatToolAgent
from ToolAgents.messages import ChatMessage

from ToolAgents.provider import OpenAIChatAPI, OpenAISettings, GroqChatAPI, GroqSettings, AnthropicChatAPI
from ToolAgents.knowledge.default_providers import TrafilaturaWebCrawler, DDGWebSearchProvider
from dotenv import load_dotenv

load_dotenv()

# Local OpenAI like API, like vllm or llama-cpp-server
# Groq API
api = AnthropicChatAPI(api_key=os.getenv("ANTHROPIC_API_KEY"), model="claude-3-5-sonnet-20241022")
settings = api.get_default_settings()
settings.temperature = 0.45
# Create the ChatAPIAgent
agent = ChatToolAgent(chat_api=api)
web_crawler = TrafilaturaWebCrawler()
web_search_provider = DDGWebSearchProvider()


tool_registry = ToolRegistry()

tool_registry.add_tools([web_crawler.get_tool(api=api), web_search_provider.get_tool()])


messages = [
    ChatMessage.create_system_message("You are a helpful assistant with tool calling capabilities. Only reply with a tool call if the function exists in the library provided by the user. Use JSON format to output your function calls. If it doesn't exist, just reply directly in natural language. When you receive a tool call response, use the output to format an answer to the original user question."),
    ChatMessage.create_user_message("Retrieve latest information about american foreign politics.")
]


chat_response = agent.get_response(
    messages=messages,
    settings=settings, tool_registry=tool_registry)

print(chat_response.response.strip())

print('\n'.join([msg.model_dump_json(indent=4) for msg in chat_response.messages]))