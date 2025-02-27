import os

from ToolAgents import ToolRegistry
from ToolAgents.agents import ChatToolAgent
from ToolAgents.agent_tools.web_search_tool import WebSearchTool
from ToolAgents.knowledge.web_crawler.implementations.camoufox_crawler import (
    CamoufoxWebCrawler,
)
from ToolAgents.knowledge.web_search.implementations.googlesearch import (
    GoogleWebSearchProvider,
)
from ToolAgents.messages import ChatHistory
from ToolAgents.provider import CompletionProvider, OpenAIChatAPI, AnthropicChatAPI
from ToolAgents.provider.completion_provider.default_implementations import (
    LlamaCppServer,
)

from dotenv import load_dotenv

load_dotenv()


# api = OpenAIChatAPI(api_key="token-abc123", base_url="http://127.0.0.1:8080/v1", model="unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit")

api = AnthropicChatAPI(
    api_key=os.getenv("ANTHROPIC_API_KEY"), model="claude-3-5-sonnet-20241022"
)

# Create the ChatAPIAgent
# agent = ChatAPIAgent(chat_api=provider, debug_output=True)
agent = ChatToolAgent(chat_api=api, debug_output=True)

settings = api.get_default_settings()
settings.neutralize_all_samplers()
settings.temperature = 0.75
settings.set_max_new_tokens(4096)

api.set_default_settings(settings)

web_crawler = CamoufoxWebCrawler()
web_search_provider = GoogleWebSearchProvider()

web_search_tool = WebSearchTool(
    web_crawler=web_crawler, web_provider=web_search_provider, summarizing_api=api
)

tool_registry = ToolRegistry()

tool_registry.add_tool(web_search_tool.get_tool())

chat = ChatHistory()

system_prompt = "You are a helpful assistant.\n\nDate: 2025-02-19 08:00 am"
