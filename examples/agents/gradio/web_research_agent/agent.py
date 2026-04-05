import os

from ToolAgents import ToolRegistry
from ToolAgents.agents import ChatToolAgent
from ToolAgents.agent_tools.web_search_tool import WebSearchTool
from ToolAgents.knowledge.web_crawler.implementations.camoufox_crawler import (
    CamoufoxWebCrawler,
)
from ToolAgents.knowledge.web_crawler.implementations.trafilatura import TrafilaturaWebCrawler
from ToolAgents.knowledge.web_search.implementations.duck_duck_go import DDGWebSearchProvider
from ToolAgents.knowledge.web_search.implementations.googlesearch import (
    GoogleWebSearchProvider,
)
from ToolAgents.data_models.chat_history import ChatHistory
from ToolAgents.provider import CompletionProvider, OpenAIChatAPI, AnthropicChatAPI
from ToolAgents.provider.completion_provider.default_implementations import (
    LlamaCppServer,
)

from dotenv import load_dotenv

load_dotenv()


# api = OpenAIChatAPI(api_key="token-abc123", base_url="http://127.0.0.1:8080/v1", model="unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit")
# Openrouter API
api = OpenAIChatAPI(
    api_key=os.getenv("OPENROUTER_API_KEY"),
    model="mistralai/ministral-8b-2512",
    base_url="https://openrouter.ai/api/v1",
)

# Create the ChatAPIAgent
# agent = ChatAPIAgent(chat_api=provider, log_output=True)
agent = ChatToolAgent(chat_api=api, log_output=True)

# Create a samplings settings object
settings = api.get_default_settings()

# Set sampling settings
settings.temperature = 0.3
settings.top_p = 0.95

api.set_default_settings(settings)

web_crawler = TrafilaturaWebCrawler()
web_search_provider = DDGWebSearchProvider()

web_search_tool = WebSearchTool(
    web_crawler=web_crawler, web_provider=web_search_provider, summarizing_api=api
)

tool_registry = ToolRegistry()

tool_registry.add_tool(web_search_tool.get_tool())

chat = ChatHistory()

system_prompt = "You are a helpful assistant.\n\nDate: 2026-04-01 08:00 am"

