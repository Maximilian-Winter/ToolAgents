import asyncio
import json
import os

from ToolAgents import ToolRegistry
from ToolAgents.agents.chat_tool_agent import AsyncChatToolAgent
from ToolAgents.messages.chat_message import ChatMessage
from ToolAgents.provider.chat_api_provider.anthropic import AsyncAnthropicChatAPI
from ToolAgents.provider.chat_api_provider.groq import AsyncGroqChatAPI
from ToolAgents.provider.chat_api_provider.mistral import AsyncMistralChatAPI
from ToolAgents.provider.chat_api_provider.open_ai import AsyncOpenAIChatAPI

from example_tools import calculator_function_tool, current_datetime_function_tool, get_weather_function_tool

from dotenv import load_dotenv

load_dotenv()

# Official OpenAI API
#api = AsyncOpenAIChatAPI(api_key=os.getenv("OPENAI_API_KEY"), model="gpt-4o-mini")

# Anthropic API
#api = AsyncAnthropicChatAPI(api_key=os.getenv("ANTHROPIC_API_KEY"), model="claude-3-5-sonnet-20241022")

# Groq API
#api = AsyncGroqChatAPI(api_key=os.getenv("GROQ_API_KEY"), model="llama-3.3-70b-versatile")

# Mistral API
api = AsyncMistralChatAPI(api_key=os.getenv("MISTRAL_API_KEY"), model="mistral-small-latest")

# Create the ChatAPIAgent
agent = AsyncChatToolAgent(chat_api=api)
# Create a samplings settings object
settings = api.get_default_settings()
settings.temperature = 0.45
settings.top_p = 1.0

# Define the tools
tools = [calculator_function_tool, current_datetime_function_tool, get_weather_function_tool]
tool_registry = ToolRegistry()

tool_registry.add_tools(tools)
messages = [
    ChatMessage.create_system_message("You are a helpful assistant with tool calling capabilities. Only reply with a tool call if the function exists in the library provided by the user. Use JSON format to output your function calls. If it doesn't exist, just reply directly in natural language. When you receive a tool call response, use the output to format an answer to the original user question."),
    ChatMessage.create_user_message("Get the weather in London and New York. Calculate 420 x 420 and retrieve the date and time in the format: %Y-%m-%d %H:%M:%S.")
]

async def main():
    result = agent.get_streaming_response(
        messages=messages,
        settings=settings, tool_registry=tool_registry)
    async for res in result:
        if res.get_tool_results():
            print()
            print(f"Tool Use: {res.get_tool_name()}")
            print(f"Tool Arguments: {json.dumps(res.get_tool_arguments())}")
            print(f"Tool Result: {res.get_tool_results()}")
            print()
        print(res.chunk, end='', flush=True)
    print(flush=True)

asyncio.run(main())