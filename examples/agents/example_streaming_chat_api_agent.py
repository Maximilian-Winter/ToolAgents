import json
import os

from ToolAgents import ToolRegistry
from ToolAgents.agents import ChatToolAgent
from ToolAgents.messages.chat_message import ChatMessage
from ToolAgents.provider import AnthropicChatAPI, OpenAIChatAPI, GroqChatAPI, MistralChatAPI

from ToolAgents.provider.completion_provider.default_implementations import LlamaCppServer
from example_tools import calculator_function_tool, current_datetime_function_tool, get_weather_function_tool

from dotenv import load_dotenv

load_dotenv()

# Local OpenAI like API, like vllm or llama-cpp-server with --jinja flag for tool calling support without streaming.
#api = OpenAIChatAPI(api_key="token-abc123", base_url="http://127.0.0.1:8080/v1", model="unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit")

# Official OpenAI API
api = OpenAIChatAPI(api_key=os.getenv("OPENAI_API_KEY"), model="gpt-4o-mini")

# Anthropic API
api = AnthropicChatAPI(api_key=os.getenv("ANTHROPIC_API_KEY"), model="claude-3-5-sonnet-20241022")

# Groq API
api = GroqChatAPI(api_key=os.getenv("GROQ_API_KEY"), model="llama-3.3-70b-versatile")

# Llama Cpp Server Completion Based API without --jinja flag and with streaming, works best with --special flag
#api = CompletionProvider(completion_endpoint=LlamaCppServer("http://127.0.0.1:8080"))

# Mistral API
#api = MistralChatAPI(api_key=os.getenv("MISTRAL_API_KEY"), model="mistral-small-latest")

# Create the ChatAPIAgent
agent = ChatToolAgent(chat_api=api)
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

result = agent.get_streaming_response(
    messages=messages,
    settings=settings, tool_registry=tool_registry)

for res in result:
    if res.get_tool_results():
        print()
        print(f"Tool Use: {res.get_tool_name()}")
        print(f"Tool Arguments: {json.dumps(res.get_tool_arguments())}")
        print(f"Tool Result: {res.get_tool_results()}")
        print()
    print(res.chunk, end='', flush=True)
