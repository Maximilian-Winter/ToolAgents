from ToolAgents import ToolRegistry
from ToolAgents.agents import ChatToolAgent
from ToolAgents.messages import ChatHistory
from ToolAgents.messages.chat_message import ChatMessage, ChatMessageRole, TextContent
from ToolAgents.provider import OpenAIChatAPI, OpenAISettings

from example_tools import calculator_function_tool, current_datetime_function_tool, get_weather_function_tool

from dotenv import load_dotenv

load_dotenv()

# Local OpenAI like API, like vllm or llama-cpp-server
api = OpenAIChatAPI(api_key="token-abc123", base_url="http://127.0.0.1:8080/v1", model="unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit")
settings = OpenAISettings()

# Create the ChatAPIAgent
agent = ChatToolAgent(chat_api=api)

settings.temperature = 0.35
settings.top_p = 1.0

# Define the tools
tools = [calculator_function_tool, current_datetime_function_tool, get_weather_function_tool]
tool_registry = ToolRegistry()

tool_registry.add_tools(tools)
messages = [
    {
        "role": "system",
        "content": "You are Funky, an AI assistant specialized in interpreting user requests and generating appropriate function calls. Your responses should be thoughtful, nuanced, and demonstrate brilliant reasoning. Your primary task is to analyze user requests and, when necessary, create function calls in JSON format to perform the requested tasks.\nChoose the appropriate function based on the task you want to perform. Provide your function calls in JSON format. Here is an example for how your function calls should look like: [{\"name\": \"get_location_news\", \"arguments\": {\"location\": \"London, Great Britain\"}}, {\"name\": \"get_location_news\", \"arguments\": {\"location\": \"Berlin, Germany\"}}]"
    },
    {
        "role": "user",
        "content": "Perform all the following tasks: Get the current weather in celsius in the city of London, Great Britain, New York City, New York and at the North Pole, Arctica. Solve the following calculations: 42 * 42, 74 + 26, 7 * 26, 4 + 6  and 96/8. And retrieve the date and time in the format: %Y-%m-%d %H:%M:%S."
    }
]
chat_history = ChatHistory()

chat_history.add_messages(ChatMessage.from_dictionaries(messages))

response = agent.get_response(
    messages=chat_history.get_messages(),
    settings=settings, tool_registry=tool_registry)

print(response.content[0].content)

chat_history.add_messages(agent.last_messages_buffer)
chat_history.save_to_json("chat_history_example.json")

