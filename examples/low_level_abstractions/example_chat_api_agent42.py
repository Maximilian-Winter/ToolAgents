import datetime
import json
import os

from ToolAgents.messages.chat_message import ChatMessage, ToolCallResultContent, ChatMessageRole
from ToolAgents.provider import AnthropicChatAPI, AnthropicSettings, OpenAIChatAPI, OpenAISettings
from ToolAgents.provider.chat_api_provider.mistral import MistralChatAPI, MistralSettings

from example_tools import calculator_function_tool, current_datetime_function_tool, get_weather_function_tool

from dotenv import load_dotenv

load_dotenv()

#api = OpenAIChatAPI(api_key=os.getenv("OPENAI_API_KEY"), base_url="https://api.openai.com/v1", model="gpt-4o-mini")
#settings = OpenAISettings()

#api = AnthropicChatAPI(api_key=os.getenv("ANTHROPIC_API_KEY"), model="claude-3-5-sonnet-20241022")
#settings = AnthropicSettings()

api = MistralChatAPI(api_key=os.getenv("MISTRAL_API_KEY"), model="mistral-small-latest")
settings = MistralSettings()

# Define the tools
tools = [calculator_function_tool, current_datetime_function_tool, get_weather_function_tool]

messages = [
  {
    "role": "system",
    "content": "You are Funky, an AI assistant specialized in interpreting user requests and generating appropriate function calls. Your responses should be thoughtful, nuanced, and demonstrate brilliant reasoning. Your primary task is to analyze user requests and, when necessary, create function calls in JSON format to perform the requested tasks.\nChoose the appropriate function based on the task you want to perform."
  },
  {
    "role": "user",
    "content": "Perform all the following tasks: Get the current weather in celsius in the city of London, Great Britain, New York City, New York and at the North Pole, Arctica. Solve the following calculations: 42 * 42, 74 + 26, 7 * 26, 4 + 6  and 96/8. And retrieve the date and time in the format: %Y-%m-%d %H:%M:%S."
  }
]
stream = api.get_response(messages=messages, settings=settings, tools=tools)

print(stream.model_dump_json(indent=4))


if stream is None:
  import datetime
  import json
  import os

  from ToolAgents.messages.chat_message import ChatMessage, ToolCallResultContent, ChatMessageRole
  from ToolAgents.provider import AnthropicChatAPI, AnthropicSettings, OpenAIChatAPI, OpenAISettings

  from example_tools import calculator_function_tool, current_datetime_function_tool, get_weather_function_tool

  from dotenv import load_dotenv

  load_dotenv()

  #api = OpenAIChatAPI(api_key=os.getenv("OPENAI_API_KEY"), base_url="https://api.openai.com/v1", model="gpt-4o-mini")
  #settings = OpenAISettings()

  api = AnthropicChatAPI(api_key=os.getenv("ANTHROPIC_API_KEY"), model="claude-3-5-sonnet-20241022")
  settings = AnthropicSettings()

  #api = MistralChatAPI(api_key=os.getenv("MISTRAL_API_KEY"), model="mistral-small-latest")
  #settings = MistralSettings()

  # Define the tools
  tools = [calculator_function_tool, current_datetime_function_tool, get_weather_function_tool]

  messages = [
    {
      "role": "system",
      "content": "You are Funky, an AI assistant specialized in interpreting user requests and generating appropriate function calls. Your primary task is to analyze user requests and, when necessary, create function calls in JSON format to perform the requested tasks.\nChoose the appropriate function based on the task you want to perform."
    },
    {
      "role": "user",
      "content": "Retrieve the date and time in the format: %Y-%m-%d %H:%M:%S."
    }
  ]
  stream = api.get_streaming_response(messages=messages, settings=settings, tools=tools)

  new_msg = None
  for chunk in stream:
      print(chunk.chunk, end="", flush=True)
      if chunk.get_finished_chat_message() is not None:
        print()
        new_msg = chunk.get_finished_chat_message()
        print(new_msg.model_dump_json(indent=4), flush=True)


  result_msg = ChatMessage(id=new_msg.id, content=[ToolCallResultContent(tool_call_result_id=new_msg.id, tool_call_id=new_msg.content[1].tool_call_id, tool_call_name=new_msg.content[1].tool_call_name, tool_call_result="2025-02-11 06:01:10")], created_at=datetime.datetime.now(), updated_at=datetime.datetime.now(), role=ChatMessageRole.Tool)
  conv_message = api.convert_chat_messages([new_msg,result_msg])
  messages.extend(conv_message)
  print(json.dumps(messages, indent=4))
  stream = api.get_streaming_response(messages=messages, settings=settings, tools=tools)
  for chunk in stream:
      print(chunk.chunk, end="", flush=True)
      if chunk.get_finished_chat_message() is not None:
        print()
        new_msg = chunk.get_finished_chat_message()
        print(new_msg.model_dump_json(indent=4), flush=True)