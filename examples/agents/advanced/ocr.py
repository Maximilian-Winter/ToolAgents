import json
from dotenv import load_dotenv

from ToolAgents.agents import ChatToolAgent
from ToolAgents.data_models.messages import ChatMessage
from ToolAgents.provider import OpenAIChatAPI

from ToolAgentsDev.examples.agents.advanced.ocr_prompts import PROMPT_MAPPING

load_dotenv()
api = OpenAIChatAPI(api_key="token-abc123", base_url="http://127.0.0.1:8080/v1", model="xxx")

# Create the ChatAPIAgent
agent = ChatToolAgent(chat_api=api)

# Create a samplings settings object
settings = api.get_default_settings()
settings.extra_body = { "min_p": 0.00 }

# Set sampling settings
settings.temperature = 0.3
settings.top_p = 1.0

messages = [
    ChatMessage.create_system_message("You are a helpful assistant.")
]

user_msg = ChatMessage.create_empty_user_message()
user_msg.add_image_file_data("daozang_01_book_page_7.png", "png")
user_msg.add_text(PROMPT_MAPPING["ocr_layout"])
messages.append(user_msg)

#chat_response = agent.get_response(messages=messages, settings=settings)

#print(chat_response.response)

result = agent.get_streaming_response(
    messages=messages, settings=settings
)

for res in result:
    if res.get_tool_results():
        print()
        print(f"Tool Use: {res.get_tool_name()}")
        print(f"Tool Arguments: {json.dumps(res.get_tool_arguments())}")
        print(f"Tool Result: {res.get_tool_results()}")
        print()
    print(res.chunk, end="", flush=True)