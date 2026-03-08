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

for temp in [0.1, 0.1, 0.1]:
    print(f"\n\nTemperature: {temp}")
    # Set sampling settings
    settings.temperature = temp
    settings.top_p = 1.0

    messages = [
        ChatMessage.create_system_message("You are an OCR expert for extracting Chinese text from images. Ensure that the extracted text is correct."),
    ]

    user_msg = ChatMessage.create_empty_user_message()
    user_msg.add_image_file_data("daozang_01_book_page_7.png", "png")
    user_msg.add_text("Extract the chinese text from the image.")
    messages.append(user_msg)

    result = agent.get_streaming_response(
        messages=messages, settings=settings
    )

    for res in result:
        print(res.chunk, end="", flush=True)