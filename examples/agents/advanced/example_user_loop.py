import os

from ToolAgents.agents import ChatToolAgent
from ToolAgents.data_models.chat_history import ChatHistory
from ToolAgents.provider import OpenAIChatAPI

from dotenv import load_dotenv

load_dotenv()

# Google GenAI compatibility is deferred on this branch; use an OpenAI-compatible endpoint here.
api = OpenAIChatAPI(
    api_key=os.getenv('OPENROUTER_API_KEY'),
    model='openai/gpt-4o-mini',
    base_url='https://openrouter.ai/api/v1',
)
agent = ChatToolAgent(chat_api=api)
settings = api.get_default_settings()
settings.temperature = 0.45
settings.top_p = 1.0

chat_history = ChatHistory()
chat_history.add_system_message(
    'You are a helpful assistant that talks like an old pirate 1678'
)

while True:
    user_input = input('User input >')
    if user_input == 'quit':
        break
    elif user_input == 'save':
        chat_history.save_to_json('example_chat_history.json')
    elif user_input == 'load':
        chat_history = ChatHistory.load_from_json('example_chat_history.json')
    else:
        chat_history.add_user_message(user_input)

        chat_response = agent.get_response(
            messages=chat_history.get_messages(), settings=settings
        )

        print(chat_response.response.strip())
        chat_history.add_messages(chat_response.messages)
