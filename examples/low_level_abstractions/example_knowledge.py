import os

from ToolAgents.agents import ChatToolAgent

from ToolAgents.provider import OpenAIChatAPI, OpenAISettings, GroqChatAPI, GroqSettings
from ToolAgents.knowledge.summarizer import summarize_docs
from dotenv import load_dotenv

load_dotenv()

# Local OpenAI like API, like vllm or llama-cpp-server
# Groq API
api = OpenAIChatAPI(api_key=os.getenv("OPENROUTER_API_KEY"), model="google/gemini-2.0-pro-exp-02-05:free", base_url="https://openrouter.ai/api/v1")
settings = api.get_default_settings()
settings.temperature = 0.45
# Create the ChatAPIAgent
agent = ChatToolAgent(chat_api=api)


summarize_docs(agent, settings, docs)