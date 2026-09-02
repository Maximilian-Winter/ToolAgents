import os
from pathlib import Path

from dotenv import load_dotenv

from ToolAgents.agents import ChatToolAgent
from ToolAgents.pipelines import Pipeline
from ToolAgents.provider.chat_api_provider.open_ai import OpenAIChatAPI

load_dotenv()

# Openrouter API
api = OpenAIChatAPI(
    api_key=os.getenv("OPENROUTER_API_KEY"),
    model="qwen/qwen3.5-9b",
    base_url="https://openrouter.ai/api/v1",
)
agent = ChatToolAgent(chat_api=api)


# Create a samplings settings object
settings = api.get_default_settings()

# Set sampling settings
settings.temperature = 0.3
settings.top_p = 0.9
api.set_default_settings(settings)

pipeline_path = Path(__file__).with_name("math_greeting_pipeline.json")
pipeline = Pipeline.load_from_json(pipeline_path, default_agent=agent)

results = pipeline.run_pipeline(operation="multiply", num1=5, num2=3, name="Alex")
print(results["greeting"])
