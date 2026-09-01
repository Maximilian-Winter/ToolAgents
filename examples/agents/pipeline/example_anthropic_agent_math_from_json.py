import os
from pathlib import Path

from dotenv import load_dotenv

from ToolAgents.agents import ChatToolAgent
from ToolAgents.pipelines import Pipeline
from ToolAgents.provider.chat_api_provider.anthropic import AnthropicChatAPI


load_dotenv()

api = AnthropicChatAPI(
    api_key=os.getenv("ANTHROPIC_API_KEY"),
    model="claude-3-5-sonnet-20241022",
)

agent = ChatToolAgent(chat_api=api)

settings = api.get_default_settings()
settings.neutralize_all_samplers()
settings.temperature = 0.3
settings.set_max_new_tokens(4096)
api.set_default_settings(settings)

pipeline_path = Path(__file__).with_name("math_greeting_pipeline.json")
pipeline = Pipeline.load_from_json(pipeline_path, default_agent=agent)

results = pipeline.run_pipeline(operation="multiply", num1=5, num2=3, name="Alex")
print(results["greeting"])
