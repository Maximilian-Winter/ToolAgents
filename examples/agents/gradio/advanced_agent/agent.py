from ToolAgents.agents import ChatToolAgent
from ToolAgents.agent_memory import (
    semantic_memory_nomic_text_gpu_config,
    SummarizationExtractPatternStrategy,
)
from ToolAgents.agents.advanced_agent import AdvancedAgent
from ToolAgents.agents.advanced_agent import AgentConfig
from ToolAgents.provider import OpenAIChatAPI


from dotenv import load_dotenv

from prompts import (
    system_message,
    summarization_prompt_pairs,
    summarization_prompt_summaries,
)

load_dotenv()


api = OpenAIChatAPI(api_key="token-abc123", base_url="http://127.0.0.1:8080/v1", model="unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit")

# provider = AnthropicChatAPI(api_key=os.getenv("ANTHROPIC_API_KEY"), model="claude-3-5-sonnet-20241022")

# Create the ChatAPIAgent
# agent = ChatAPIAgent(chat_api=provider, debug_output=True)
agent = ChatToolAgent(chat_api=api)

settings = api.get_default_settings()
settings.neutralize_all_samplers()
settings.temperature = 0.4
settings.set_max_new_tokens(4096)

api.set_default_settings(settings)

summarizer_settings = api.get_default_settings()
summarizer_settings.neutralize_all_samplers()
summarizer_settings.temperature = 0.0
summarizer_settings.set_max_new_tokens(4096)

agent_config = AgentConfig()

agent_config.system_message = system_message

agent_config.save_dir = "./agent_data"
agent_config.max_chat_history_length = 30
agent_config.use_semantic_chat_history_memory = True
agent_config.give_agent_edit_tool = True
agent_config.initial_app_state_file = "initial_app_state.yaml"
agent_config.semantic_chat_history_config = semantic_memory_nomic_text_gpu_config
agent_config.semantic_chat_history_config.debug_mode = True
agent_config.summarize_chat_pairs_before_storing = True
agent_config.semantic_chat_history_config.extract_pattern_strategy = (
    SummarizationExtractPatternStrategy(
        user_name="Maximilian Winter",
        assistant_name="Jiutian Xuannü, the mysterious lady of the nine heavens",
        agent=agent,
        summarizer_settings=summarizer_settings,
        debug_mode=True
    )
)

configurable_agent = AdvancedAgent(
    agent=agent, agent_config=agent_config, user_name="User", assistant_name="Assistant"
)
configurable_agent.set_summarization_prompt(summarization_prompt_pairs)
# configurable_agent.add_to_chat_history_from_json("./abc_cleaned.json")
# configurable_agent.process_chat_history()
configurable_agent.save_agent()
