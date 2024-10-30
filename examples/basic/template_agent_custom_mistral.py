from ToolAgents import ToolRegistry
from ToolAgents.agents.hosted_tool_agents import TemplateAgent
from ToolAgents.provider import LlamaCppServerProvider
from ToolAgents.utilities.chat_history import AdvancedChatFormatter, ChatHistory
from example_tools import get_weather_function_tool, calculator_function_tool, current_datetime_function_tool

provider = LlamaCppServerProvider("http://127.0.0.1:8080/")

system_template = "<system_instructions>\n{content}\n</system_instructions>\n\n"
assistant_template = "{content}</s>"
user_template = "[INST] {content} [/INST]"

advanced_chat_formatter = AdvancedChatFormatter({
    "system": system_template,
    "user": user_template,
    "assistant": assistant_template,
}, include_system_message_in_first_user_message=True)

# agent = MistralAgent(provider=provider, debug_output=True)
agent = TemplateAgent(provider, advanced_chat_formatter=advanced_chat_formatter, generation_prompt=None,
                      debug_output=True)

settings = provider.get_default_settings()
settings.neutralize_all_samplers()
settings.temperature = 0.1
settings.set_stop_tokens(["</s>"], None)
settings.set_max_new_tokens(4096)

tools = [calculator_function_tool, current_datetime_function_tool, get_weather_function_tool]

chat_history = ChatHistory()
chat_history.load_history("./test_tools_chat_history.json")

tool_registry = ToolRegistry()

tool_registry.add_tools(tools)

result = agent.get_streaming_response(
    messages=chat_history.to_list(),
    settings=settings, tool_registry=tool_registry)
for tok in result:
    print(tok, end="", flush=True)
print()

chat_history.add_list_of_dicts(agent.last_messages_buffer)
chat_history.save_history("./test_chat_history_after_template_agent.json")
