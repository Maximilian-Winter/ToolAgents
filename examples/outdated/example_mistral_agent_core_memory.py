from ToolAgents import ToolRegistry
from ToolAgents.outdated.core_memory_manager import CoreMemoryManager
from ToolAgents.agents import MistralAgent
from ToolAgents.provider import LlamaCppServerProvider

provider = LlamaCppServerProvider("http://127.0.0.1:8080/")

agent = MistralAgent(provider=provider, debug_output=True)

settings = provider.get_default_settings()
settings.neutralize_all_samplers()
settings.temperature = 0.65

settings.set_max_new_tokens(4096)

messages = [{"role": "system", "content": "You are a personal AI assistant with a core memory system to keep track of information about you and the user. You core memory is currently empty, you have to use append_core_memory and replace_in_core_memory to manage your core memory. You have to use these tool explicit!"},
            {"role": "user", "content": "Hello! My name is Max Winter and my favorite color is deep purple! Save these information in your core memory."}]


core_memory = CoreMemoryManager(["Assistant", "User"], {"User": "", "Assistant": ""})

tools = core_memory.get_tools()

tool_registry = ToolRegistry()

tool_registry.add_tools(tools)

result = agent.get_streaming_response(
    messages=messages,
    settings=settings, tool_registry=tool_registry)
for tok in result:
    print(tok, end="", flush=True)
print()