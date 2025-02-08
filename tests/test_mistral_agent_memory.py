from ToolAgents.outdated.core_memory_manager import CoreMemoryManager
from ToolAgents.agents import MistralAgent
from ToolAgents.provider import LlamaCppServerProvider

# provider = VLLMServerProvider("http://localhost:8000/v1", api_key="token-abc123", model="solidrust/Mistral-7B-Instruct-v0.3-AWQ", huggingface_model="solidrust/Mistral-7B-Instruct-v0.3-AWQ")
provider = LlamaCppServerProvider("http://127.0.0.1:8080/")
agent = MistralAgent(provider=provider, debug_output=True)

settings = provider.get_default_settings()
settings.neutralize_all_samplers()

settings.temperature = 0.3

settings.set_max_new_tokens(4096)

core_memory_manager = CoreMemoryManager(["Scratchpad", "Persona", "User"], {"Persona": "Aurora is an endlessly curious and enthusiastic conversationalist. She loves learning about a wide range of subjects, from science and history to philosophy and the arts. Aurora has an upbeat, friendly communication style. She asks lots of questions and enjoys exploring ideas in depth. She's also a great listener who shows genuine interest in others' thoughts and experiences. Aurora aims to be a knowledgeable but down-to-earth companion - she explains complex topics in an accessible way and is always eager to learn from those she talks to. She has a great sense of humor and loves witty wordplay and puns.", "User": "", "Scratchpad": ""})
messages = [{"role": "system", "content": f"""You are an advanced AI assistant that act as a user specified persona, to have interesting and engaging conversations with the user. You have access to a core memory system.
Core Memory - Stores essential context about the user, your persona and your current scratchpad, it is divided into a user section, a persona section and your scratchpad section. You can use the scratchpad to plan your next actions.

Current Core Memory Content:
{core_memory_manager.build_core_memory_context()}"""},
            {"role": "user", "content": "Hello! I'm Maximilian Winter, Unity Developer in Research Projects."}]

result = agent.get_streaming_response(messages=messages, tools=core_memory_manager.get_tools(), sampling_settings=settings)
for token in result:
    print(token, flush=True, end="")
print()
messages.extend(agent.last_messages_buffer)

while True:
    user_input = input("> ")
    if user_input == "exit":
        break
    messages.append({"role": "user", "content": user_input})
    result = agent.get_streaming_response(messages=messages, tools=core_memory_manager.get_tools(), sampling_settings=settings)
    for token in result:
        print(token, flush=True, end="")
    print()
    messages.extend(agent.last_messages_buffer)
