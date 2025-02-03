from ToolAgents.agent_memory.context_app_state import ContextAppState
from ToolAgents import ToolRegistry
from ToolAgents.agent_memory.semantic_memory.memory import SemanticMemory
from ToolAgents.agents.hosted_tool_agents import TemplateAgent, AdvancedChatFormatter, MistralAgent
from ToolAgents.interfaces.llm_tool_call import TemplateToolCallHandler
from ToolAgents.provider import LlamaCppServerProvider
from ToolAgents.utilities import ChatHistory

provider = LlamaCppServerProvider("http://127.0.0.1:8080/")

agent = MistralAgent(provider=provider, debug_output=True)

settings = provider.get_default_settings()
settings.neutralize_all_samplers()
settings.temperature = 0.4
settings.set_stop_tokens(["</s>", "<|im_end|>"], None)
settings.set_max_new_tokens(4096)

pair = []
pair_list = []

memory = SemanticMemory("./test_semantic_memory")

system_prompt = f"""You are are personal AI assistant. Your task is to engage in interesting conversations with the user. You have access to a memory system, which will remember information not in your current context. The most recent user message will contain additional context information from past interactions that are not part of the current context."""

chat_history = [
    {"role": "user", "content": "Can you explain Einstein's theory of relativity in simple terms?"},
    {"role": "assistant",
     "content": "Of course! Einstein's theory of relativity consists of two main parts: Special Relativity and General Relativity. Special Relativity states that the speed of light is always constant, no matter how fast you're moving, leading to time dilation and length contraction. General Relativity extends this idea to gravity, describing it as the warping of spacetime by massive objects."},

    {"role": "user", "content": "That sounds fascinating! What do you mean by 'warping of spacetime'?"},
    {"role": "assistant",
     "content": "Great question! Imagine spacetime as a flexible sheet, like a trampoline. If you place a heavy object, like a bowling ball, in the middle, it creates a dent. Smaller objects, like marbles, will roll toward it, not because they're pulled, but because the surface is curved. This is how massive objects like planets and stars bend spacetime, creating what we perceive as gravity."},

    {"role": "user", "content": "So is that why time runs slower near a black hole?"},
    {"role": "assistant",
     "content": "Exactly! The stronger the gravitational field, the more spacetime bends, and this includes time itself. Near a black hole, gravity is so strong that time slows down significantly compared to someone far away. This effect is called gravitational time dilation and has been confirmed through experiments, like observing time differences in clocks placed at different altitudes on Earth."},

    {"role": "user",
     "content": "Does that mean if I traveled near a black hole and came back, I would age slower than people on Earth?"},
    {"role": "assistant",
     "content": "Yes, that's a real possibility! If you orbited close to a black hole for a while and then returned to Earth, you would have aged less than those who stayed here. This idea was beautifully illustrated in the movie *Interstellar*, where one hour on a planet near a black hole equaled seven years on Earth."},

    {"role": "user", "content": "That’s mind-blowing! Does this mean time travel to the future is possible?"},
    {"role": "assistant",
     "content": "In a way, yes! Since time slows down in strong gravitational fields or at high speeds (as predicted by relativity), traveling near light speed or staying near a black hole could allow you to move into the future relative to others. However, traveling back in time is a whole other challenge, requiring exotic physics that we haven't proven yet!"},
]

for message in chat_history:
    pair.append(f"<{message['role'].capitalize()}> {message['content']} </{message['role'].capitalize()}>")
    if len(pair) == 2:
        out = ''.join(pair)
        pair_list.append(out)
        pair = []
        memory.store(out)

user_input = "Do you remember my question about aging when traveling near a black hole? Because I have forgotten what I asked and what you said!"

results = memory.recall(user_input, 3)

additional_context = "\n--- Additional Context From Past Interactions ---\n"
for r in results:
    additional_context += f"Memories: {r['content']}\n\n---\n\n"


user_input += '\n' + additional_context.strip()

result = agent.get_streaming_response(
    messages=[
    {"role": "system", "content": system_prompt},
    {"role": "user", "content": user_input}
  ],
    settings=settings)
for tok in result:
    print(tok, end="", flush=True)
print()