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
{"role": "user",
     "content": "I've been thinking more about our discussion on relativity. You mentioned time dilation near a black hole—does that mean astronauts could use this to extend their lifespan?"},
    {"role": "assistant",
     "content": "Great follow-up! Yes, in theory, an astronaut orbiting near a black hole, in a region with extreme gravitational time dilation, would experience time much slower than people on Earth. This means they would effectively age slower, relative to those who remained on Earth. However, surviving near a black hole would be incredibly challenging due to extreme tidal forces and radiation."},

    {"role": "user",
     "content": "Right, that makes sense. You also mentioned that moving at high speeds can slow time. Would it be possible to build a spaceship that travels near the speed of light and allows astronauts to time travel into the future?"},
    {"role": "assistant",
     "content": "Yes, that's a real consequence of Special Relativity! If a spaceship traveled at 99% the speed of light, time for those on board would pass much slower compared to people on Earth. So, an astronaut on a high-speed journey could return to find that centuries have passed on Earth, even though they only experienced a few years."},

    {"role": "user",
     "content": "That sounds like something out of a sci-fi movie! But wouldn't accelerating to such speeds require a huge amount of energy?"},
    {"role": "assistant",
     "content": "Absolutely! According to relativity, as an object approaches the speed of light, its mass effectively increases, meaning it requires exponentially more energy to accelerate further. This makes it practically impossible to reach light speed with conventional technology. Scientists are exploring concepts like nuclear propulsion and antimatter engines, but for now, we're far from achieving such speeds."},

    {"role": "user",
     "content": "You also mentioned in our last discussion that traveling back in time would require 'exotic physics.' What kind of theories explore that?"},
    {"role": "assistant",
     "content": "Good memory! Some theoretical models suggest that time travel to the past might be possible under special conditions. One idea involves 'closed timelike curves,' predicted in solutions to Einstein's equations, like those involving rotating black holes (Kerr black holes) or wormholes. However, these require exotic matter with negative energy, which hasn’t been observed in nature."},

    {"role": "user",
     "content": "So if wormholes were real, could they connect different points in time as well as space?"},
    {"role": "assistant",
     "content": "Exactly! A traversable wormhole, if it existed, could theoretically link two distant locations in space and even different times. But stabilizing a wormhole would require negative energy, and even if one existed naturally, we have no idea how to control or navigate it. For now, it's more of a theoretical possibility than a practical one."},

    {"role": "user",
     "content": "Fascinating! I love how relativity keeps challenging our intuition about time and space."},
    {"role": "assistant",
     "content": "It really does! Einstein revolutionized our understanding of reality, showing that time isn’t absolute and space can bend. The universe is far stranger than we ever imagined!"},
    {"role": "user",
     "content": "I've been thinking about what you said last time—if time slows down near a black hole, does that mean someone could theoretically survive to see the end of the universe?"},
    {"role": "assistant",
     "content": "That’s a fascinating thought! In theory, yes—if someone stayed near the event horizon of a supermassive black hole without falling in, time for them would pass much slower than for the rest of the universe. Billions of years could pass outside while they only experience a fraction of that time. However, surviving that long near a black hole would be nearly impossible due to extreme radiation and tidal forces."},

    {"role": "user",
     "content": "Right, you mentioned tidal forces before. How exactly do they work near a black hole?"},
    {"role": "assistant",
     "content": "Great question! Tidal forces occur because gravity pulls more strongly on objects that are closer to a massive body. Near a black hole, this difference in gravitational pull can be extreme—if you were falling in feet-first, your feet would experience much stronger gravity than your head, stretching you into what scientists call 'spaghettification.'"},

    {"role": "user",
     "content": "Yikes! Sounds like falling into a black hole is a one-way trip. But what about those rotating black holes you mentioned last time? Could they allow for a different kind of experience?"},
    {"role": "assistant",
     "content": "Yes! Rotating black holes, also known as Kerr black holes, have a unique feature called an ergosphere, where space itself is dragged along with the rotation. Some theories suggest that if you enter a Kerr black hole at just the right angle, you might avoid the singularity and possibly emerge somewhere else—though whether that 'somewhere else' is another universe or just oblivion is unknown."},

    {"role": "user",
     "content": "That reminds me of wormholes. If a Kerr black hole could, in theory, let you escape, could it be related to how a wormhole might work?"},
    {"role": "assistant",
     "content": "You're making great connections! Some theories suggest that a rotating black hole could act as a natural wormhole, but we don’t have any evidence to confirm this. Wormholes require exotic matter to stay open, whereas a Kerr black hole still has an event horizon, meaning once you cross it, escape isn’t guaranteed."},

    {"role": "user",
     "content": "Speaking of wormholes, if one could connect two points in time, could that allow for communication with the past?"},
    {"role": "assistant",
     "content": "In theory, yes! If one end of a wormhole was moved at relativistic speeds and then brought back, it could be at a different time than the other end, potentially allowing messages to be sent back in time. However, maintaining a stable wormhole, let alone using it for communication, is purely speculative at this point."},

    {"role": "user",
     "content": "So if time travel to the past is at least theoretically possible, why do most physicists think it's unlikely?"},
    {"role": "assistant",
     "content": "One reason is causality—allowing backward time travel could lead to paradoxes, like the famous 'grandfather paradox,' where someone could alter their own past and create logical inconsistencies. Additionally, even theoretical models that permit time loops require conditions that may not exist in reality, like exotic matter with negative energy density."},

    {"role": "user",
     "content": "Ah, the grandfather paradox! But what about the idea that the universe might prevent paradoxes naturally, like in the 'self-consistency principle'?"},
    {"role": "assistant",
     "content": "Good point! The Novikov self-consistency principle suggests that even if time travel were possible, events would always unfold in a way that avoids paradoxes. For example, if you traveled back in time and tried to prevent your grandparents from meeting, something would always intervene to ensure history stays consistent. This concept appears in some physics models but remains untested."},

    {"role": "user", "content": "It's wild how relativity keeps leading to deeper and deeper questions!"},
    {"role": "assistant",
     "content": "Absolutely! Einstein’s work opened up a universe of strange possibilities, many of which we’re still exploring today. The more we learn, the more mysterious and beautiful the cosmos becomes!"}
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