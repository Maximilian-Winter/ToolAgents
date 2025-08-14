import os

from dotenv import load_dotenv

from ToolAgents.agents import ChatToolAgent
from ToolAgents.data_models.messages import ChatMessage

from ToolAgents.provider import OpenAIChatAPI
from ToolAgents.agent_memory.stream.stream_memory import StreamMemory
from ToolAgents.knowledge.vector_database.implementations.sentence_transformer_embeddings import SentenceTransformerEmbeddingProvider

load_dotenv()
api = OpenAIChatAPI(api_key=os.getenv("OPENROUTER_API_KEY"), base_url="https://openrouter.ai/api/v1", model="google/gemini-2.5-flash")

agent = ChatToolAgent(chat_api=api)

settings = api.get_default_settings()
settings.neutralize_all_samplers()
settings.temperature = 0.4

settings.set_max_new_tokens(4096)

embedding_provider = SentenceTransformerEmbeddingProvider("nomic-ai/nomic-embed-text-v1.5", trust_remote_code=True)

stream_memory = StreamMemory(embedding_provider=embedding_provider)

system_prompt = f"""You are are personal AI assistant. Your task is to engage in interesting conversations with the user. You have access to a memory system, which will remember information not in your current context. The most recent user message will contain additional context information from past interactions that are not part of the current context."""

chat_history = [
  {
    "role": "user",
    "content": "Hey, can you help me keep track of some personal preferences?"
  },
  {
    "role": "assistant",
    "content": "Of course! What kind of preferences would you like me to remember?"
  },
  {
    "role": "user",
    "content": "Let’s start with food. My favorite dish is sushi, especially salmon nigiri."
  },
  {
    "role": "assistant",
    "content": "Got it! You love sushi, and salmon nigiri is your favorite. Do you have any favorite drinks to go with it?"
  },
  {
    "role": "user",
    "content": "Yeah, I usually go for green tea or sometimes a cold matcha latte."
  },
  {
    "role": "assistant",
    "content": "Noted! Green tea and matcha latte are your go-to drinks. Would you like me to remember any foods you dislike?"
  },
  {
    "role": "user",
    "content": "Good idea! I really don’t like olives—too bitter for me."
  },
  {
    "role": "assistant",
    "content": "Understood! Avoid olives. Anything else I should keep in mind?"
  },
  {
    "role": "user",
    "content": "Yeah, I also have a peanut allergy, so I always check for that in ingredients."
  },
  {
    "role": "assistant",
    "content": "That’s important! I’ll make sure to remind you about peanuts when discussing food recommendations."
  },
  {
    "role": "user",
    "content": "Thanks! Now, let's talk about music. My favorite artist is Nujabes."
  },
  {
    "role": "assistant",
    "content": "Great choice! Nujabes’ blend of jazz and hip-hop is legendary. Do you have a favorite album or song?"
  },
  {
    "role": "user",
    "content": "Hard to pick, but I’d say ‘Modal Soul’ is my favorite album, and ‘Feather’ is a song I always come back to."
  },
  {
    "role": "assistant",
    "content": "Noted! ‘Modal Soul’ is your favorite album, and ‘Feather’ is a special track for you. Would you like me to recommend similar artists in the future?"
  },
  {
    "role": "user",
    "content": "That’d be great! I’m also into lo-fi beats and jazz fusion."
  },
  {
    "role": "assistant",
    "content": "Lo-fi beats and jazz fusion, got it! I can suggest artists like Uyama Hiroto, J Dilla, and Yussef Kamaal for a similar vibe."
  },
  {
    "role": "user",
    "content": "That’s perfect! Okay, what about books? I love Haruki Murakami’s writing."
  },
  {
    "role": "assistant",
    "content": "Murakami is a fantastic author! Do you have a favorite book of his?"
  },
  {
    "role": "user",
    "content": "Definitely. ‘Kafka on the Shore’ is my all-time favorite, but I also really liked ‘Norwegian Wood.’"
  },
  {
    "role": "assistant",
    "content": "Got it! ‘Kafka on the Shore’ is your favorite, with ‘Norwegian Wood’ as another favorite. I can recommend similar authors if you’d like!"
  },
  {
    "role": "user",
    "content": "Yeah, I'd love that!"
  },
  {
    "role": "assistant",
    "content": "You might enjoy authors like Yoko Ogawa, Kazuo Ishiguro, or Banana Yoshimoto—they have a similar surreal or introspective style."
  },
  {
    "role": "user",
    "content": "That sounds great! Now, let’s talk hobbies. I’ve been getting into photography lately."
  },
  {
    "role": "assistant",
    "content": "Photography, nice! Are you more into landscapes, portraits, or something else?"
  },
  {
    "role": "user",
    "content": "Mostly street photography. I love capturing candid moments in urban settings."
  },
  {
    "role": "assistant",
    "content": "That’s a great style! Do you have a favorite camera or editing software?"
  },
  {
    "role": "user",
    "content": "Yeah, I use a Fujifilm X100V, and I edit with Lightroom."
  },
  {
    "role": "assistant",
    "content": "Noted! You shoot with a Fujifilm X100V and edit in Lightroom. I can suggest photography tips and inspiration if you’d like!"
  },
  {
    "role": "user",
    "content": "I’d love that! Also, I’m trying to wake up earlier. Can you help me stay accountable?"
  },
  {
    "role": "assistant",
    "content": "Absolutely! What time are you aiming to wake up?"
  },
  {
    "role": "user",
    "content": "I want to wake up by 6:30 AM every day."
  },
  {
    "role": "assistant",
    "content": "Got it! 6:30 AM wake-up time. I can check in with you every morning to remind you. Would you like motivation tips too?"
  },
  {
    "role": "user",
    "content": "Yeah! Maybe some productivity quotes or morning routines?"
  },
  {
    "role": "assistant",
    "content": "Sounds good! I’ll send you reminders, along with occasional productivity tips and quotes to keep you motivated!"
  },
  {
    "role": "user",
    "content": "Thanks! That should be everything for now."
  },
  {
    "role": "assistant",
    "content": "You’re welcome! I’ll remember all this and assist you however I can."
  }
]


pair = []
for message in chat_history:
    pair.append(
        f"<{message['role'].capitalize()}> {message['content']} </{message['role'].capitalize()}>"
    )
    if len(pair) == 2:
        out = "".join(pair)
        pair = []
        stream_memory.process_input(out)


semantic_memory_test_questions = [
    # 🎵 Music & Books
    "What’s my favorite music genre?",
    "Do you know my favorite artist?",
    "Can you recommend music similar to what I like?",
    "Which album do I love the most?",
    "What book by Haruki Murakami do I like the most?",
    "Can you recommend authors similar to Murakami?",
    # 🍣 Food Preferences
    "What’s my favorite dish?",
    "Is there any food I dislike?",
    "Do I have any food allergies?",
    "What do I usually drink with sushi?",
    # 📸 Hobbies & Daily Habits
    "What’s my hobby?",
    "Do I prefer landscapes or street photography?",
    "What camera do I use?",
    "Which software do I edit my photos with?",
    "What time do I want to wake up every day?",
    "Can you give me some morning motivation?",
    # 📢 Memory Validation & Contextual Understanding
    "Remind me of everything you know about my preferences.",
    "Summarize my favorite things in a few sentences.",
    "Did I ever mention anything I dislike?",
    "What have I told you about my morning routine?",
    # Edge Cases
    "What are all my food-related preferences?",  # Should get sushi, drinks, olives, peanuts
    "Tell me about my creative interests",  # Should get photography and books
    "What should you remember about my health?",  # Should prioritize peanut allergy
    "List everything we discussed, organized by topic"  # Ultimate test

]

for question in semantic_memory_test_questions:
    print(f"User Input with Memory: ", end="", flush=True)


    results = stream_memory.recall(f"<User> {question} <User>", 10)
    additional_context = "\n\n--- Additional Context From Past Interactions ---\n\n"

    for r in results["memories"]:
        additional_context += f"Memory: {r['content']}\n\n"

    question += "\n" + additional_context.strip()

    print(question, flush=True)
    print(f"Assistant Output: ", end="", flush=True)

    result = agent.get_streaming_response(
        messages=[
            ChatMessage.create_system_message(system_prompt),
            ChatMessage.create_user_message(question),
        ],
        settings=settings,
    )

    for tok in result:
        print(tok.chunk, end="", flush=True)
    print("\n---\n")

