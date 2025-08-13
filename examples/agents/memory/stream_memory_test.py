from ToolAgents.agents import ChatToolAgent
from ToolAgents.data_models.messages import ChatMessage

from ToolAgents.provider import OpenAIChatAPI
from ToolAgents.agent_memory.stream.stream_memory import StreamMemory
from ToolAgents.knowledge.vector_database.implementations.sentence_transformer_embeddings import SentenceTransformerEmbeddingProvider


api = OpenAIChatAPI(api_key="token-abc123", base_url="http://127.0.0.1:8080/v1", model="Mistral-Small-3.2-24B-Instruct-2506")

agent = ChatToolAgent(chat_api=api)

settings = api.get_default_settings()
settings.neutralize_all_samplers()
settings.temperature = 0.4

settings.set_max_new_tokens(4096)

embedding_provider = SentenceTransformerEmbeddingProvider()

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

chat_history += [
  {"role": "user", "content": "One thing about sushi—I like it even better when I’m watching a Studio Ghibli movie."},
  {"role": "assistant", "content": "That’s a wonderful combo! Sushi and Studio Ghibli. Do you have a favorite Ghibli film?"},
  {"role": "user", "content": "Spirited Away is timeless, but ‘Whisper of the Heart’ hits differently when I’m feeling nostalgic."},
  {"role": "assistant", "content": "Noted! ‘Spirited Away’ for timeless moments, and ‘Whisper of the Heart’ when you’re in a nostalgic mood."},
  {"role": "user", "content": "By the way, I’m trying to avoid caffeine after 3PM now."},
  {"role": "assistant", "content": "Good to know! I’ll avoid recommending caffeinated drinks in the afternoon."},
  {"role": "user", "content": "I used to play piano, but now I mostly listen to jazz piano tracks."},
  {"role": "assistant", "content": "Got it! Jazz piano is your vibe now. I can find playlists for that."},
  {"role": "user", "content": "Actually, I sometimes miss playing. Do you think I should get back to it?"},
  {"role": "assistant", "content": "If it brought you joy, maybe easing back into it could be rewarding! Want me to help plan a short daily practice?"},
  {"role": "user", "content": "Maybe later. Right now, I'm more focused on learning Japanese."},
  {"role": "assistant", "content": "Nice! Learning Japanese ties well with your interest in Japanese culture and authors."},
  {"role": "user", "content": "Exactly. I’ve been watching Terrace House to help with listening skills."},
  {"role": "assistant", "content": "Great idea! Watching native content helps with immersion."},
  {"role": "user", "content": "Also, I tend to be a night owl, but I’m trying to shift that gradually."},
  {"role": "assistant", "content": "Noted! A gradual shift from night owl to early riser—it’s a common challenge."},
  {"role": "user", "content": "Sometimes I still stay up reading. That’s my weakness."},
  {"role": "assistant", "content": "A relatable one! Maybe evening reading rituals could be part of the new schedule."},
  {"role": "user", "content": "Also, I really dislike the sound of chewing. It distracts me a lot."},
  {"role": "assistant", "content": "Understood! I’ll keep that in mind when suggesting environments or habits."},
  {"role": "user", "content": "Oh, I forgot to say—I'm lactose intolerant, mildly."},
  {"role": "assistant", "content": "Thanks for the heads-up. I’ll avoid suggesting dairy-heavy dishes."},
  {"role": "user", "content": "But weirdly, I can tolerate some types of cheese, like aged cheddar."},
  {"role": "assistant", "content": "Interesting! Aged cheddar is okay, but generally cautious with dairy—got it."},
  {"role": "user", "content": "Can you remind me to take breaks during long study sessions?"}
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

user_input = "Do you remember my question about aging when traveling near a black hole? Because I have forgotten what I asked and what you said!"



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
]

hard_questions = [
    # Cross-domain
    "What kind of movies do I like to watch when eating sushi?",
    "What’s my favorite Ghibli film when I feel nostalgic?",
    "What kind of music do I enjoy that connects to my past?",
    "Have I mentioned playing an instrument before?",
    "Why am I watching Terrace House lately?",
    "Am I more of a morning person or a night owl?",
    "What habit am I trying to change about my daily rhythm?",
    "What’s something that distracts me a lot?",
    "What kind of dairy can I tolerate despite being lactose intolerant?",
    "Have I asked you to help me with my study habits?",
    "What contradictions exist between my ideal wake-up time and my current habits?",
    "Can you identify a pattern between my food, music, and media preferences?",
    "What cultural interests do I have that show up in multiple domains?",
    "What language am I trying to learn, and how am I doing it?",
    "Why might it be hard for me to wake up at 6:30 AM every day?",
    "How might my preference for reading at night conflict with other goals?",
    "Am I more introverted or extroverted based on what you’ve learned?",
    "What subtle food-related risks should be considered when planning meals for me?",
    "If I were planning a relaxing evening, what would be an ideal setup based on my preferences?",
    "Based on everything you know, how would you describe my ideal day?",
]
semantic_memory_test_questions += hard_questions


for question in semantic_memory_test_questions:
    print(f"Q: {question}", flush=True)
    print(f"A: ", end="", flush=True)

    results = stream_memory.recall(question, 6)
    additional_context = "\n--- Additional Context From Past Interactions ---\n"

    for r in results["memories"]:
        additional_context += f"Memories: {r['content']}\n\n---\n\n"

    question += "\n" + additional_context.strip()

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

