---
title: Working with Chat History
---

# Working with Chat History

Maintaining chat history is essential for building coherent conversations with LLM agents. This guide covers the ChatHistory class and related functionality in ToolAgents.

## The ChatHistory Class

ToolAgents provides a `ChatHistory` class that simplifies managing conversation messages:

```python
from ToolAgents.messages import ChatHistory
from ToolAgents.messages.chat_message import ChatMessage

# Create a new chat history
chat_history = ChatHistory()

# Add a system message
chat_history.add_system_message(
    "You are a helpful assistant with tool calling capabilities."
)

# Add a user message
chat_history.add_user_message("Can you help me find the weather in San Francisco?")

# Get all messages as a list
messages = chat_history.get_messages()
```

## Adding Messages

You can add different types of messages to the chat history:

```python
# Add a system message
chat_history.add_system_message("Additional system instructions...")

# Add a user message
chat_history.add_user_message("What's the weather like today?")

# Add an assistant message
chat_history.add_assistant_message("I'll check the weather for you.")

# Add a tool result message
chat_history.add_tool_message("get_weather", {"temperature": 22, "condition": "sunny"})

# Add multiple messages at once
additional_messages = [
    ChatMessage.create_user_message("Another question"),
    ChatMessage.create_assistant_message("Another answer")
]
chat_history.add_messages(additional_messages)
```

## Using Chat History with Agents

Here's how to use chat history with a `ChatToolAgent`:

```python
from ToolAgents.agents import ChatToolAgent
from ToolAgents.provider import OpenAIChatAPI
from ToolAgents import ToolRegistry

# Set up the API and agent
api = OpenAIChatAPI(api_key="your-api-key", model="gpt-4o-mini")
agent = ChatToolAgent(chat_api=api)
settings = api.get_default_settings()

# Set up tools
tool_registry = ToolRegistry()
tool_registry.add_tools([your_tool1, your_tool2])

# Initialize chat history
chat_history = ChatHistory()
chat_history.add_system_message(
    "You are a helpful assistant with access to tools."
)

# User interaction loop
while True:
    user_input = input("User: ")
    if user_input.lower() == "exit":
        break
        
    # Add user message to history
    chat_history.add_user_message(user_input)
    
    # Get response from agent
    response = agent.get_response(
        messages=chat_history.get_messages(),
        settings=settings,
        tool_registry=tool_registry
    )
    
    # Display response
    print(f"Assistant: {response.response}")
    
    # Add generated messages to history
    chat_history.add_messages(response.messages)
```

## Saving and Loading Chat History

You can save chat history to and load it from JSON files:

```python
# Save chat history to a file
chat_history.save_to_json("conversation.json")

# Load chat history from a file
loaded_history = ChatHistory.load_from_json("conversation.json")
```

## Manipulating Chat History

You can also manipulate the chat history:

```python
# Get the number of messages
message_count = len(chat_history)

# Get a specific message
first_message = chat_history[0]

# Remove the last message
chat_history.pop()

# Clear the entire history
chat_history.clear()

# Set a completely new list of messages
new_messages = [
    ChatMessage.create_system_message("System prompt"),
    ChatMessage.create_user_message("User message")
]
chat_history.set_messages(new_messages)
```

## Advanced: Chat Database

For more complex applications, ToolAgents also provides a `ChatDatabase` class that can manage multiple conversations:

```python
from ToolAgents.messages import ChatDatabase

# Create a chat database
chat_db = ChatDatabase()

# Create a new conversation
conversation_id = chat_db.create_conversation("My Conversation")

# Add messages to a conversation
chat_db.add_system_message(conversation_id, "System instructions")
chat_db.add_user_message(conversation_id, "Hello")

# Get all messages for a conversation
messages = chat_db.get_messages(conversation_id)

# Save and load the entire database
chat_db.save_to_json("chat_database.json")
loaded_db = ChatDatabase.load_from_json("chat_database.json")
```

## Chat History with Memory

For agents that need long-term memory, you can combine chat history with semantic memory:

```python
from ToolAgents.agent_memory.semantic_memory import HierarchicalMemory

# Create a memory instance
memory = HierarchicalMemory()

# Store important information in memory
memory.add_memory("User likes hiking.")
memory.add_memory("User is from San Francisco.")

# Retrieve relevant memories for the current conversation
query = "What activities should I recommend?"
relevant_memories = memory.search_memory(query, top_k=3)

# Incorporate memories into the system message
system_message = f"""
You are a helpful assistant.
Here's what you know about the user:
{' '.join(relevant_memories)}
"""

chat_history = ChatHistory()
chat_history.add_system_message(system_message)
```

## Best Practices

When working with chat history:

1. **Keep system messages concise**: Include only what the agent needs to know
2. **Manage context length**: For long conversations, consider summarizing or truncating older messages
3. **Include tool calls**: Keep tool calls and their results in the history for context
4. **Preserve format**: Avoid modifying message formats to maintain compatibility
5. **Save regularly**: Save chat history at appropriate intervals to prevent data loss

## Next Steps

Now that you know how to work with chat history:

- [Use streaming responses](streaming.md) for a better user experience
- [Explore different agent types](../components/agents.md)
- [Learn about memory features](../examples/memory.md)
- [See complete examples](../examples/basic-agents.md)