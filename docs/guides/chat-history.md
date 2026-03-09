---
title: Working with Chat History
---

# Working with Chat History

`ChatHistory` is the main in-memory conversation container in ToolAgents.

## Basic Usage

```python
from ToolAgents.data_models.chat_history import ChatHistory

chat_history = ChatHistory()
chat_history.add_system_message(
    "You are a helpful assistant with tool calling capabilities."
)
chat_history.add_user_message("Can you help me find the weather in San Francisco?")

messages = chat_history.get_messages()
```

## Adding Messages

```python
from ToolAgents.data_models.messages import ChatMessage

chat_history.add_system_message("Additional system instructions...")
chat_history.add_user_message("What's the weather like today?")
chat_history.add_assistant_message("I'll check that for you.")

chat_history.add_message(
    ChatMessage.create_user_message("Another question")
)
chat_history.add_messages(
    [
        ChatMessage.create_assistant_message("Another answer"),
    ]
)
```

For simple text dictionaries, you can also use:

```python
chat_history.add_message_from_dictionary(
    {"role": "user", "content": "Hello"}
)
chat_history.add_messages_from_dictionaries(
    [
        {"role": "system", "content": "You are helpful."},
        {"role": "user", "content": "What can you do?"},
    ]
)
```

## Using Chat History with Agents

```python
from ToolAgents import ToolRegistry
from ToolAgents.agents import ChatToolAgent
from ToolAgents.provider import OpenAIChatAPI

api = OpenAIChatAPI(api_key="your-api-key", model="gpt-4o-mini")
agent = ChatToolAgent(chat_api=api)
settings = api.get_default_settings()

tool_registry = ToolRegistry()
# tool_registry.add_tool(your_tool)

response = agent.get_response(
    messages=chat_history.get_messages(),
    settings=settings,
    tool_registry=tool_registry,
)

chat_history.add_messages(response.messages)
```

## Saving and Loading

```python
chat_history.save_to_json("conversation.json")
loaded_history = ChatHistory.load_from_json("conversation.json")
```

## Common Operations

```python
message_count = chat_history.get_message_count()
last_message = chat_history.get_last_message()
recent_messages = chat_history.get_last_k_messages(5)

chat_history.remove_last_message()
chat_history.clear()
chat_history.clear_history()
```

## Managing Multiple Chats

For multiple in-memory conversations, use `Chats`:

```python
from ToolAgents.data_models.chat_history import Chats

chats = Chats()
chat_id = chats.create_chat("My Conversation")
chats.add_system_message(chat_id, "You are helpful.")
chats.add_user_message(chat_id, "Hello")

messages = chats.get_messages(chat_id)
chats.save_to_json("chats.json")
loaded_chats = Chats.load_from_json("chats.json")
```

## Database-Backed Chat Storage

If you want SQLite-backed persistence, use `ChatManager`:

```python
from ToolAgents.data_models.messages import ChatMessage
from ToolAgents.utilities.chat_database import ChatManager

chat_db = ChatManager()
chat = chat_db.create_chat("My Conversation")
chat_db.add_message(chat["id"], ChatMessage.create_user_message("Hello"))
messages = chat_db.get_chat_messages(chat["id"])
```

`ChatManager` is database-oriented; JSON save/load is handled by `ChatHistory` and `Chats`.

## Best Practices

1. Keep `ChatHistory` as the source of truth for the current conversation.
2. Append full `response.messages` after agent calls so tool results stay in context.
3. Use `Chats` when you want multiple JSON-serializable conversations.
4. Use `ChatManager` only when you want database-backed storage.
