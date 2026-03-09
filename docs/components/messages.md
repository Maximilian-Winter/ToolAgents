---
title: Messages
---

# Messages

ToolAgents uses `ChatMessage` as the shared provider-independent message format.

## Core Types

### ChatMessage

Use `ChatMessage` for provider-neutral messages:

```python
from ToolAgents.data_models.messages import ChatMessage

system_message = ChatMessage.create_system_message("You are helpful.")
user_message = ChatMessage.create_user_message("Hello")
assistant_message = ChatMessage.create_assistant_message("Hi there")
```

### ChatHistory

`ChatHistory` is the main in-memory conversation helper:

```python
from ToolAgents.data_models.chat_history import ChatHistory

chat_history = ChatHistory()
chat_history.add_system_message("You are a helpful assistant.")
chat_history.add_user_message("What's the capital of France?")
messages = chat_history.get_messages()
```

Useful operations:

- `add_message(...)`
- `add_messages(...)`
- `get_messages()`
- `get_last_message()`
- `get_last_k_messages(k)`
- `remove_last_message()`
- `get_message_count()`
- `save_to_json(...)`
- `ChatHistory.load_from_json(...)`

### Chats

Use `Chats` when you want multiple JSON-serializable conversations:

```python
from ToolAgents.data_models.chat_history import Chats

chats = Chats()
chat_id = chats.create_chat("Support Chat")
chats.add_user_message(chat_id, "Hello")
```

### ChatManager

Use `ChatManager` when you want database-backed persistence:

```python
from ToolAgents.data_models.messages import ChatMessage
from ToolAgents.utilities.chat_database import ChatManager

manager = ChatManager()
chat = manager.create_chat("Weather Chat")
manager.add_message(chat["id"], ChatMessage.create_user_message("Hello"))
```

## Templates and Prompt Construction

### MessageTemplate

```python
from ToolAgents.utilities.message_template import MessageTemplate

template = MessageTemplate.from_string(
    "You are an assistant specialized in {specialty}."
)
content = template.generate_message_content(specialty="weather")
```

### PromptBuilder

```python
from ToolAgents.utilities.prompt_builder import PromptBuilder

prompt = (
    PromptBuilder()
    .add_text("# Weather Assistant")
    .add_bullet_list(["Answer clearly", "Use tools when needed"])
    .build()
)
```

## Working with Agents

The usual flow is:

1. Build `ChatMessage` values directly or through `ChatHistory`.
2. Pass `chat_history.get_messages()` into an agent.
3. Append `response.messages` back into the history.
4. Persist with `save_to_json(...)` if you need a local record.
