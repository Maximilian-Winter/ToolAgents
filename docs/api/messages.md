---
title: Messages API
---

# Messages API

The current message surface is centered on `ChatMessage`, `ChatHistory`, `Chats`, `MessageTemplate`, and `PromptBuilder`.

## ChatMessage

```python
from ToolAgents.data_models.messages import ChatMessage
```

`ChatMessage` is the unified message model used across providers.

Common factory helpers:

- `ChatMessage.create_system_message(message)`
- `ChatMessage.create_user_message(message)`
- `ChatMessage.create_assistant_message(message)`
- `ChatMessage.create_custom_role_message(message, custom_role)`
- `ChatMessage.from_dictionaries(messages)` for simple `{"role", "content"}` payloads

Useful methods:

- `get_as_text()`
- `contains_tool_call()`
- `get_tool_calls()`
- `get_tool_call_results()`
- `model_dump()` / `model_dump_json()` from Pydantic

## ChatHistory

```python
from ToolAgents.data_models.chat_history import ChatHistory
```

Main methods:

- `add_message(message)`
- `add_messages(messages)`
- `add_system_message(message)`
- `add_user_message(message)`
- `add_assistant_message(message)`
- `add_message_from_dictionary(message)`
- `add_messages_from_dictionaries(messages)`
- `get_messages()`
- `get_last_message()`
- `get_last_k_messages(k)`
- `get_message_count()`
- `remove_last_message()`
- `clear()`
- `clear_history()`
- `save_to_json(filepath)`
- `ChatHistory.load_from_json(filepath)`

Example:

```python
history = ChatHistory()
history.add_system_message("You are helpful.")
history.add_user_message("Hello")
messages = history.get_messages()
```

## Chats

```python
from ToolAgents.data_models.chat_history import Chats
```

`Chats` is the JSON-serializable container for multiple `ChatHistory` objects.

Main methods:

- `create_chat(title)`
- `get_messages(chat_id)`
- `get_last_k_messages(chat_id, k)`
- `add_message(chat_id, message)`
- `add_system_message(chat_id, message)`
- `add_user_message(chat_id, message)`
- `add_assistant_message(chat_id, message)`
- `save_to_json(filepath)`
- `Chats.load_from_json(filepath)`

## ChatManager

```python
from ToolAgents.utilities.chat_database import ChatManager
```

`ChatManager` is the SQLite-backed storage option.

Main methods:

- `create_chat(title=None)`
- `add_message(chat_id, message)`
- `get_chat(chat_id)`
- `get_chat_messages(chat_id)`
- `delete_chat(chat_id)`
- `update_chat_title(chat_id, title)`

Example:

```python
chat_db = ChatManager()
chat = chat_db.create_chat("Weather Chat")
chat_db.add_message(chat["id"], ChatMessage.create_user_message("Hello"))
messages = chat_db.get_chat_messages(chat["id"])
```

## MessageTemplate

```python
from ToolAgents.utilities.message_template import MessageTemplate
```

Current construction helpers:

- `MessageTemplate.from_string(template_string)`
- `MessageTemplate.from_file(template_file)`
- `generate_message_content(template_fields=None, **kwargs)`

Example:

```python
template = MessageTemplate.from_string(
    "You are an assistant specialized in {specialty}."
)
content = template.generate_message_content(specialty="weather")
```

## PromptBuilder

```python
from ToolAgents.utilities.prompt_builder import PromptBuilder
```

Current builder helpers:

- `add_text(text)`
- `add_prompt_part(part)`
- `add_file_content(file_path)`
- `add_empty_line(n=1)`
- `add_numbered_list(items)`
- `add_bullet_list(items)`
- `add_code_block(code, language="")`
- `add_separator(char="-", length=40)`
- `build()`

Example:

```python
builder = PromptBuilder()
prompt = (
    builder
    .add_text("# Weather Assistant")
    .add_bullet_list(["Answer weather questions", "Use tools when needed"])
    .build()
)
```
