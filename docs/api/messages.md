---
title: Messages API
---

# Messages API

The Messages module provides classes for managing conversation messages, chat history, and templates.

## ChatMessage

`ChatMessage` is the unified message format for all providers in ToolAgents.

```python
from ToolAgents.messages.chat_message import ChatMessage
```

### Properties

- `id` (str): Unique identifier
- `role` (str): Message role (system, user, assistant, tool)
- `content` (str): Message content
- `tool_calls` (List[dict]): Tool calls in the message
- `tool_call_id` (str): ID of the tool call (for tool messages)
- `name` (str): Name of the tool (for tool messages)
- `embedding` (List[float]): Vector embedding for semantic search

### Factory Methods

#### `create_system_message(content)`

Creates a system message.

**Parameters:**
- `content` (str): The message content

**Returns:**
- `ChatMessage`: A new system message

#### `create_user_message(content)`

Creates a user message.

**Parameters:**
- `content` (str): The message content

**Returns:**
- `ChatMessage`: A new user message

#### `create_assistant_message(content, tool_calls=None)`

Creates an assistant message.

**Parameters:**
- `content` (str): The message content
- `tool_calls` (List[dict], optional): List of tool calls

**Returns:**
- `ChatMessage`: A new assistant message

#### `create_tool_message(name, content, tool_call_id=None)`

Creates a tool message.

**Parameters:**
- `name` (str): The tool name
- `content` (any): The tool result content
- `tool_call_id` (str, optional): ID of the tool call

**Returns:**
- `ChatMessage`: A new tool message

### Methods

#### `get_as_text()`

Gets the message content as text.

**Returns:**
- `str`: Message content as text

#### `contains_tool_call()`

Checks if the message contains tool calls.

**Returns:**
- `bool`: True if message contains tool calls

#### `get_tool_calls()`

Gets tool calls from the message.

**Returns:**
- `List[dict]`: List of tool calls

#### `to_dict()`

Converts the message to a dictionary.

**Returns:**
- `dict`: Message as a dictionary

#### `from_dict(data)`

Creates a message from a dictionary.

**Parameters:**
- `data` (dict): Message data

**Returns:**
- `ChatMessage`: A new message

## ChatHistory

`ChatHistory` manages collections of chat messages.

```python
from ToolAgents.messages import ChatHistory

# Create a new chat history
chat_history = ChatHistory()
```

### Methods

#### `add_message(message)`

Adds a message to the history.

**Parameters:**
- `message` (ChatMessage): The message to add

#### `add_messages(messages)`

Adds multiple messages to the history.

**Parameters:**
- `messages` (List[ChatMessage]): Messages to add

#### `add_system_message(content)`

Adds a system message to the history.

**Parameters:**
- `content` (str): The message content

#### `add_user_message(content)`

Adds a user message to the history.

**Parameters:**
- `content` (str): The message content

#### `add_assistant_message(content, tool_calls=None)`

Adds an assistant message to the history.

**Parameters:**
- `content` (str): The message content
- `tool_calls` (List[dict], optional): List of tool calls

#### `add_tool_message(name, content, tool_call_id=None)`

Adds a tool message to the history.

**Parameters:**
- `name` (str): The tool name
- `content` (any): The tool result content
- `tool_call_id` (str, optional): ID of the tool call

#### `get_messages()`

Gets all messages in the history.

**Returns:**
- `List[ChatMessage]`: All messages

#### `get_last_k_messages(k)`

Gets the most recent messages.

**Parameters:**
- `k` (int): Number of messages to retrieve

**Returns:**
- `List[ChatMessage]`: The k most recent messages

#### `set_messages(messages)`

Sets the messages in the history.

**Parameters:**
- `messages` (List[ChatMessage]): New messages

#### `clear()`

Clears all messages from the history.

#### `pop()`

Removes and returns the last message.

**Returns:**
- `ChatMessage`: The removed message

#### `save_to_json(filepath)`

Saves the history to a JSON file.

**Parameters:**
- `filepath` (str): Path to the output file

#### `load_from_json(filepath)`

Loads history from a JSON file.

**Parameters:**
- `filepath` (str): Path to the input file

**Returns:**
- `ChatHistory`: A new chat history

#### `estimate_token_count()`

Estimates the number of tokens in the history.

**Returns:**
- `int`: Estimated token count

## ChatDatabase

`ChatDatabase` manages multiple conversations.

```python
from ToolAgents.messages import ChatDatabase

# Create a new chat database
chat_db = ChatDatabase()
```

### Methods

#### `create_conversation(name)`

Creates a new conversation.

**Parameters:**
- `name` (str): The conversation name

**Returns:**
- `str`: The conversation ID

#### `get_conversation(conversation_id)`

Gets a conversation by ID.

**Parameters:**
- `conversation_id` (str): The conversation ID

**Returns:**
- `ChatHistory`: The conversation history

#### `get_messages(conversation_id)`

Gets messages from a conversation.

**Parameters:**
- `conversation_id` (str): The conversation ID

**Returns:**
- `List[ChatMessage]`: Conversation messages

#### `add_message(conversation_id, message)`

Adds a message to a conversation.

**Parameters:**
- `conversation_id` (str): The conversation ID
- `message` (ChatMessage): The message to add

#### `add_messages(conversation_id, messages)`

Adds multiple messages to a conversation.

**Parameters:**
- `conversation_id` (str): The conversation ID
- `messages` (List[ChatMessage]): Messages to add

#### `add_system_message(conversation_id, content)`

Adds a system message to a conversation.

**Parameters:**
- `conversation_id` (str): The conversation ID
- `content` (str): The message content

#### `add_user_message(conversation_id, content)`

Adds a user message to a conversation.

**Parameters:**
- `conversation_id` (str): The conversation ID
- `content` (str): The message content

#### `add_assistant_message(conversation_id, content, tool_calls=None)`

Adds an assistant message to a conversation.

**Parameters:**
- `conversation_id` (str): The conversation ID
- `content` (str): The message content
- `tool_calls` (List[dict], optional): List of tool calls

#### `add_tool_message(conversation_id, name, content, tool_call_id=None)`

Adds a tool message to a conversation.

**Parameters:**
- `conversation_id` (str): The conversation ID
- `name` (str): The tool name
- `content` (any): The tool result content
- `tool_call_id` (str, optional): ID of the tool call

#### `list_conversations()`

Lists all conversations.

**Returns:**
- `List[dict]`: Conversation metadata

#### `delete_conversation(conversation_id)`

Deletes a conversation.

**Parameters:**
- `conversation_id` (str): The conversation ID

#### `save_to_json(filepath)`

Saves the database to a JSON file.

**Parameters:**
- `filepath` (str): Path to the output file

#### `load_from_json(filepath)`

Loads database from a JSON file.

**Parameters:**
- `filepath` (str): Path to the input file

**Returns:**
- `ChatDatabase`: A new chat database

## MessageTemplate

`MessageTemplate` provides a simple template system for messages.

```python
from ToolAgents.messages.message_template import MessageTemplate

# Create a template
template = MessageTemplate("You are an assistant specialized in {specialty}.")
```

### Methods

#### `format_message_content(**kwargs)`

Fills the template with values.

**Parameters:**
- `**kwargs`: Template variables

**Returns:**
- `str`: Formatted message content

#### `format_system_message(**kwargs)`

Creates a system message from the template.

**Parameters:**
- `**kwargs`: Template variables

**Returns:**
- `ChatMessage`: A new system message

#### `format_user_message(**kwargs)`

Creates a user message from the template.

**Parameters:**
- `**kwargs`: Template variables

**Returns:**
- `ChatMessage`: A new user message

#### `format_assistant_message(**kwargs)`

Creates an assistant message from the template.

**Parameters:**
- `**kwargs`: Template variables

**Returns:**
- `ChatMessage`: A new assistant message

## PromptBuilder

`PromptBuilder` helps construct complex prompts.

```python
from ToolAgents.messages.prompt_builder import PromptBuilder

# Create a builder
builder = PromptBuilder()
```

### Methods

#### `add_text(text)`

Adds text to the prompt.

**Parameters:**
- `text` (str): Text to add

#### `add_code(code, language=None)`

Adds a code block to the prompt.

**Parameters:**
- `code` (str): Code to add
- `language` (str, optional): Programming language

#### `add_list(items, ordered=False)`

Adds a list to the prompt.

**Parameters:**
- `items` (List[str]): List items
- `ordered` (bool): Whether to use ordered (numbered) list

#### `add_table(headers, rows)`

Adds a table to the prompt.

**Parameters:**
- `headers` (List[str]): Table headers
- `rows` (List[List[str]]): Table rows

#### `build()`

Builds the complete prompt.

**Returns:**
- `str`: The constructed prompt