---
title: Messages
---

# Messages

ToolAgents uses a unified message format to handle chat history and communication between agents and LLM providers. This consistent format makes it easy to switch between different LLM providers while maintaining your chat history.

## Core Message Components

### ChatMessage

The `ChatMessage` class is the fundamental building block for communication:

```python
from ToolAgents.messages.chat_message import ChatMessage

# Create different types of messages
system_message = ChatMessage.create_system_message(
    "You are a helpful assistant that provides accurate information."
)

user_message = ChatMessage.create_user_message(
    "What's the capital of France?"
)

assistant_message = ChatMessage.create_assistant_message(
    "The capital of France is Paris."
)

# Create a message with a tool call
tool_call_message = ChatMessage.create_assistant_message(
    "", # Content is often empty with tool calls
    tool_calls=[
        {
            "id": "call_abc123",
            "type": "function",
            "function": {
                "name": "get_weather",
                "arguments": {"location": "Paris", "unit": "celsius"}
            }
        }
    ]
)

# Create a tool response message
tool_response_message = ChatMessage.create_tool_message(
    "get_weather",
    {"temperature": 22, "condition": "sunny", "humidity": 65}
)
```

### ChatHistory

The `ChatHistory` class manages collections of messages:

```python
from ToolAgents.messages import ChatHistory

# Create a new chat history
chat_history = ChatHistory()

# Add different types of messages
chat_history.add_system_message(
    "You are a helpful assistant with access to tools."
)
chat_history.add_user_message("What's the weather in Paris?")
chat_history.add_assistant_message("I'll check that for you.")
chat_history.add_tool_message(
    "get_weather",
    {"temperature": 22, "condition": "sunny", "location": "Paris"}
)

# Get all messages as a list
messages = chat_history.get_messages()

# Manipulate the history
chat_history.add_messages([another_message, yet_another_message])
chat_history.pop()  # Remove the last message
chat_history.clear()  # Remove all messages
```

### ChatDatabase

The `ChatDatabase` class manages multiple conversations:

```python
from ToolAgents.messages import ChatDatabase

# Create a new chat database
chat_db = ChatDatabase()

# Create a new conversation
conversation_id = chat_db.create_conversation("Weather Chat")

# Add messages to the conversation
chat_db.add_system_message(conversation_id, "You are a weather assistant.")
chat_db.add_user_message(conversation_id, "What's the weather in Paris?")

# Get messages from a conversation
messages = chat_db.get_messages(conversation_id)

# List all conversations
conversations = chat_db.list_conversations()

# Delete a conversation
chat_db.delete_conversation(conversation_id)
```

## Working with Messages

### Message Properties

`ChatMessage` objects have several properties:

```python
# Access message properties
print(message.role)       # "system", "user", "assistant", or "tool"
print(message.content)    # The text content of the message
print(message.tool_calls) # List of tool calls (may be empty)
print(message.tool_call_id) # ID of the tool call (for tool messages)
print(message.name)       # Name of the tool (for tool messages)
```

### Saving and Loading Chat History

You can save chat history to and load it from JSON files:

```python
# Save chat history to a file
chat_history.save_to_json("conversation.json")

# Load chat history from a file
loaded_history = ChatHistory.load_from_json("conversation.json")

# Save chat database to a file
chat_db.save_to_json("chat_database.json")

# Load chat database from a file
loaded_db = ChatDatabase.load_from_json("chat_database.json")
```

### Message Templates

ToolAgents provides message templates for common patterns:

```python
from ToolAgents.messages.message_template import MessageTemplate

# Create a template
template = MessageTemplate("""
You are an assistant specialized in {specialty}.
Your task is to help the user with {task}.
Remember to always be {trait}.
""")

# Fill in the template
system_message = template.format_system_message(
    specialty="weather forecasting",
    task="getting accurate weather information",
    trait="helpful and accurate"
)

# Add to chat history
chat_history.add_message(system_message)
```

### Prompt Building

For more complex prompts, use the `PromptBuilder`:

```python
from ToolAgents.messages.prompt_builder import PromptBuilder

# Create a builder
builder = PromptBuilder()

# Add components
builder.add_text("# Weather Assistant")
builder.add_text("You are a specialized weather assistant.")
builder.add_text("## Capabilities")
builder.add_list([
    "Check current weather conditions",
    "Provide weather forecasts",
    "Analyze weather patterns"
])
builder.add_text("## Instructions")
builder.add_text("Always provide temperatures in both Celsius and Fahrenheit.")

# Generate the prompt
prompt = builder.build()

# Create a system message
system_message = ChatMessage.create_system_message(prompt)
```

## Tool Calls and Responses

### Creating Tool Calls

```python
from ToolAgents.messages.chat_message import ChatMessage

# Create a message with a single tool call
calculator_call = ChatMessage.create_assistant_message(
    "",  # Content is often empty with tool calls
    tool_calls=[
        {
            "id": "call_123",
            "type": "function",
            "function": {
                "name": "calculator",
                "arguments": {"expression": "42 * 8"}
            }
        }
    ]
)

# Create a message with multiple tool calls
multi_call = ChatMessage.create_assistant_message(
    "",  # Content is often empty with tool calls
    tool_calls=[
        {
            "id": "call_123",
            "type": "function",
            "function": {
                "name": "calculator",
                "arguments": {"expression": "42 * 8"}
            }
        },
        {
            "id": "call_456",
            "type": "function",
            "function": {
                "name": "get_time",
                "arguments": {"timezone": "UTC"}
            }
        }
    ]
)
```

### Creating Tool Responses

```python
# Create tool response messages
calculator_response = ChatMessage.create_tool_message(
    "calculator",
    336,  # Result of 42 * 8
    tool_call_id="call_123"
)

time_response = ChatMessage.create_tool_message(
    "get_time",
    "2024-02-25T12:34:56Z",
    tool_call_id="call_456"
)
```

## Advanced Message Handling

### Message Conversion

ToolAgents handles message conversion between different provider formats internally:

```python
from ToolAgents.provider.message_converter import (
    OpenAIMessageConverter,
    AnthropicMessageConverter
)

# Create converters
openai_converter = OpenAIMessageConverter()
anthropic_converter = AnthropicMessageConverter()

# Convert to provider-specific format
openai_messages = openai_converter.convert_to_provider_messages(chat_history.get_messages())
anthropic_messages = anthropic_converter.convert_to_provider_messages(chat_history.get_messages())

# Convert from provider-specific format
unified_messages = openai_converter.convert_from_provider_messages(openai_messages)
```

### Handling Large Histories

For long conversations, you may need to manage context length:

```python
# Check the approximate token count
approximate_tokens = chat_history.estimate_token_count()

# Truncate older messages if needed
if approximate_tokens > 4000:
    # Keep only the last N messages
    chat_history.set_messages(chat_history.get_messages()[-10:])
    
    # Or keep system message and most recent messages
    system_message = chat_history.get_messages()[0]
    recent_messages = chat_history.get_messages()[-10:]
    chat_history.set_messages([system_message] + recent_messages)
```

## Best Practices

1. **Use Appropriate Message Types**: Match message types to their intended purpose
2. **Keep System Messages Concise**: Include only what the agent needs to know
3. **Maintain Context**: Preserve important context while managing token limits
4. **Include Tool Calls and Results**: Keep tool interactions in the history
5. **Structure User Messages**: Format complex user inputs clearly
6. **Save Regularly**: Persist chat history at appropriate intervals
7. **Use Templates for Consistency**: Standardize prompts with templates
8. **Validate Messages**: Ensure messages have appropriate content
9. **Tool Response Formatting**: Format tool responses consistently
10. **Consider Privacy**: Remove sensitive information before saving

## Next Steps

- [Learn about different agent types](agents.md)
- [Explore tool options](tools.md)
- [Understand provider differences](providers.md)
- [See examples of chat history usage](../guides/chat-history.md)