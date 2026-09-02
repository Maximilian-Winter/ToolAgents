---
title: Messages API
---

# Messages API

`ChatMessage` is the format every provider converts to and from, which is what
lets a chat history move between OpenAI, Anthropic, Groq and Mistral unchanged.

```python
from ToolAgents.data_models.messages import ChatMessage
from ToolAgents.data_models.chat_history import ChatHistory
```

## ChatMessage

::: ToolAgents.data_models.messages.ChatMessage

::: ToolAgents.data_models.messages.ChatMessageRole

## Message content

A message's content is a list of typed parts, so text, reasoning, tool calls
and binary attachments coexist in one message.

::: ToolAgents.data_models.messages.ContentType

::: ToolAgents.data_models.messages.ContentBase

::: ToolAgents.data_models.messages.TextContent

::: ToolAgents.data_models.messages.ReasoningContent

::: ToolAgents.data_models.messages.ToolCallContent

::: ToolAgents.data_models.messages.ToolCallResultContent

::: ToolAgents.data_models.messages.BinaryContent

::: ToolAgents.data_models.messages.BinaryStorageType

## Streaming and usage

::: ToolAgents.data_models.messages.StreamingChatMessage

::: ToolAgents.data_models.messages.TokenUsage

## Chat history

::: ToolAgents.data_models.chat_history.ChatHistory

::: ToolAgents.data_models.chat_history.Chats

## Templates and prompt construction

`MessageTemplate` fills `{placeholder}` fields in a prompt string. A
placeholder may address a nested value with `/`, which is how a
[pipeline](../guides/pipelines.md#results-sections) step reads a section of its
results:

```python
template = MessageTemplate.from_string("Revise {outputs/draft} for {inputs/audience}")
template.generate_message_content(results)
```

!!! warning "Pass the mapping; do not unpack it"

    Path resolution needs the structure intact.
    `generate_message_content(results)` resolves `{outputs/draft}`;
    `generate_message_content(**results)` flattens the mapping first and the
    path can no longer resolve. Mixing `template_fields` with keyword arguments
    flattens too.

Unmatched placeholders behave differently by shape:

- A **bare** name with no matching field is blanked, and a line containing
  nothing else is dropped (`remove_empty_template_field=True`, the default).
- A **path** that does not resolve is left verbatim, so text that was always
  literal — `{a/b}` — survives, and a mistyped path stays visible rather than
  vanishing.
- A field whose value is `None` counts as absent and is blanked, rather than
  rendering the literal string `"None"`. This changed in 0.3.3 — see
  [the note in the utilities reference](utilities.md#unmatched-placeholders).

See the [utilities API](utilities.md#message-template) for the full
`MessageTemplate` reference, and
[`PromptBuilder`](utilities.md#prompt-builder) for assembling multi-part
prompts.
