---
title: Utilities API
---

# Utilities API

Helpers for prompt construction, JSON-schema and grammar generation, LLM-facing
documentation, chat persistence and logging.

## Message Template

Fills `{placeholder}` fields in a prompt string. Used by pipeline steps to turn
a `prompt_template` into the actual prompt.

```python
from ToolAgents.utilities.message_template import MessageTemplate

template = MessageTemplate.from_string("Hello {name}, welcome to {place}.")
template.generate_message_content(name="Max", place="Königswinter")
```

### Placeholder paths

A placeholder may address a nested value with `/`:

```python
template = MessageTemplate.from_string("Revise {outputs/draft} for {inputs/audience}")
template.generate_message_content(results)
```

This works against any nested mapping, and against a
[`PipelineResults`](pipelines.md#results) object, which resolves the first
segment as a section name.

!!! warning "Pass the mapping; do not unpack it"

    Path resolution needs the structure intact.
    `generate_message_content(results)` resolves `{outputs/draft}`;
    `generate_message_content(**results)` flattens the mapping first and the
    path can no longer resolve. Mixing `template_fields` with keyword arguments
    flattens too.

### Unmatched placeholders

- A **bare** name with no matching field is blanked, and a line containing
  nothing else is dropped (`remove_empty_template_field=True`, the default).
  This is deliberate for optional sections of a prompt, but it means a typo in
  a bare placeholder fails silently.
- A **path** that does not resolve is left verbatim, so text that was always
  literal — `{a/b}` — survives, and a mistyped path stays visible.
- A field whose value is `None` counts as absent and is blanked, rather than
  rendering the literal string `"None"`.

Set `remove_empty_template_field=False` to leave every unmatched placeholder in
the output.

!!! warning "Behaviour change"

    Two things about placeholder handling changed when path support was added
    (unreleased at time of writing; it lands in the first release after 0.3.2).
    Neither breaks a template that was previously working, but both are worth
    knowing before upgrading.

    **Placeholder names may now contain `/`.** The pattern widened from
    `\{(\w+)\}` to `\{([\w/]+)\}`. Text that was always literal is unaffected:
    a placeholder containing `/` that does not resolve is left exactly as
    written, so `{a/b}` in an existing prompt still renders as `{a/b}`.

    **A `None` value now counts as absent.** Previously a field set to `None`
    rendered the literal string `"None"` into the prompt; it is now blanked
    like any other missing field. This is the one case where output changes for
    an unmodified template. If you were relying on the old behaviour, pass the
    string yourself:

    ```python
    template.generate_message_content(value="None" if value is None else value)
    ```

::: ToolAgents.utilities.message_template.MessageTemplate

::: ToolAgents.utilities.message_template.resolve_template_path

::: ToolAgents.utilities.message_template.ChatFormatter

## Prompt Builder

A fluent builder for multi-part prompts.

::: ToolAgents.utilities.prompt_builder.PromptBuilder

::: ToolAgents.utilities.prompt_builder.PromptPart

::: ToolAgents.utilities.prompt_builder.PromptVar

::: ToolAgents.utilities.prompt_builder.PromptLine

## JSON Schema Generation

`custom_json_schema` is the entry point: it produces a fully `$ref`-resolved
schema for a Pydantic model, which is what tool definitions are built from.

::: ToolAgents.utilities.json_schema_generator.schema_generator.custom_json_schema

::: ToolAgents.utilities.json_schema_generator.schema_generator.get_tools_schema

::: ToolAgents.utilities.json_schema_generator.schema_generator.refine_schema

::: ToolAgents.utilities.json_schema_generator.schema_generator.insert_additional_fields

::: ToolAgents.utilities.json_schema_generator.schema_generator.AdditionalSchemaField

::: ToolAgents.utilities.json_schema_generator.schema_generator.AdditionalFieldPosition

## GBNF Grammar Generation

Constrained-decoding grammars for local inference backends such as
llama.cpp, generated from Pydantic models.

::: ToolAgents.utilities.gbnf_grammar_generator.gbnf_grammar_from_pydantic_models.generate_gbnf_grammar_and_documentation

::: ToolAgents.utilities.gbnf_grammar_generator.gbnf_grammar_from_pydantic_models.generate_gbnf_grammar_from_pydantic_models

::: ToolAgents.utilities.gbnf_grammar_generator.gbnf_grammar_from_pydantic_models.generate_gbnf_grammar_and_documentation_from_dictionaries

::: ToolAgents.utilities.gbnf_grammar_generator.gbnf_grammar_from_pydantic_models.generate_and_save_gbnf_grammar_and_documentation

::: ToolAgents.utilities.gbnf_grammar_generator.gbnf_grammar_from_pydantic_models.PydanticDataType

## LLM Documentation

Renders Pydantic models as documentation *for the model to read*, so a tool's
shape can be explained inside a prompt.

::: ToolAgents.utilities.llm_documentation.documentation_generation.generate_markdown_documentation

::: ToolAgents.utilities.llm_documentation.documentation_generation.generate_text_documentation

::: ToolAgents.utilities.llm_documentation.documentation_generation.generate_type_definitions

## Pydantic Helpers

Note the module name is `pydantic_utilites` (the spelling in the source).

::: ToolAgents.utilities.pydantic_utilites.create_dynamic_model_from_function

::: ToolAgents.utilities.pydantic_utilites.pydantic_model_to_openai_function_definition

## MCP Schema Conversion

Turns an MCP JSON schema into a Pydantic model, so an MCP tool becomes an
ordinary `FunctionTool`.

::: ToolAgents.utilities.mcp_conversion.convert_json_schema

::: ToolAgents.utilities.mcp_conversion.convert_schema_to_pydantic_model

!!! note "MCP clients moved"

    `ToolAgents.utilities.mcp_session` is a compatibility shim. Import MCP
    client helpers from `ToolAgents.tool_adapters.mcp_client` instead.

## Chat Persistence

SQLAlchemy-backed chat storage. Requires the `storage` extra.

::: ToolAgents.utilities.chat_database.ChatManager

::: ToolAgents.utilities.chat_database.Chat

::: ToolAgents.utilities.chat_database.ChatMessageDb

## Logging

::: ToolAgents.utilities.logger_utilities.EasyLogger
