---
title: Utilities API
---

# Utilities API

The Utilities module provides helper functionality for working with JSON schemas, GBNF grammars, and LLM documentation generation.

## JSON Schema Generator

Utilities for generating JSON schemas from Python objects.

```python
from ToolAgents.utilities.json_schema_generator.schema_generator import JSONSchemaGenerator
```

### Methods

#### `generate_schema(obj, title=None, description=None)`

Generates a JSON schema from a Python object.

**Parameters:**
- `obj`: The Python object to generate a schema from
- `title` (str, optional): Schema title
- `description` (str, optional): Schema description

**Returns:**
- `dict`: JSON schema

#### `generate_schema_from_type(type_obj, title=None, description=None)`

Generates a JSON schema from a Python type.

**Parameters:**
- `type_obj`: The Python type to generate a schema from
- `title` (str, optional): Schema title
- `description` (str, optional): Schema description

**Returns:**
- `dict`: JSON schema

#### `generate_schema_from_function(func, title=None, description=None)`

Generates a JSON schema from a Python function.

**Parameters:**
- `func`: The Python function to generate a schema from
- `title` (str, optional): Schema title
- `description` (str, optional): Schema description

**Returns:**
- `dict`: JSON schema

## GBNF Grammar Generator

Utilities for generating GBNF grammars from Pydantic models.

```python
from ToolAgents.utilities.gbnf_grammar_generator.gbnf_grammar_from_pydantic_models import GBNFGrammarGenerator
```

### Methods

#### `generate_grammar(pydantic_model, root_rule_name="root")`

Generates a GBNF grammar from a Pydantic model.

**Parameters:**
- `pydantic_model`: The Pydantic model to generate a grammar from
- `root_rule_name` (str): The name of the root rule

**Returns:**
- `str`: GBNF grammar

#### `generate_grammar_from_pydantic_models(model_list, root_rule_name="root")`

Generates a GBNF grammar from multiple Pydantic models.

**Parameters:**
- `model_list` (list): List of Pydantic models
- `root_rule_name` (str): The name of the root rule

**Returns:**
- `str`: GBNF grammar

## LLM Documentation

Utilities for generating documentation from code using LLMs.

```python
from ToolAgents.utilities.llm_documentation.documentation_generation import DocumentationGenerator
```

### Constructor Parameters

- `chat_api` (ChatAPIProvider): Provider for generating documentation
- `settings` (ProviderSettings, optional): Provider settings

### Methods

#### `generate_docstring(code, classname=None)`

Generates a docstring for a code snippet.

**Parameters:**
- `code` (str): The code to document
- `classname` (str, optional): Name of the class being documented

**Returns:**
- `str`: Generated docstring

#### `generate_class_documentation(code)`

Generates comprehensive documentation for a class.

**Parameters:**
- `code` (str): The class code to document

**Returns:**
- `str`: Generated documentation

#### `generate_function_documentation(code)`

Generates comprehensive documentation for a function.

**Parameters:**
- `code` (str): The function code to document

**Returns:**
- `str`: Generated documentation

#### `generate_module_documentation(code)`

Generates comprehensive documentation for a module.

**Parameters:**
- `code` (str): The module code to document

**Returns:**
- `str`: Generated documentation

#### `improve_docstring(code, original_docstring)`

Improves an existing docstring.

**Parameters:**
- `code` (str): The code with the docstring
- `original_docstring` (str): The original docstring

**Returns:**
- `str`: Improved docstring