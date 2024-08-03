# Getting Started with ToolAgents

ToolAgents is a lightweight and flexible framework for creating function-calling agents with various language models and APIs. This guide will help you get started with ToolAgents, covering the basics of installation, configuration, and usage.

## Prerequisites

Before you begin, make sure you have:

- Python 3.7 or later installed
- pip (Python package installer)
- Access to a supported LLM provider (e.g., llama.cpp server, OpenAI API, Anthropic API, or Ollama)

## Quick Start

1. Install ToolAgents:
   ```
   pip install ToolAgents
   ```

2. Create a simple agent:

   ```python
   from ToolAgents.agents import MistralAgent
   from ToolAgents.provider import LlamaCppServerProvider

   # Initialize the provider and agent
   provider = LlamaCppServerProvider("http://127.0.0.1:8080/")
   agent = MistralAgent(llm_provider=provider, system_prompt="You are a helpful assistant.")

   # Get a response
   result = agent.get_response("Hello! How are you?")
   print(result)
   ```

4. Run your script:
   ```
   python your_script.py
   ```

## Next Steps

- Learn how to [create custom tools](configuration.md#creating-custom-tools)
- Explore different [agent types](configuration.md#agent-types)
- Understand [provider-specific settings](configuration.md#provider-settings)

For more detailed information, check out the [Installation Guide](installation.md) and [Configuration Guide](configuration.md).