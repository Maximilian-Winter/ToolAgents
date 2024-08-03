# Installation Guide

This guide will walk you through the process of installing ToolAgents and setting up your environment.

## System Requirements

- Python 3.7 or later
- pip (Python package installer)

## Installation Steps

1. Install ToolAgents using pip:
   ```
   pip install ToolAgents
   ```

2. (Optional) If you're using API-based providers like OpenAI or Anthropic, install the `python-dotenv` package to manage environment variables:
   ```
   pip install python-dotenv
   ```

3. Set up your environment variables. Create a `.env` file in your project root and add your API keys:
   ```
   OPENAI_API_KEY=your_openai_api_key
   ANTHROPIC_API_KEY=your_anthropic_api_key
   ```

4. Install any additional dependencies based on your chosen provider:
    - For llama.cpp server: No additional installation needed
    - For vLLM server: No additional installation needed
    - For Ollama: Install Ollama on your system (see [Ollama documentation](https://ollama.ai/))

## Verifying Installation

To verify that ToolAgents is installed correctly, run the following Python code:

```python
from ToolAgents.agents import MistralAgent
from ToolAgents.provider import LlamaCppServerProvider

# This should run without any import errors
print("ToolAgents is installed successfully!")
```

## Troubleshooting

If you encounter any issues during installation:

1. Ensure you're using a compatible Python version.
2. Check that you have the latest version of pip: `pip install --upgrade pip`
3. If you're behind a proxy, make sure your pip is configured correctly.
4. For API-related issues, verify that your environment variables are set correctly.

If problems persist, please check the [project's GitHub issues](https://github.com/Your-GitHub-Username/ToolAgents/issues) or open a new issue for support.

## Next Steps

Now that you have ToolAgents installed, you can proceed to the [Configuration Guide](configuration.md) to set up your first agent.