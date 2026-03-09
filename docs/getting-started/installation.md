---
title: Installation
---

# Installation

This guide will help you install ToolAgents and set up your development environment.

## Requirements

ToolAgents requires:

- Python 3.10 or higher
- pip (Python package installer)

## Basic Installation

The simplest way to install ToolAgents is through pip:

```bash
pip install ToolAgents
```

This installs the core framework and the built-in chat providers.

## Installing Optional Features

ToolAgents now groups optional functionality by feature area so you only install what you need.

### Advanced Agent App State

For `ContextAppState` and YAML-backed agent state files:

```bash
pip install ToolAgents[advanced]
```

This installs:

- `PyYAML`

### Database-Backed Chat Storage

For SQLite-backed chat storage with `ChatManager`:

```bash
pip install ToolAgents[storage]
```

This installs:

- `SQLAlchemy`

### Semantic Memory and Vector Search

For semantic memory, Chroma-backed storage, and sentence-transformer embeddings:

```bash
pip install ToolAgents[memory]
```

This installs:

- `chromadb`
- `hdbscan`
- `numpy`
- `sentence-transformers`
- `torch`

### Local Transformer-Based Inference

For Hugging Face-based completion backends and tokenizer support:

```bash
pip install ToolAgents[local-inference]
```

This installs:

- `transformers`
- `sentencepiece`
- `protobuf`

### OCR and PDF Processing

For OCR-based document ingestion:

```bash
pip install ToolAgents[ocr]
```

This installs:

- `PyMuPDF`
- `joblib`
- `pdf2image`
- `pytesseract`

### Web Search and Crawling

For web search and crawling functionality:

```bash
pip install ToolAgents[search]
```

This installs:

- `beautifulsoup4`
- `camoufox`
- `googlesearch-python`
- `html5lib`
- `trafilatura`

### Model Context Protocol

For MCP client/server tooling:

```bash
pip install ToolAgents[mcp]
```

This installs:

- `mcp[cli]`

### Complete Installation

To install ToolAgents with all optional dependencies:

```bash
pip install ToolAgents[all]
```

## Development Installation

If you want to contribute to ToolAgents or modify the source code:

```bash
git clone https://github.com/Maximilian-Winter/ToolAgents
cd ToolAgents
pip install -e .
```

To include every optional feature during development:

```bash
pip install -e .[all]
```

## API Keys Setup

Most LLM providers require API keys for authentication. You'll need to obtain keys from your chosen providers:

- [OpenAI API](https://platform.openai.com/)
- [Anthropic API](https://www.anthropic.com/product)
- [Mistral API](https://mistral.ai/)
- [Groq API](https://groq.com/)

We recommend using environment variables or a `.env` file to manage your API keys securely.

## Verifying Installation

After installation, you can verify that ToolAgents is correctly installed by running:

```python
from ToolAgents import ToolRegistry
from ToolAgents.agents import ChatToolAgent
from ToolAgents.provider import OpenAIChatAPI

print(ToolRegistry)
print(ChatToolAgent)
print(OpenAIChatAPI)
```

## Troubleshooting

If you encounter any issues during installation:

- Ensure you're using Python 3.10 or higher
- Try upgrading pip: `pip install --upgrade pip`
- If an optional feature import fails, install the matching extra for that subsystem
- For OCR issues, ensure you have the required system dependencies installed, especially Tesseract OCR

If problems persist, please [create an issue](https://github.com/Maximilian-Winter/ToolAgents/issues) on GitHub.
