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

This will install the core ToolAgents library with basic dependencies required for function calling with various language model providers.

## Installing Optional Features

ToolAgents offers additional features that can be installed based on your needs:

### Vector Embeddings and RAG Support

If you plan to use vector embeddings, semantic search, or RAG capabilities:

```bash
pip install ToolAgents[additional_deps]
```

This installs:

- sentence_transformers, hdbscan, transformers
- sentencepiece, protobuf, chromadb
- PDF processing tools (pdf2image, pytesseract)
- HTML processing tools (lxml_html_clean)

### Web Search Capabilities

For web search and crawling functionality:

```bash
pip install ToolAgents[search]
```

This installs:

- googlesearch-python
- markdownify
- camoufox

### Complete Installation

To install ToolAgents with all optional dependencies:

```bash
pip install ToolAgents[additional_deps,search]
```

## Development Installation

If you want to contribute to ToolAgents or modify the source code:

```bash
git clone https://github.com/Maximilian-Winter/ToolAgents
cd ToolAgents
pip install -e .
```

This will install the package in development mode, allowing you to make changes to the code and see them reflected immediately.

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
import ToolAgents
print(ToolAgents.__version__)
```

## Troubleshooting

If you encounter any issues during installation:

- Ensure you're using Python 3.10 or higher
- Try upgrading pip: `pip install --upgrade pip`
- If installing with optional dependencies fails, try installing dependencies individually
- For PDF processing issues, ensure you have the required system dependencies (e.g., Tesseract OCR)

If problems persist, please [create an issue](https://github.com/Maximilian-Winter/ToolAgents/issues) on GitHub.