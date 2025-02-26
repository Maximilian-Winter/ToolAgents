# ToolAgents Development Guide

## Build & Test Commands
```bash
# Install package in development mode
pip install -e .
# Run all tests
python -m unittest discover -s tests
# Run a specific test
python -m unittest tests/test_tools.py
# Build package
python -m build
```

## Code Style Guidelines
- **Naming**: Classes=PascalCase, functions/variables=snake_case, constants=UPPER_SNAKE_CASE
- **Imports**: stdlib → third-party → project (absolute imports), grouped by type with blank lines
- **Type Hints**: Required for all parameters and return values, use Optional/Union when needed
- **Docstrings**: Google style with Args/Returns sections, examples for complex functionality
- **Error Handling**: Specific exceptions with descriptive messages, try/except with specific types
- **Organization**: Abstract base classes define interfaces, implementation classes for concrete behavior
- **Validation**: Use Pydantic models for input validation, Field(...) for required parameters

## Development Practices
- Python 3.10+ required
- Comprehensive test coverage expected for all new features
- Follow single responsibility principle for functions and classes
- Use composition over inheritance when possible