# ToolAgents Development Guide

## Build/Test/Lint Commands
- Run all tests: `python3 -m unittest discover -s tests -p "test_*.py"`
- Run single test: `python3 -m unittest tests/test_file.py`
- Run specific test case: `python3 -m unittest tests/test_file.py::TestClass::test_method`
- Build package: `python -m build`
- Install package locally: `pip install -e .`

## Code Style Guidelines
- **Imports**: Group imports by standard library, third-party, and project modules
- **Types**: Use type hints for all function parameters and return values
- **Naming**:
  - Classes: PascalCase
  - Functions/Methods/Variables: snake_case
  - Constants: UPPER_SNAKE_CASE
- **Documentation**: Use docstrings for all public modules, classes, methods
- **Error Handling**: Use specific exceptions with descriptive messages
- **Pydantic Models**: Prefer Pydantic models for structured data with validation
- **Async Support**: Maintain both sync and async versions of client-facing APIs
- **Preprocessors/Postprocessors**: Use for parameter/result transformations

## File Organization
- Place related functionality in appropriate subdirectories
- Use `__init__.py` to expose public interfaces
- Keep implementation details in separate modules
- Group related tools in the same file or subdirectory

## Tool Development
- Implement tools using Pydantic models with clear documentation
- Properly handle errors and provide informative messages
- Support conversion to different LLM provider formats
- Follow consistent patterns for execution flows