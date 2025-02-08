import ast
import io
import re
import sys
import asyncio
import inspect
from enum import Enum
from typing import List, Dict, Any, Optional, Tuple, Union, Callable
from dataclasses import dataclass
from contextlib import contextmanager
from pydantic import BaseModel
from ToolAgents import FunctionTool
from ToolAgents.utilities import ChatHistory
from ToolAgents.utilities.documentation_generation import generate_type_definitions

# System messages for LLM instruction
SYSTEM_MESSAGES = {
    'base': """You are an advanced AI assistant with the ability to execute Python code. You have access to a Python code interpreter that allows you to execute Python code to accomplish various tasks. This capability enables you to perform a wide range of operations, from simple calculations to complex data analysis and system interactions. Only you can see the direct results of the last code execution.

## Using the Python Interpreter

To use the Python code interpreter, write the code you want to execute in a markdown python-interpreter code block. The following examples show you how to use it:

Example: Print 'Hello World!'
```python-interpreter
print('Hello, World!')
```""",

    'debug': """You can use debug mode by using the python-debug code block:
```python-debug
x = 1
y = 2
print(x + y)
```

In debug mode, you'll receive step-by-step information about the execution.""",

    'async': """For asynchronous operations, use the python-async code block:
```python-async
await asyncio.sleep(1)
print("Async operation completed")
```""",

    'safe_file': """For safe file operations, use the provided safe_open function:
```python-interpreter
with safe_open('example.txt', 'w') as f:
    f.write('Hello from safe file operation!')
```"""
}


@dataclass
class ExecutionResult:
    """Represents the result of a code execution for LLM consumption."""
    output: str
    error: Optional[str]
    has_error: bool
    return_value: Any = None
    debug_info: Optional[Dict[str, Any]] = None


class CodeBlock:
    """Represents a code block with its execution context."""

    def __init__(self, code: str, block_type: str = "python-interpreter"):
        self.code = code.strip()
        self.block_type = block_type
        self.context: Dict[str, Any] = {}


class PythonExecutor:
    """Python Code Executor designed specifically for LLM interaction."""

    def __init__(self,
                 predefined_types: Optional[List[type]] = None,
                 predefined_functions: Optional[List[FunctionTool]] = None,
                 predefined_variables: Optional[List[Any]] = None,
                 enable_safety_features: bool = True):
        """Initialize the executor with LLM-friendly features."""
        self.code_patterns = {
            'python-interpreter': re.compile(r'```python-interpreter\n(.*?)```', re.DOTALL),
            'python-debug': re.compile(r'```python-debug\n(.*?)```', re.DOTALL),
            'python-async': re.compile(r'```python-async\n(.*?)```', re.DOTALL)
        }

        self.global_context = {}
        self.enable_safety_features = enable_safety_features
        self.execution_history: List[ExecutionResult] = []

        # Add these missing attributes
        self.predefined_types: Dict[str, type] = {}
        self.predefined_functions: Dict[str, FunctionTool] = {}
        self.predefined_variables: Dict[str, Any] = {}

        # Setup execution environment
        self._setup_safe_environment()
        self._initialize_resources(predefined_types, predefined_functions, predefined_variables)

        # Generate system message for LLM
        self.system_message = self._generate_system_message()

    def _initialize_resources(self,
                              types: Optional[List[type]] = None,
                              functions: Optional[List[FunctionTool]] = None,
                              variables: Optional[List[Any]] = None) -> None:
        """Initialize all predefined resources and make them available in the execution context."""
        # Initialize types
        if types:
            for type_cls in types:
                self.predefined_types[type_cls.__name__] = type_cls
                if issubclass(type_cls, BaseModel):
                    self.global_context[type_cls.__name__] = self._create_wrapped_class(type_cls)
                else:
                    self.global_context[type_cls.__name__] = type_cls

        # Initialize functions
        if functions:
            for func in functions:
                func_name = func.model.__name__
                self.predefined_functions[func_name] = func
                self.global_context[func_name] = self._create_wrapped_function(func)

        # Initialize variables
        if variables:
            for var in variables:
                var_name = var.__name__
                self.predefined_variables[var_name] = var
                self.global_context[var_name] = var

    def _create_wrapped_function(self, func_tool):
        def wrapped_function(*args, **kwargs):
            model_class = func_tool.model

            # Convert args to kwargs if any
            if args:
                arg_names = list(model_class.model_fields.keys())
                kwargs.update(zip(arg_names, args))

            for kwarg_name, kwarg_value in kwargs.items():
                if isinstance(kwarg_value, Enum):
                    kwargs[kwarg_name] = kwarg_value.value

            # Instantiate the model using only keyword arguments
            instance = model_class(**kwargs)

            return instance.run(**func_tool.additional_parameters)

        return wrapped_function

    def _create_wrapped_class(self, class_obj):
        if issubclass(class_obj, BaseModel):
            def wrapped_class(*args, **kwargs):
                model_class = class_obj

                # Convert args to kwargs if any
                if args:
                    arg_names = list(model_class.model_fields.keys())
                    kwargs.update(zip(arg_names, args))

                # Instantiate the model using only keyword arguments
                instance = model_class(**kwargs)

                return instance

            return wrapped_class
        return class_obj

    def _setup_safe_environment(self):
        """Setup a safe execution environment with controlled access."""

        def safe_print(*args, **kwargs):
            """Safe print function that captures output for LLM consumption."""
            output = io.StringIO()
            kwargs['file'] = output
            print(*args, **kwargs)
            return output.getvalue()

        def safe_input(prompt=""):
            """Safe input function that notifies about input requests."""
            raise NotImplementedError("Input operations are not supported in this environment")

        self.global_context['print'] = safe_print
        self.global_context['input'] = safe_input

        # Add safe file operations
        self.global_context['safe_open'] = self._create_safe_file_handler()

    def _create_safe_file_handler(self):
        """Create a safe file handling function for LLM use."""
        import tempfile
        import pathlib

        temp_dir = tempfile.mkdtemp(prefix='llm_safe_')
        safe_path = pathlib.Path(temp_dir)

        def safe_open(filename: str, mode: str = 'r') -> io.IOBase:
            """Safely open files within a temporary directory."""
            if mode[0] not in {'r', 'w', 'a'}:
                raise ValueError("Unsupported file mode")
            return open(safe_path / filename, mode)

        return safe_open

    async def execute_code_block(self, block: CodeBlock) -> ExecutionResult:
        """Execute a code block based on its type."""
        try:
            if block.block_type == 'python-async':
                return await self._execute_async_code(block.code)
            elif block.block_type == 'python-debug':
                return await self._execute_debug_code(block.code)
            else:
                return self._execute_normal_code(block.code)
        except Exception as e:
            return ExecutionResult(
                output="",
                error=str(e),
                has_error=True
            )

    def _execute_normal_code(self, code: str) -> ExecutionResult:
        """Execute code in normal mode with proper output capturing."""
        with self._redirect_output() as (stdout, stderr):
            try:
                tree = ast.parse(code)
                last_value = None

                for stmt in tree.body:
                    if isinstance(stmt, ast.Expr):
                        last_value = eval(ast.unparse(stmt), self.global_context)
                        if last_value is not None:
                            print(repr(last_value))
                    else:
                        exec(ast.unparse(stmt), self.global_context)

                return ExecutionResult(
                    output=stdout.getvalue(),
                    error=stderr.getvalue(),
                    has_error=False,
                    return_value=last_value
                )
            except Exception as e:
                return ExecutionResult("", str(e), True)

    async def _execute_async_code(self, code: str) -> ExecutionResult:
        """Execute asynchronous code."""
        try:
            # Wrap code in async function
            wrapped_code = f"async def __async_exec():\n" + \
                           "\n".join(f"    {line}" for line in code.split("\n"))

            local_vars = {}
            exec(wrapped_code, self.global_context, local_vars)
            result = await local_vars['__async_exec']()

            return ExecutionResult(
                output="Async execution completed successfully",
                error=None,
                has_error=False,
                return_value=result
            )
        except Exception as e:
            return ExecutionResult("", str(e), True)

    async def _execute_debug_code(self, code: str) -> ExecutionResult:
        """Execute code in debug mode with step information."""
        debug_info = []

        def trace_lines(frame, event, arg):
            if event == 'line':
                info = {
                    'line': frame.f_lineno,
                    'locals': dict(frame.f_locals),
                    'code': frame.f_code.co_name
                }
                debug_info.append(info)
            return trace_lines

        sys.settrace(trace_lines)
        try:
            exec(code, self.global_context)
            return ExecutionResult(
                output="Debug execution completed",
                error=None,
                has_error=False,
                debug_info={"steps": debug_info}
            )
        finally:
            sys.settrace(None)

    @contextmanager
    def _redirect_output(self):
        """Context manager for handling output redirection."""
        old_stdout, old_stderr = sys.stdout, sys.stderr
        stdout = io.StringIO()
        stderr = io.StringIO()
        sys.stdout, sys.stderr = stdout, stderr
        try:
            yield stdout, stderr
        finally:
            sys.stdout, sys.stderr = old_stdout, old_stderr

    def _generate_system_message(self) -> str:
        """Generate the system message for LLM instruction."""
        message_parts = [SYSTEM_MESSAGES['base']]

        if self.predefined_types or self.predefined_functions or self.predefined_variables:
            message_parts.append("\n## Available Resources\n")

            if self.predefined_types:
                types_doc = generate_type_definitions(list(self.predefined_types.values()))
                message_parts.append("### Predefined Types\n```python\n{}\n```".format('\n\n'.join(types_doc)))

            if self.predefined_functions:
                funcs_doc = [func.get_python_documentation() for func in self.predefined_functions.values()]
                message_parts.append("### Predefined Functions\n```python\n{}\n```".format('\n\n'.join(funcs_doc)))

            if self.predefined_variables:
                vars_doc = [f"{var.__name__}" for var in self.predefined_variables.values()]
                message_parts.append("### Predefined Variables\n```python\n{}\n```".format('\n'.join(vars_doc)))

        message_parts.append(SYSTEM_MESSAGES['debug'])
        message_parts.append(SYSTEM_MESSAGES['async'])
        message_parts.append(SYSTEM_MESSAGES['safe_file'])

        return '\n\n'.join(message_parts)

    def get_system_message(self) -> str:
        """Get the system message for LLM instruction."""
        return self.system_message

    async def run(self, llm_response: str) -> Tuple[str, bool]:
        """Process and execute code from LLM response."""
        outputs = []
        has_error = False

        for pattern_name, pattern in self.code_patterns.items():
            matches = pattern.findall(llm_response)
            for code in matches:
                block = CodeBlock(code.strip(), pattern_name)
                result = await self.execute_code_block(block)

                if result.has_error:
                    outputs.append(f"Error in {pattern_name}:\n{result.error}")
                    has_error = True
                else:
                    outputs.append(f"Output from {pattern_name}:\n{result.output}")

                self.execution_history.append(result)

        if not outputs:
            return "No executable code found in the response.", False

        return "\n".join(outputs), has_error


def run_llm_code_agent(agent, settings, chat_history: ChatHistory, user_input: str,
                       executor: PythonExecutor):
    """Run the code agent with LLM interaction."""
    print("User: " + user_input)
    print("Response: ", end="")
    chat_history.add_user_message(user_input)

    result_gen = agent.get_streaming_response(
        messages=chat_history.to_list(),
        settings=settings)

    full_response = ""
    for tok in result_gen:
        print(tok, end="", flush=True)
        full_response += tok
    print()

    while True:
        chat_history.add_assistant_message(message=full_response)
        if any(pattern in full_response for pattern in ['```python-interpreter', '```python-debug', '```python-async']):
            output, has_error = asyncio.run(executor.run(full_response))
            print("Execution Output:")
            print(output)
            chat_history.add_message("user", "Execution results:\n" + output)

            print("Response: ", end="")
            result_gen = agent.get_streaming_response(
                messages=chat_history.to_list(),
                settings=settings)
            full_response = ""
            for tok in result_gen:
                print(tok, end="", flush=True)
                full_response += tok
            print()
        else:
            break
