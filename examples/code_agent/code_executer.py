import ast
import io
import re
import sys
from typing import List

from ToolAgents import FunctionTool


class PythonCodeExecutor:
    def __init__(self, tools: List[FunctionTool] = None):
        self.code_pattern = re.compile(r'```python_interpreter\n(.*?)```', re.DOTALL)
        self.global_context = {}
        self.predefined_functions = {}
        if tools:
            for tool in tools:
                self.predefined_functions[tool.model.__name__] = tool

        self._setup_predefined_functions()

    def _setup_predefined_functions(self):
        for func_name, func_tool in self.predefined_functions.items():
            self.global_context[func_name] = self._create_wrapped_function(func_tool)

    def _create_wrapped_function(self, func_tool):
        def wrapped_function(*args, **kwargs):
            if isinstance(func_tool.model, type):
                instance = func_tool.model(*args, **kwargs)
                return instance.run(**func_tool.additional_parameters)
            else:
                return func_tool.model(*args, **kwargs, **func_tool.additional_parameters)

        return wrapped_function

    def extract_code(self, response):
        match = self.code_pattern.search(response)
        if match:
            return match.group(1).strip()
        return None

    def execute_code(self, code):
        old_stdout = sys.stdout
        old_stderr = sys.stderr
        redirected_output = sys.stdout = io.StringIO()
        redirected_error = sys.stderr = io.StringIO()
        global_context = self.global_context.copy()
        try:
            # Parse the code into an AST
            tree = ast.parse(code)

            # Execute all statements except the last one
            for stmt in tree.body[:-1]:
                exec(ast.unparse(stmt), global_context)

            # For the last statement, we'll evaluate it and print the result if it's an expression
            last_stmt = tree.body[-1]
            if isinstance(last_stmt, ast.Expr):
                result = eval(ast.unparse(last_stmt), global_context)
                if result is not None:
                    print(repr(result))
            else:
                exec(ast.unparse(last_stmt), global_context)

            output = redirected_output.getvalue()
            error = redirected_error.getvalue()
            return output, error
        except Exception as e:
            return "", str(e)
        finally:
            sys.stdout = old_stdout
            sys.stderr = old_stderr

    def run(self, response):
        code = self.extract_code(response)
        if code:
            output, error = self.execute_code(code)
            if error:
                return f"Error occurred:\n{error}"
            elif output:
                return f"{output}"
            else:
                return "Code executed successfully!"
        else:
            return "No Python code found in the response."

    def get_variable(self, var_name):
        return self.global_context.get(var_name, f"Variable '{var_name}' not found in the context.")

    def get_context(self):
        return {k: v for k, v in self.global_context.items() if not k.startswith('__')}


system_message_code_agent = """You are an advanced AI assistant with the ability to interact directly with a computer system using Python code. You have access to a Python code interpreter that allows you to execute Python code to accomplish various tasks. This capability enables you to perform a wide range of operations, from simple calculations to complex data analysis and system interactions.

## Using the Python Interpreter

To use the Python code interpreter, write the code you want to execute in a markdown 'python_interpreter' code block. Here's a basic example:

```python_interpreter
print('Hello, World!')
```"""
