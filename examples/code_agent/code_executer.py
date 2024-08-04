import ast
import io
import re
import sys
from typing import List

from ToolAgents import FunctionTool
from ToolAgents.utilities import ChatHistory


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
            model_class = func_tool.model

            # Convert args to kwargs if any
            if args:
                arg_names = []
                for field_key, field_info in model_class.model_fields.items():
                    arg_names.append(field_key)

                kwargs.update(zip(arg_names, args))

            # Instantiate the model using only keyword arguments
            instance = model_class(**kwargs)

            return instance.run(**func_tool.additional_parameters)

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
        global_context = self.global_context
        try:
            # Parse the code into an AST
            tree = ast.parse(code)

            # Execute all statements
            for stmt in tree.body:
                if isinstance(stmt, ast.Expr):
                    # If it's an expression, evaluate it and print the result if it's not None
                    result = eval(ast.unparse(stmt), global_context)
                    if result is not None:
                        print(repr(result))
                else:
                    # For other statements, just execute them
                    exec(ast.unparse(stmt), global_context)

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


def run_code_agent(agent, settings, chat_history: ChatHistory, user_input: str,
                   python_code_executor: PythonCodeExecutor):
    print("User: " + user_input)
    print("Response: ", end="")
    chat_history.add_user_message(user_input)
    result_gen = agent.get_streaming_response(
        messages=chat_history.to_list(),
        sampling_settings=settings, tools=None)

    full_response = ""
    for tok in result_gen:
        print(tok, end="", flush=True)
        full_response += tok
    print()
    while True:
        chat_history.add_assistant_message(message=full_response)
        if "```python_interpreter" in full_response:
            code_ex = python_code_executor.run(full_response)
            print("Python Execution Output: ")
            print(code_ex)
            chat_history.add_user_message("Results of last Code execution:\n" + code_ex)

            print("Response: ", end="")
            result_gen = agent.get_streaming_response(
                messages=chat_history.to_list(),
                sampling_settings=settings, tools=None)
            full_response = ""
            for tok in result_gen:
                print(tok, end="", flush=True)
                full_response += tok
            print()
        else:
            break
