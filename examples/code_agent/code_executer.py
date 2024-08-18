import ast
import io
import re
import sys
from enum import Enum
from inspect import isclass
from typing import List, Dict, Any

from pydantic import BaseModel
from ToolAgents import FunctionTool
from ToolAgents.utilities import ChatHistory
from ToolAgents.utilities.documentation_generation import generate_function_definition, generate_class_definition
from ToolAgents.utilities.message_template import MessageTemplate

system_message_code_agent = """You are an advanced AI assistant with Python code execution capabilities. You have access to a Python interpreter that allows you to run code in real-time, enabling you to perform a wide range of tasks from simple calculations to complex data analysis and system interactions.

## Using the Python Interpreter

To execute Python code, enclose it within a markdown code block tagged as Python. For example:

```python
print('Hello, World!')
```

After writing the code, it will be automatically executed, and the output will be displayed.

## Available Resources

You have access to the following pre-defined resources. These are already imported and ready to use - you don't need to import or define them again.

### Predefined Types

The following are class stubs. These define the structure and attributes of available custom types, but do not include method implementations. You can use these types in your code, but you don't need to (and shouldn't) implement their methods or modify their structure.

```python
{predefined_types}
```

### Predefined Functions

The following are function stubs. These define the interface and purpose of available functions, but do not include the actual implementation. You can use these functions in your code, but you don't need to (and shouldn't) implement them yourself.

```python
{predefined_functions}
```

### Predefined Variables

The following are predefined variables available in the environment. These variables are already initialized and can be used directly in your code. Do not attempt to reinitialize or modify these variables unless explicitly instructed to do so.

```python
{predefined_variables}
```"""


class PythonCodeExecutor:
    def __init__(self, predefined_functions: List[FunctionTool] = None, predefined_types: list = None,
                 predefined_variables: list = None):
        self.code_pattern = re.compile(r'```python\n(.*?)```', re.DOTALL)
        self.global_context = {}
        self.predefined_types = {}
        self.predefined_functions = {}
        self.predefined_variables = {}

        predefined_types_docs = []
        predefined_functions_docs = []
        predefined_variables_docs = []

        if predefined_types:
            for predefined_type in predefined_types:
                self.predefined_types[predefined_type.__name__] = predefined_type
                if issubclass(predefined_type, BaseModel):
                    predefined_types.append(generate_class_definition(predefined_type, predefined_type.__name__))

        if predefined_functions:
            for function in predefined_functions:
                self.predefined_functions[function.model.__name__] = function
                predefined_functions_docs.append(function.get_python_documentation())

        if predefined_variables:
            for variable in predefined_variables:
                self.predefined_variables[variable.__name__] = variable
                predefined_variables_docs.append(variable.__name__)

        template = MessageTemplate.from_string(system_message_code_agent)
        self.system_message_code_agent = template.generate_message_content(
            predefined_types='\n\n'.join(predefined_types_docs),
            predefined_functions='\n\n'.join(predefined_functions_docs),
            predefined_variables='\n'.join(predefined_functions_docs))
        self._setup_predefined_types()
        self._setup_predefined_functions()
        self._setup_predefined_variables()

    def _setup_predefined_types(self):
        for class_name, class_obj in self.predefined_types.items():
            self.global_context[class_name] = self._create_wrapped_class(class_obj)

    def _setup_predefined_functions(self):
        for func_name, func_tool in self.predefined_functions.items():
            self.global_context[func_name] = self._create_wrapped_function(func_tool)

    def _setup_predefined_variables(self):
        for variable_name, variable_obj in self.predefined_variables.items():
            self.global_context[variable_name] = variable_obj

    def get_python_interpreter_system_message(self):
        return self.system_message_code_agent

    def _create_wrapped_function(self, func_tool):
        def wrapped_function(*args, **kwargs):
            model_class = func_tool.model

            # Convert args to kwargs if any
            if args:
                arg_names = list(model_class.model_fields.keys())
                kwargs.update(zip(arg_names, args))

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

    def extract_code(self, response):
        matches = self.code_pattern.findall(response)
        return [match.strip() for match in matches]

    def execute_code(self, code):
        old_stdout = sys.stdout
        old_stderr = sys.stderr
        redirected_output = sys.stdout = io.StringIO()
        redirected_error = sys.stderr = io.StringIO()
        try:
            # Parse the code into an AST
            tree = ast.parse(code)

            # Execute all statements
            for stmt in tree.body:
                if isinstance(stmt, ast.Expr):
                    # If it's an expression, evaluate it and print the result if it's not None
                    result = eval(ast.unparse(stmt), self.global_context)
                    if result is not None:
                        print(repr(result))
                else:
                    # For other statements, just execute them
                    exec(ast.unparse(stmt), self.global_context)

            output = redirected_output.getvalue()
            error = redirected_error.getvalue()
            return output, error
        except Exception as e:
            return "", str(e)
        finally:
            sys.stdout = old_stdout
            sys.stderr = old_stderr

    def run(self, response):
        code_blocks = self.extract_code(response)
        if code_blocks:
            outputs = []
            has_error = False
            for code in code_blocks:
                output, error = self.execute_code(code)
                if error:
                    outputs.append(f"Error occurred:\n{error}")
                    has_error = True
                elif output:
                    outputs.append(f"{output}")
                else:
                    outputs.append("Code executed successfully!")
            output = ""
            counter = 1
            for out in outputs:
                output += f"{counter}. Codeblock Output: {out}\n"
                counter += 1
            return output, has_error

        else:
            return "No Python code found in the response.", False

    def get_variable(self, var_name):
        return self.global_context.get(var_name, f"Variable '{var_name}' not found in the context.")

    def get_context(self):
        return {k: v for k, v in self.global_context.items() if not k.startswith('__')}


def run_code_agent(agent, settings, chat_history: ChatHistory, user_input: str,
                   python_code_executor: PythonCodeExecutor):
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
        if "```python_interpreter" in full_response:
            full_response += "\n```\n"
            code_ex, has_error = python_code_executor.run(full_response)
            print("Python Execution Output: ")
            print(code_ex)
            chat_history.add_message("user",
                                     "This is an automatic response from the Python Interpreter:\n\nResults of last Code execution:\n" + code_ex)

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
