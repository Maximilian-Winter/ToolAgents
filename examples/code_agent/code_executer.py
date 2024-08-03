import re
import sys
import io
import contextlib


class PythonCodeExecutor:
    def __init__(self):
        self.code_pattern = re.compile(r'```python_interpreter\n(.*?)```', re.DOTALL)
        self.global_context = {}

    def extract_code(self, response):
        match = self.code_pattern.search(response)
        if match:
            return match.group(1).strip()
        return None

    def execute_code(self, code):
        # Redirect stdout and stderr
        old_stdout = sys.stdout
        old_stderr = sys.stderr
        redirected_output = sys.stdout = io.StringIO()
        redirected_error = sys.stderr = io.StringIO()

        try:
            # Execute the code within the global context
            exec(code, self.global_context)
            output = redirected_output.getvalue()
            error = redirected_error.getvalue()
            return output, error
        except Exception as e:
            return "", str(e)
        finally:
            # Restore stdout and stderr
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
```
## Function Definitions and Calling

You can define functions and call them within the same code block or in subsequent blocks. Here's an example:

```python_interpreter
def greet(name):
    return f"Hello, {name}! Welcome to the AI assistant."

# Call the function
result = greet("User")
print(result)
```

## Accessing External Libraries

You have access to various Python libraries that can be imported and used. Here are some examples:

1. Using `os` for system operations:
```python_interpreter
import os

# List files in the current directory
files = os.listdir('.')
print(f"Files in the current directory: {files}")
```

2. Using `requests` for HTTP requests:
```python_interpreter
import requests

response = requests.get('https://api.example.com/data')
if response.status_code == 200:
    print(f"Data received: {response.json()}")
else:
    print(f"Error: {response.status_code}")
```

3. Using `pandas` for data analysis:
```python_interpreter
import pandas as pd

# Create a sample dataframe
data = {'Name': ['Alice', 'Bob', 'Charlie'],
        'Age': [25, 30, 35],
        'City': ['New York', 'San Francisco', 'London']}
df = pd.DataFrame(data)

# Display basic statistics
print(df.describe())
```

## Error Handling

Always include proper error handling in your code to manage potential issues:

```python_interpreter
try:
    # Your code here
    result = 10 / 0  # This will raise a ZeroDivisionError
except Exception as e:
    print(f"An error occurred: {str(e)}")
```

## Persistent Variables

Variables defined in one code block can be accessed in subsequent blocks within the same conversation. Use this feature to build on previous computations:

```python_interpreter
# First code block
x = 5
y = 10

# Second code block
z = x + y
print(f"The sum of x and y is: {z}")
```"""
# Example usage
if __name__ == "__main__":
    executor = PythonCodeExecutor()

    # First execution
    response1 = '''
```python_interpreter
x = 10
y = 20
print(f"x + y = {x + y}")
```
    '''
    print("First execution:")
    print(executor.run(response1))

    # Second execution, using the context from the first
    response2 = '''
```python_interpreter
z = x * y
print(f"x * y = {z}")
```
    '''
    print("\nSecond execution:")
    print(executor.run(response2))

    # Check the value of a specific variable
    print("\nValue of z:")
    print(executor.get_variable('z'))

    # Print the entire context
    print("\nEntire context:")
    print(executor.get_context())
