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
                return f"Output:\n{output}"
            else:
                return "Code executed successfully with no output."
        else:
            return "No Python code found in the response."

    def get_variable(self, var_name):
        return self.global_context.get(var_name, f"Variable '{var_name}' not found in the context.")

    def get_context(self):
        return {k: v for k, v in self.global_context.items() if not k.startswith('__')}


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
