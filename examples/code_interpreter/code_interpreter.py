import os
import platform
import shutil
import subprocess
import sys
import traceback

import venv
import tempfile
from typing import List, Dict

import psutil
import requests

from ToolAgents import FunctionTool


class CodeInterpreter:
    def __init__(self, venv_path, additional_packages=None,
                 additional_blocking_commands=None):
        self.cwd = os.getcwd()
        self.venv_path = venv_path
        if not os.path.exists(venv_path):
            self.create_venv(venv_path)
            default_packages = ['numpy', 'pandas', 'matplotlib', 'seaborn']
            if additional_packages is not None:
                default_packages.extend(additional_packages)
            self.install_dependencies(default_packages)
        self.blocking_commands = ["npm start", "npm run", "node server.js"]
        if additional_blocking_commands is not None:
            self.blocking_commands.extend(additional_blocking_commands)

    def is_blocking_command(self, command: str) -> bool:
        """
        Determine if a command is a blocking command.

        Args:
            command (str): The command to check.

        Returns:
            bool: True if the command is blocking, False otherwise.
        """
        return any(blocking_cmd in command for blocking_cmd in self.blocking_commands)

    def change_directory(self, new_dir: str) -> str:
        """
        Change the current working directory if it is different from the current one.

        Args:
            new_dir (str): The directory to change to.

        Returns:
            str: A message indicating the result of the directory change.
        """
        try:
            if not os.path.isabs(new_dir):
                new_dir = os.path.join(self.cwd, new_dir)
            new_dir = os.path.normpath(new_dir)

            if os.path.isdir(new_dir):
                if self.cwd != new_dir:
                    self.cwd = new_dir
                    return f"Changed directory to {self.cwd}\n"
                else:
                    return f"Already in directory: {self.cwd}\n"
            else:
                return f"Directory does not exist: {new_dir}\n"
        except Exception as e:
            tb = traceback.format_exc()
            return f"Error changing directory: {str(e)}\n{tb}"

    def execute_command(self, command: str) -> str:
        """
        Execute a single CLI command.

        Args:
            command (str): The CLI command to execute.

        Returns:
            str: The output or error of the command.
        """
        try:
            if self.is_blocking_command(command):
                subprocess.Popen(command, shell=True, cwd=self.cwd)
                return f"Started blocking command: {command}\n"
            else:
                result = subprocess.run(command, shell=True, cwd=self.cwd, capture_output=True, text=True)
                if result.returncode == 0:
                    return result.stdout
                else:
                    return f"Error executing command: {command}\n{result.stderr}"
        except Exception as e:
            tb = traceback.format_exc()
            return f"Exception occurred while executing command: {str(e)}\n{tb}"

    def execute_cli_commands(self, commands: List[str]) -> str:
        """
        Executes the given CLI commands on the system. Holds the current working directory between calls.

        Args:
            commands (List[str]): The CLI commands to execute.

        Returns:
            str: The combined results of the executed commands.
        """
        results = []

        for command in commands:
            sub_commands = command.split('&&')
            for sub_command in sub_commands:
                sub_command = sub_command.strip()
                if sub_command.startswith("cd "):
                    new_dir = sub_command[3:].strip()
                    normalized_path = os.path.normpath(new_dir)
                    base_name = os.path.basename(normalized_path)
                    if not self.cwd.endswith(base_name):
                        results.append(self.change_directory(new_dir))
                else:
                    results.append(self.execute_command(sub_command))

        return '\n'.join(results)

    def create_venv(self, venv_path: str):
        """
        Create a virtual environment at the specified path.

        Args:
            venv_path (str): The path to create the virtual environment.
        """
        self.venv_path = venv_path
        venv.create(self.venv_path, with_pip=True)
        print(f"Virtual environment created at {self.venv_path}")

    def install_dependencies(self, packages: List[str]):
        """
        Install the necessary dependencies in the virtual environment.

        Args:
            packages (str): packages to install.
        """
        if not self.venv_path:
            raise ValueError("Virtual environment path must be specified.")

        pip_executable = os.path.join(self.venv_path, 'Scripts', 'pip')
        command = [pip_executable, 'install']
        command.extend(packages)
        subprocess.check_call(command)

    def python_code_interpreter(self, code: str) -> str:
        """
        Interprets the provided Python code.

        Args:
            code (str): The Python code to interpret.

        Returns:
            str: The output of the executed code.
        """
        if not self.venv_path:
            raise ValueError("Virtual environment path must be specified.")

        python_executable = os.path.join(self.venv_path, 'Scripts', 'python')
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix='.py') as temp_script:
                temp_script.write(code.encode('utf-8'))
                temp_script_path = temp_script.name

            result = subprocess.run([python_executable, temp_script_path], cwd=self.cwd, capture_output=True, text=True)

            os.remove(temp_script_path)
            if result.returncode == 0:
                return result.stdout
            else:
                return f"Error during Python code execution:\n{result.stderr}"
        except Exception as e:
            tb = traceback.format_exc()
            return f"Exception occurred:\n{str(e)}\n{tb}"

    def get_environment_information(self) -> Dict[str, str]:
        """
        Gets detailed information about the current environment.

        Returns:
            Dict[str, str]: A dictionary containing various system information.
        """
        info = {
            "OS": platform.system(),
            "OS Version": platform.version(),
            "OS Release": platform.release(),
            "Python Version": sys.version,
            "CPU": platform.processor(),
            "CPU Cores": str(psutil.cpu_count(logical=False)),
            "Logical CPUs": str(psutil.cpu_count(logical=True)),
            "Total RAM": f"{psutil.virtual_memory().total / (1024 ** 3):.2f} GB",
            "Available RAM": f"{psutil.virtual_memory().available / (1024 ** 3):.2f} GB",
            "Disk Usage": f"{shutil.disk_usage('/').used / (1024 ** 3):.2f} GB / {shutil.disk_usage('/').total / (1024 ** 3):.2f} GB",
            "Current Working Directory": self.cwd,
            "Virtual Environment Path": self.venv_path
        }
        return info

    def list_directory_contents(self, path: str = None) -> List[str]:
        """
        Lists the contents of a directory.

        Args:
            path (str, optional): The path to list. Defaults to current working directory.

        Returns:
            List[str]: A list of files and directories in the specified path.
        """
        if path is None:
            path = self.cwd
        try:
            return os.listdir(path)
        except Exception as e:
            return [f"Error listing directory: {str(e)}"]

    def read_file_contents(self, file_path: str) -> str:
        """
        Reads and returns the contents of a file.

        Args:
            file_path (str): The path to the file to read.

        Returns:
            str: The contents of the file.
        """
        try:
            with open(file_path, 'r') as file:
                return file.read()
        except Exception as e:
            return f"Error reading file: {str(e)}"

    def write_file_contents(self, file_path: str, content: str) -> str:
        """
        Writes content to a file.

        Args:
            file_path (str): The path to the file to write.
            content (str): The content to write to the file.

        Returns:
            str: A message indicating success or failure.
        """
        try:
            with open(file_path, 'w') as file:
                file.write(content)
            return f"Successfully wrote to {file_path}"
        except Exception as e:
            return f"Error writing to file: {str(e)}"

    def download_file(self, url: str, save_path: str) -> str:
        """
        Downloads a file from a given URL and saves it to the specified path.

        Args:
            url (str): The URL of the file to download.
            save_path (str): The path where the file should be saved.

        Returns:
            str: A message indicating success or failure.
        """
        try:
            response = requests.get(url)
            response.raise_for_status()
            with open(save_path, 'wb') as file:
                file.write(response.content)
            return f"Successfully downloaded file to {save_path}"
        except Exception as e:
            return f"Error downloading file: {str(e)}"

    def get_tools(self):
        return [
            FunctionTool(self.get_environment_information),
            FunctionTool(self.python_code_interpreter),
            FunctionTool(self.execute_cli_commands),
            FunctionTool(self.install_dependencies),
            FunctionTool(self.list_directory_contents),
            FunctionTool(self.read_file_contents),
            FunctionTool(self.write_file_contents),
            FunctionTool(self.download_file)
        ]
