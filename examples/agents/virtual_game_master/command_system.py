from typing import Callable, Dict, Tuple, Optional
import inspect

class CommandSystem:
    commands: Dict[str, Dict[str, any]] = {}
    command_prefix: str = "@"

    @classmethod
    def command(cls, name: str, description: Optional[str] = None):
        def decorator(func: Callable):
            cls.commands[name.lower()] = {
                "func": func,
                "description": description,
                "params": inspect.signature(func).parameters
            }
            return func
        return decorator

    @classmethod
    def handle_command(cls, vgm, command: str) -> Tuple[str, bool]:
        if not command.startswith(cls.command_prefix):
            return f"Invalid command prefix. Use '{cls.command_prefix}' to start a command.", False

        command_parts = command[len(CommandSystem.command_prefix):].split()
        command = command_parts[0]
        args = command_parts[1:]
        if command in cls.commands:
            cmd_info = cls.commands[command]
            func = cmd_info["func"]
            params = cmd_info["params"]

            # Prepare keyword arguments
            kwargs = {}
            for i, (param_name, param) in enumerate(list(params.items())[1:]):  # Skip 'vgm' parameter
                if i < len(args):
                    kwargs[param_name] = args[i]
                elif param.default != inspect.Parameter.empty:
                    kwargs[param_name] = param.default
                else:
                    return f"Missing required argument: {param_name}", False

            return func(vgm, **kwargs)
        return f"Unknown command: {command}", False

    @classmethod
    def set_command_prefix(cls, prefix: str):
        cls.command_prefix = prefix

    @classmethod
    def get_command_descriptions(cls) -> Dict[str, str]:
        return {name: cmd["description"] or "No description available."
                for name, cmd in cls.commands.items()}

    @classmethod
    def get_command_usage(cls, command: str) -> str:
        if command not in cls.commands:
            return f"Unknown command: {command}"

        cmd_info = cls.commands[command]
        params = cmd_info["params"]
        usage = f"{cls.command_prefix}{command}"

        for param_name, param in list(params.items())[1:]:  # Skip 'vgm' parameter
            if param.default == inspect.Parameter.empty:
                usage += f" <{param_name}>"
            else:
                usage += f" [{param_name}={param.default}]"

        return usage