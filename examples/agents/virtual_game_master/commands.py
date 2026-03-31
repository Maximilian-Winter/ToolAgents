from typing import Tuple

from chat_history import ChatFormatter
from command_system import CommandSystem


@CommandSystem.command("exit", description="Save the game and exit.")
def exit_command(vgm) -> Tuple[str, bool]:
    vgm.save()
    return "Game saved. Goodbye!", True


@CommandSystem.command("save", description="Manually save the current game state.")
def save_command(vgm) -> Tuple[str, bool]:
    vgm.manual_save()
    return "Game saved successfully!", False


@CommandSystem.command("view_fields", description="Display all template fields and their current values.")
def view_fields(vgm):
    fields = vgm.template_fields
    output = "Template Fields:\n"
    output += "-----------------\n"
    for key, value in fields.items():
        output += f"{key}: {value}\n"
    return output, False


@CommandSystem.command("edit_field", description="Edit the value of a specific template field.")
def edit_field(vgm, field_name: str, new_value: str):
    if field_name in vgm.template_fields:
        vgm.template_fields[field_name] = new_value
        return f"Field '{field_name}' updated successfully.", False
    else:
        return f"Field '{field_name}' not found.", False


@CommandSystem.command("view_messages", description="Display the last N messages in the chat history.")
def view_messages(vgm, count: int = 10):
    messages = vgm.history.messages[-count:]
    output = f"Last {count} Messages:\n"
    output += "-----------------\n"
    for msg in messages:
        output += f"ID: {msg.id}, Role: {msg.role}\n"
        output += f"Content: {msg.content[:50]}{'...' if len(msg.content) > 50 else ''}\n\n"
    return output, False


@CommandSystem.command("edit_message", description="Edit the content of a specific message by its ID.")
def edit_message(vgm, message_id: int, new_content: str):
    if vgm.edit_message(message_id, new_content):
        return f"Message {message_id} updated successfully.", False
    else:
        return f"Message {message_id} not found.", False


@CommandSystem.command("delete_last", description="Delete the last N messages from the chat history.")
def delete_last_messages(vgm, count: int):
    if count <= 0:
        return "Please provide a positive number of messages to delete.", False
    deleted = vgm.history.delete_last_messages(count)
    return f"Deleted the last {deleted} message(s).", False


@CommandSystem.command("rm_all", description="Delete all messages from the chat history.")
def delete_all_messages(vgm):
    deleted = vgm.history.delete_last_messages(100000)
    return f"Deleted {deleted} message(s). Chat history is now empty.", False


@CommandSystem.command("show_history", description="Display the currently used chat history currently in use.")
def show_history(vgm):
    output = vgm.get_current_history_formatted()
    return output, False


@CommandSystem.command("show_history_full", description="Display the complete chat history currently in use.")
def show_history_full(vgm):
    output = vgm.get_complete_history_formatted()
    return output, False


@CommandSystem.command("help", description="Display all available commands and their descriptions.")
def help_command(vgm, command: str = None) -> Tuple[str, bool]:
    if command:
        if command in CommandSystem.commands:
            description = CommandSystem.commands[command]["description"]
            usage = CommandSystem.get_command_usage(command)
            return f"Command: {command}\nDescription: {description}\nUsage: {usage}", False
        else:
            return f"Unknown command: {command}", False
    else:
        descriptions = CommandSystem.get_command_descriptions()
        output = "Available commands:\n"
        for name, description in descriptions.items():
            usage = CommandSystem.get_command_usage(name)
            output += f"{usage}\n    {description}\n\n"
        return output, False
