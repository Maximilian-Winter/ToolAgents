import os
import asyncio
from typing import List, Dict, Any

from ToolAgents import ToolRegistry, FunctionTool
from ToolAgents.agents import ChatToolAgent
from ToolAgents.messages import ChatHistory
from ToolAgents.messages.chat_message import ChatMessage
from ToolAgents.provider import OpenAIChatAPI
from ToolAgents.agent_tools.discord_tool import (
    init_discord_tools,
    DiscordEmbedData,
)

from dotenv import load_dotenv

load_dotenv()

# Make sure you have set DISCORD_BOT_TOKEN environment variable
if not os.getenv("DISCORD_BOT_TOKEN"):
    raise EnvironmentError(
        "DISCORD_BOT_TOKEN environment variable not set. "
        "Please set it to your Discord bot token."
    )

# IMPORTANT: Discord API Requirements
# 1. Make sure you have invited the bot to your server with the necessary permissions:
#    - Read/Send Messages
#    - Read Message History
#    - Add Reactions
#    - Manage Channels (if you want to create/delete channels)
#    - Manage Messages (if you want to delete messages)
#
# 2. For functionality requiring privileged intents:
#    - Go to https://discord.com/developers/applications/
#    - Select your application
#    - Go to "Bot" section
#    - Enable "Message Content Intent" and "Server Members Intent"
#    - If your bot is in 100+ servers, you'll need to get verified by Discord

# Set up API client
api = OpenAIChatAPI(api_key=os.getenv("OPENAI_API_KEY"), model="gpt-4o-mini")
settings = api.get_default_settings()
settings.temperature = 0.2
agent = ChatToolAgent(chat_api=api)

# Create a tool registry and add Discord tools
tool_registry = ToolRegistry()

# Initialize Discord tools, decide whether to enable privileged intents
# Set to True ONLY if you've enabled them in the Discord Developer Portal
ENABLE_PRIVILEGED_INTENTS = False  # Change to True if you've enabled privileged intents

# If you're getting PrivilegedIntentsRequired errors, either:
# 1. Set ENABLE_PRIVILEGED_INTENTS to False (limits some functionality)
# 2. Go to https://discord.com/developers/applications/ and enable the intents for your bot:
#    - Message Content Intent
#    - Server Members Intent
discord_tools = init_discord_tools(enable_privileged_intents=ENABLE_PRIVILEGED_INTENTS)
tool_registry.add_tools(discord_tools)

# Initialize chat history
chat_history = ChatHistory()
chat_history.add_system_message(
    """
You are a Discord management agent with the ability to interact with Discord servers.
You can send messages, list channels, get server information, and more.

You need to always confirm the guild_id and channel_id before performing any actions.
For security reasons, you must not perform actions on servers or channels that the user
has not explicitly mentioned.

Available Discord tools:
- SendDiscordMessage: Send a text or embed message to a Discord channel
- GetDiscordChannels: List all channels in a Discord server
- GetDiscordMessages: Get recent messages from a Discord channel
- GetDiscordMembers: List all members in a Discord server
- CreateDiscordChannel: Create a new channel in a Discord server
- GetDiscordGuildInfo: Get information about a Discord server
- AddDiscordReaction: Add a reaction to a Discord message

When the user requests an action, ask for any missing information needed to
complete the request. Always prioritize security and privacy.
"""
)


def run_conversation():
    """Run an interactive conversation with the Discord agent"""
    print("Discord Agent Example")
    print("--------------------")
    print("This agent can help you manage your Discord server.")
    print("Type 'exit' to end the conversation")
    print("Type 'help' to see available Discord tools")

    guild_id = None
    while True:
        user_input = input("\nYou > ")

        if user_input.lower() == "exit":
            break

        elif user_input.lower() == "help":
            print("\nAvailable Discord tools:")
            for tool in discord_tools:
                openai_tool = tool.to_openai_tool()
                print(
                    f"- {openai_tool['function']['name']}: {openai_tool['function']['description']}"
                )
            continue

        # Process guild_id from input
        if "guild_id:" in user_input:
            try:
                # Extract guild_id if the user provides it in the format "guild_id: 123456789"
                guild_id_part = user_input.split("guild_id:")[1].split()[0].strip()
                guild_id = int(guild_id_part)
                print(f"Set current guild ID to: {guild_id}")

                # Remove the guild_id part from the user input
                user_input = user_input.replace(
                    f"guild_id: {guild_id_part}", ""
                ).strip()

                if not user_input:
                    # If there's no other input, skip processing
                    continue
            except (IndexError, ValueError):
                print("Invalid guild ID format. Please use 'guild_id: 123456789'")
                continue

        # Add guild_id to the user message if available but not explicitly mentioned
        if guild_id and "guild_id" not in user_input:
            user_input = f"{user_input} (Using guild_id: {guild_id})"

        # Add the user's message to the chat history
        chat_history.add_user_message(user_input)

        # Get a response from the agent
        print("\nAgent is thinking...")
        response = agent.get_response(
            messages=chat_history.get_messages(),
            settings=settings,
            tool_registry=tool_registry,
        )

        # Print the response
        print(f"Agent > {response.response}")

        # Add the agent's messages to the chat history
        chat_history.add_messages(response.messages)


# Example of using the Discord agent programmatically
def send_announcement(
    guild_id: int, channel_id: int, title: str, message: str, color: int = 3447003
):
    """Send an announcement with an embed to a Discord channel"""
    # Create the embed data
    embed_data = DiscordEmbedData(
        title=title,
        description=message,
        color=color,  # Blue color by default
        footer_text="Sent by Discord Agent",
    )

    # Create the input for the Discord tool
    prompt = f"""
    Please send an embed message to the Discord channel with ID {channel_id} in guild {guild_id}.
    
    The embed should have:
    - Title: {title}
    - Description: {message}
    - Color: Blue
    - Footer text: "Sent by Discord Agent"
    """

    # Add the message to chat history
    chat_history.add_user_message(prompt)

    # Get a response from the agent
    response = agent.get_response(
        messages=chat_history.get_messages(),
        settings=settings,
        tool_registry=tool_registry,
    )

    # Add the agent's messages to the chat history
    chat_history.add_messages(response.messages)

    return response.response


if __name__ == "__main__":
    # You can either run the interactive conversation
    run_conversation()

    # Or use the agent programmatically (uncomment to use)
    # Replace these with your actual guild_id and channel_id
    # GUILD_ID = 1234567890
    # CHANNEL_ID = 1234567890
    # result = send_announcement(
    #     guild_id=GUILD_ID,
    #     channel_id=CHANNEL_ID,
    #     title="Important Announcement",
    #     message="Hello everyone! This message was sent by an AI Discord agent."
    # )
    # print(f"Result: {result}")
