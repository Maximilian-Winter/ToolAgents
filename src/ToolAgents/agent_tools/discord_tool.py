import os
import datetime
import asyncio
from enum import Enum
from typing import List, Dict, Any, Optional, Union

import discord
from discord import app_commands
from discord.ext import commands
from pydantic import BaseModel, Field

from ToolAgents import FunctionTool


class MessageType(Enum):
    TEXT = "text"
    EMBED = "embed"


class DiscordPermissionLevel(Enum):
    ADMIN = "admin"
    MODERATOR = "moderator"
    MEMBER = "member"


class DiscordEmbedData(BaseModel):
    """Data for creating a Discord embed message"""
    title: Optional[str] = Field(None, description="Title of the embed")
    description: Optional[str] = Field(None, description="Description/content of the embed")
    color: Optional[int] = Field(None, description="Color of the embed in decimal format (e.g. 3447003 for blue)")
    url: Optional[str] = Field(None, description="URL the title will link to")
    image_url: Optional[str] = Field(None, description="URL of an image to include")
    thumbnail_url: Optional[str] = Field(None, description="URL of a thumbnail to include")
    footer_text: Optional[str] = Field(None, description="Footer text for the embed")
    author_name: Optional[str] = Field(None, description="Author name for the embed")
    author_url: Optional[str] = Field(None, description="URL for the author")
    author_icon_url: Optional[str] = Field(None, description="URL of the author's icon")
    fields: Optional[List[Dict[str, Any]]] = Field(None, description="List of field objects with name, value, and inline properties")


class DiscordClient:
    """A client for interacting with Discord"""
    
    def __init__(self, token: str, enable_privileged_intents: bool = False):
        self.token = token
        
        # Basic intents that don't require privileged access
        self.intents = discord.Intents.default()
        
        # Privileged intents - these require explicit approval in the Discord Developer Portal
        if enable_privileged_intents:
            self.intents.message_content = True  # Privileged: Requires verification
            self.intents.members = True          # Privileged: Requires verification
            print("WARNING: Using privileged intents. Make sure these are enabled in the Discord Developer Portal.")
        
        self.bot = commands.Bot(command_prefix="!", intents=self.intents)
        self.is_ready = False
        
        @self.bot.event
        async def on_ready():
            self.is_ready = True
            print(f"Discord bot logged in as {self.bot.user}")
    
    async def connect(self):
        """Connect the Discord bot"""
        if not self.is_ready:
            await self.bot.login(self.token)
            self.bot.loop.create_task(self.bot.connect())
            
            # Wait until the bot is ready
            while not self.is_ready:
                await asyncio.sleep(0.1)
    
    async def get_guild(self, guild_id: int):
        """Get a guild by ID"""
        return self.bot.get_guild(guild_id)
    
    async def get_channel(self, channel_id: int):
        """Get a channel by ID"""
        return self.bot.get_channel(channel_id)
    
    async def send_message(self, 
                           channel_id: int, 
                           content: str = None, 
                           embed_data: DiscordEmbedData = None):
        """Send a message to a channel"""
        channel = self.bot.get_channel(channel_id)
        
        if not channel:
            raise ValueError(f"Channel with ID {channel_id} not found")
        
        embed = None
        if embed_data:
            embed = discord.Embed(
                title=embed_data.title,
                description=embed_data.description,
                color=embed_data.color,
                url=embed_data.url
            )
            
            if embed_data.image_url:
                embed.set_image(url=embed_data.image_url)
            
            if embed_data.thumbnail_url:
                embed.set_thumbnail(url=embed_data.thumbnail_url)
            
            if embed_data.footer_text:
                embed.set_footer(text=embed_data.footer_text)
            
            if embed_data.author_name:
                embed.set_author(
                    name=embed_data.author_name,
                    url=embed_data.author_url,
                    icon_url=embed_data.author_icon_url
                )
            
            if embed_data.fields:
                for field in embed_data.fields:
                    embed.add_field(
                        name=field["name"],
                        value=field["value"],
                        inline=field.get("inline", False)
                    )
        
        return await channel.send(content=content, embed=embed)
    
    async def edit_message(self, 
                           channel_id: int, 
                           message_id: int,
                           content: str = None, 
                           embed_data: DiscordEmbedData = None):
        """Edit a message"""
        channel = self.bot.get_channel(channel_id)
        
        if not channel:
            raise ValueError(f"Channel with ID {channel_id} not found")
        
        try:
            message = await channel.fetch_message(message_id)
        except discord.NotFound:
            raise ValueError(f"Message with ID {message_id} not found")
        
        embed = None
        if embed_data:
            embed = discord.Embed(
                title=embed_data.title,
                description=embed_data.description,
                color=embed_data.color,
                url=embed_data.url
            )
            
            if embed_data.image_url:
                embed.set_image(url=embed_data.image_url)
            
            if embed_data.thumbnail_url:
                embed.set_thumbnail(url=embed_data.thumbnail_url)
            
            if embed_data.footer_text:
                embed.set_footer(text=embed_data.footer_text)
            
            if embed_data.author_name:
                embed.set_author(
                    name=embed_data.author_name,
                    url=embed_data.author_url,
                    icon_url=embed_data.author_icon_url
                )
            
            if embed_data.fields:
                for field in embed_data.fields:
                    embed.add_field(
                        name=field["name"],
                        value=field["value"],
                        inline=field.get("inline", False)
                    )
        
        return await message.edit(content=content, embed=embed)
    
    async def delete_message(self, channel_id: int, message_id: int):
        """Delete a message"""
        channel = self.bot.get_channel(channel_id)
        
        if not channel:
            raise ValueError(f"Channel with ID {channel_id} not found")
        
        try:
            message = await channel.fetch_message(message_id)
        except discord.NotFound:
            raise ValueError(f"Message with ID {message_id} not found")
        
        await message.delete()
        return {"success": True, "message": f"Message {message_id} deleted"}
    
    async def get_channels(self, guild_id: int):
        """Get all channels in a guild"""
        guild = self.bot.get_guild(guild_id)
        
        if not guild:
            raise ValueError(f"Guild with ID {guild_id} not found")
        
        channels = []
        for channel in guild.channels:
            channel_data = {
                "id": channel.id,
                "name": channel.name,
                "type": str(channel.type),
                "position": channel.position,
            }
            
            if hasattr(channel, "topic") and channel.topic:
                channel_data["topic"] = channel.topic
                
            channels.append(channel_data)
            
        return channels
    
    async def get_members(self, guild_id: int):
        """Get all members in a guild"""
        guild = self.bot.get_guild(guild_id)
        
        if not guild:
            raise ValueError(f"Guild with ID {guild_id} not found")
        
        members = []
        for member in guild.members:
            members.append({
                "id": member.id,
                "name": member.name,
                "display_name": member.display_name,
                "bot": member.bot,
                "roles": [role.name for role in member.roles[1:]],  # Skip @everyone
                "joined_at": member.joined_at.isoformat() if member.joined_at else None,
                "status": str(member.status)
            })
            
        return members
    
    async def create_channel(self, 
                            guild_id: int, 
                            name: str, 
                            channel_type: str, 
                            category_id: Optional[int] = None,
                            topic: Optional[str] = None):
        """Create a new channel"""
        guild = self.bot.get_guild(guild_id)
        
        if not guild:
            raise ValueError(f"Guild with ID {guild_id} not found")
        
        category = None
        if category_id:
            category = guild.get_channel(category_id)
            if not category or not isinstance(category, discord.CategoryChannel):
                raise ValueError(f"Category with ID {category_id} not found")
        
        # Determine channel type
        discord_channel_type = None
        if channel_type.lower() == "text":
            discord_channel_type = discord.ChannelType.text
        elif channel_type.lower() == "voice":
            discord_channel_type = discord.ChannelType.voice
        elif channel_type.lower() == "category":
            discord_channel_type = discord.ChannelType.category
        else:
            raise ValueError(f"Unsupported channel type: {channel_type}")
        
        channel_kwargs = {"name": name, "type": discord_channel_type}
        
        if category and discord_channel_type != discord.ChannelType.category:
            channel_kwargs["category"] = category
            
        if topic and discord_channel_type == discord.ChannelType.text:
            channel_kwargs["topic"] = topic
            
        channel = await guild.create_channel(**channel_kwargs)
        
        return {
            "id": channel.id,
            "name": channel.name,
            "type": str(channel.type),
            "position": channel.position,
            "topic": getattr(channel, "topic", None)
        }
    
    async def delete_channel(self, channel_id: int):
        """Delete a channel"""
        channel = self.bot.get_channel(channel_id)
        
        if not channel:
            raise ValueError(f"Channel with ID {channel_id} not found")
        
        await channel.delete()
        return {"success": True, "message": f"Channel {channel_id} deleted"}
    
    async def get_messages(self, channel_id: int, limit: int = 100):
        """Get recent messages from a channel"""
        channel = self.bot.get_channel(channel_id)
        
        if not channel:
            raise ValueError(f"Channel with ID {channel_id} not found")
        
        messages = []
        async for message in channel.history(limit=limit):
            messages.append({
                "id": message.id,
                "author": {
                    "id": message.author.id,
                    "name": message.author.name,
                    "display_name": getattr(message.author, "display_name", message.author.name),
                    "bot": message.author.bot
                },
                "content": message.content,
                "created_at": message.created_at.isoformat(),
                "edited_at": message.edited_at.isoformat() if message.edited_at else None,
                "attachments": [
                    {"url": attachment.url, "filename": attachment.filename}
                    for attachment in message.attachments
                ],
                "embeds": len(message.embeds) > 0,
                "pinned": message.pinned,
                "mentions": [user.name for user in message.mentions]
            })
            
        return messages
    
    async def add_reaction(self, channel_id: int, message_id: int, emoji: str):
        """Add a reaction to a message"""
        channel = self.bot.get_channel(channel_id)
        
        if not channel:
            raise ValueError(f"Channel with ID {channel_id} not found")
        
        try:
            message = await channel.fetch_message(message_id)
        except discord.NotFound:
            raise ValueError(f"Message with ID {message_id} not found")
        
        await message.add_reaction(emoji)
        return {"success": True, "message": f"Added reaction {emoji} to message {message_id}"}
    
    async def remove_reaction(self, channel_id: int, message_id: int, emoji: str):
        """Remove a bot's reaction from a message"""
        channel = self.bot.get_channel(channel_id)
        
        if not channel:
            raise ValueError(f"Channel with ID {channel_id} not found")
        
        try:
            message = await channel.fetch_message(message_id)
        except discord.NotFound:
            raise ValueError(f"Message with ID {message_id} not found")
        
        await message.remove_reaction(emoji, self.bot.user)
        return {"success": True, "message": f"Removed reaction {emoji} from message {message_id}"}
    
    async def get_guild_info(self, guild_id: int):
        """Get information about a guild"""
        guild = self.bot.get_guild(guild_id)
        
        if not guild:
            raise ValueError(f"Guild with ID {guild_id} not found")
        
        return {
            "id": guild.id,
            "name": guild.name,
            "description": guild.description,
            "owner": {
                "id": guild.owner.id,
                "name": guild.owner.name,
                "display_name": guild.owner.display_name
            } if guild.owner else None,
            "member_count": guild.member_count,
            "created_at": guild.created_at.isoformat(),
            "roles": [
                {"id": role.id, "name": role.name, "color": str(role.color)}
                for role in guild.roles
            ],
            "emojis": [
                {"id": emoji.id, "name": emoji.name, "animated": emoji.animated}
                for emoji in guild.emojis
            ],
            "features": guild.features,
            "premium_tier": guild.premium_tier,
            "premium_subscription_count": guild.premium_subscription_count
        }
    
    async def create_role(self, 
                         guild_id: int, 
                         name: str, 
                         color: Optional[int] = None,
                         hoist: bool = False,
                         mentionable: bool = False):
        """Create a new role in a guild"""
        guild = self.bot.get_guild(guild_id)
        
        if not guild:
            raise ValueError(f"Guild with ID {guild_id} not found")
        
        role = await guild.create_role(
            name=name,
            color=discord.Color(color) if color else discord.Color.default(),
            hoist=hoist,
            mentionable=mentionable
        )
        
        return {
            "id": role.id,
            "name": role.name,
            "color": str(role.color),
            "hoist": role.hoist,
            "mentionable": role.mentionable,
            "position": role.position
        }


class SendDiscordMessage(BaseModel):
    """
    Send a message to a Discord channel
    """
    guild_id: int = Field(..., description="ID of the Discord server/guild")
    channel_id: int = Field(..., description="ID of the channel to send the message to")
    message_type: MessageType = Field(MessageType.TEXT, description="Type of message to send (text or embed)")
    content: Optional[str] = Field(None, description="Content of the message if it's a text message")
    embed_data: Optional[DiscordEmbedData] = Field(None, description="Embed data if message_type is embed")
    
    def run(self, discord_client):

        import asyncio
        loop = asyncio.get_event_loop()
        
        # Ensure the client is connected
        loop.run_until_complete(discord_client.connect())
        
        if self.message_type == MessageType.TEXT:
            if not self.content:
                raise ValueError("Content must be provided for text messages")
            
            message = loop.run_until_complete(
                discord_client.send_message(
                    channel_id=self.channel_id,
                    content=self.content
                )
            )
        else:  # MessageType.EMBED
            if not self.embed_data:
                raise ValueError("Embed data must be provided for embed messages")
            
            message = loop.run_until_complete(
                discord_client.send_message(
                    channel_id=self.channel_id,
                    content=self.content,
                    embed_data=self.embed_data
                )
            )
        
        return {
            "message_id": message.id,
            "channel_id": message.channel.id,
            "content": message.content,
            "has_embed": len(message.embeds) > 0,
            "created_at": message.created_at.isoformat()
        }


class GetDiscordChannels(BaseModel):
    """
    Get a list of channels in a Discord server
    """
    guild_id: int = Field(..., description="ID of the Discord server/guild")
    
    def run(self, discord_client):

        import asyncio
        loop = asyncio.get_event_loop()
        
        # Ensure the client is connected
        loop.run_until_complete(discord_client.connect())
        
        channels = loop.run_until_complete(
            discord_client.get_channels(guild_id=self.guild_id)
        )
        
        return {
            "guild_id": self.guild_id,
            "channels": channels
        }


class GetDiscordMessages(BaseModel):
    """
    Get recent messages from a Discord channel
    """
    guild_id: int = Field(..., description="ID of the Discord server/guild")
    channel_id: int = Field(..., description="ID of the channel to get messages from")
    limit: int = Field(50, description="Maximum number of messages to retrieve (max 100)")
    
    def run(self, discord_client):

        import asyncio
        loop = asyncio.get_event_loop()
        
        # Ensure the client is connected
        loop.run_until_complete(discord_client.connect())
        
        messages = loop.run_until_complete(
            discord_client.get_messages(
                channel_id=self.channel_id,
                limit=min(self.limit, 100)  # Cap at 100 messages
            )
        )
        
        return {
            "guild_id": self.guild_id,
            "channel_id": self.channel_id,
            "message_count": len(messages),
            "messages": messages
        }


class GetDiscordMembers(BaseModel):
    """
    Get a list of members in a Discord server
    """
    guild_id: int = Field(..., description="ID of the Discord server/guild")
    
    def run(self, discord_client):

        import asyncio
        loop = asyncio.get_event_loop()
        
        # Ensure the client is connected
        loop.run_until_complete(discord_client.connect())
        
        members = loop.run_until_complete(
            discord_client.get_members(guild_id=self.guild_id)
        )
        
        return {
            "guild_id": self.guild_id,
            "member_count": len(members),
            "members": members
        }


class CreateDiscordChannel(BaseModel):
    """
    Create a new channel in a Discord server
    """
    guild_id: int = Field(..., description="ID of the Discord server/guild")
    name: str = Field(..., description="Name of the channel to create")
    channel_type: str = Field(..., description="Type of channel to create (text, voice, category)")
    category_id: Optional[int] = Field(None, description="ID of the category to place the channel in")
    topic: Optional[str] = Field(None, description="Topic/description for text channels")
    
    def run(self, discord_client):

        import asyncio
        loop = asyncio.get_event_loop()
        
        # Ensure the client is connected
        loop.run_until_complete(discord_client.connect())
        
        channel = loop.run_until_complete(
            discord_client.create_channel(
                guild_id=self.guild_id,
                name=self.name,
                channel_type=self.channel_type,
                category_id=self.category_id,
                topic=self.topic
            )
        )
        
        return {
            "guild_id": self.guild_id,
            "channel": channel
        }


class GetDiscordGuildInfo(BaseModel):
    """
    Get information about a Discord server/guild
    """
    guild_id: int = Field(..., description="ID of the Discord server/guild")
    
    def run(self, discord_client):

        import asyncio
        loop = asyncio.get_event_loop()
        
        # Ensure the client is connected
        loop.run_until_complete(discord_client.connect())
        
        guild_info = loop.run_until_complete(
            discord_client.get_guild_info(guild_id=self.guild_id)
        )
        
        return guild_info


class AddDiscordReaction(BaseModel):
    """
    Add a reaction to a Discord message
    """
    guild_id: int = Field(..., description="ID of the Discord server/guild")
    channel_id: int = Field(..., description="ID of the channel containing the message")
    message_id: int = Field(..., description="ID of the message to react to")
    emoji: str = Field(..., description="Emoji to add as a reaction (Unicode emoji or custom emoji ID)")
    
    def run(self, discord_client):

        import asyncio
        loop = asyncio.get_event_loop()
        
        # Ensure the client is connected
        loop.run_until_complete(discord_client.connect())
        
        result = loop.run_until_complete(
            discord_client.add_reaction(
                channel_id=self.channel_id,
                message_id=self.message_id,
                emoji=self.emoji
            )
        )
        
        return result


# Global discord client instance
_discord_client = None

def get_discord_client(enable_privileged_intents: bool = False):
    """
    Get or create the global Discord client instance
    
    Args:
        enable_privileged_intents: Whether to enable privileged intents like message_content and members.
                                  These require verification and explicit approval in the Discord Developer Portal.
                                  See https://discord.com/developers/applications/ to enable these for your bot.
    """
    global _discord_client
    
    if _discord_client is None:
        token = os.getenv("DISCORD_BOT_TOKEN")
        if not token:
            raise ValueError("DISCORD_BOT_TOKEN environment variable not set")
        
        _discord_client = DiscordClient(token=token, enable_privileged_intents=enable_privileged_intents)
    
    return _discord_client


def init_discord_tools(enable_privileged_intents: bool = False):
    """
    Initialize Discord tools for the ToolAgents framework
    
    Args:
        enable_privileged_intents: Whether to enable privileged intents like message_content and members.
                                  These require verification and explicit approval in the Discord Developer Portal.
                                  See https://discord.com/developers/applications/ to enable these for your bot.
    
    Returns:
        List of FunctionTool instances for Discord operations
    """
    # Initialize the global Discord client with the specified intent settings
    discord_client = get_discord_client(enable_privileged_intents=enable_privileged_intents)
    
    # Create function tools
    send_message_tool = FunctionTool(SendDiscordMessage, discord_client=discord_client)
    get_channels_tool = FunctionTool(GetDiscordChannels, discord_client=discord_client)
    get_messages_tool = FunctionTool(GetDiscordMessages, discord_client=discord_client)
    get_members_tool = FunctionTool(GetDiscordMembers, discord_client=discord_client)
    create_channel_tool = FunctionTool(CreateDiscordChannel, discord_client=discord_client)
    get_guild_info_tool = FunctionTool(GetDiscordGuildInfo, discord_client=discord_client)
    add_reaction_tool = FunctionTool(AddDiscordReaction, discord_client=discord_client)
    
    return [
        send_message_tool,
        get_channels_tool,
        get_messages_tool,
        get_members_tool,
        create_channel_tool,
        get_guild_info_tool,
        add_reaction_tool
    ]