from ToolAgents.data_models.chat_history import ChatHistory, Chats
from ToolAgents.data_models.messages import (
    BinaryContent,
    BinaryStorageType,
    ChatMessage,
    ChatMessageRole,
    ContentType,
    StreamingChatMessage,
    TextContent,
    ToolCallContent,
    ToolCallResultContent,
)
from ToolAgents.utilities.message_template import MessageTemplate
from ToolAgents.utilities.prompt_builder import PromptBuilder, PromptLine, PromptPart, PromptVar

from .chat_history import AdvancedChatFormatter

try:
    from ToolAgents.utilities.chat_database import ChatManager
except ImportError:
    ChatDatabase = None
else:
    ChatDatabase = ChatManager

__all__ = [
    'AdvancedChatFormatter',
    'BinaryContent',
    'BinaryStorageType',
    'ChatDatabase',
    'ChatHistory',
    'ChatMessage',
    'ChatMessageRole',
    'Chats',
    'ContentType',
    'MessageTemplate',
    'PromptBuilder',
    'PromptLine',
    'PromptPart',
    'PromptVar',
    'StreamingChatMessage',
    'TextContent',
    'ToolCallContent',
    'ToolCallResultContent',
]
