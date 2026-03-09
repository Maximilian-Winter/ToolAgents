from ToolAgents.agents.chat_tool_agent import AsyncChatToolAgent, ChatToolAgent
from ToolAgents.agents.structured_output_agent import StructuredOutputAgent

__all__ = [
    'AdvancedAgent',
    'AgentConfig',
    'AsyncChatToolAgent',
    'ChatToolAgent',
    'StructuredOutputAgent',
]


def __getattr__(name: str):
    if name in {'AdvancedAgent', 'AgentConfig'}:
        from ToolAgents.agents.advanced_agent import AdvancedAgent, AgentConfig

        return {
            'AdvancedAgent': AdvancedAgent,
            'AgentConfig': AgentConfig,
        }[name]
    raise AttributeError(f'module {__name__!r} has no attribute {name!r}')
