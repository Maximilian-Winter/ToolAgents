from ToolAgents.agents.hosted_tool_agent import HostedToolAgent

from ToolAgents.interfaces import HostedLLMProvider

from ToolAgents.agents.mistral_agent_parts import MistralTokenizer, MistralToolCallHandler, MistralTokenizerVersion
from ToolAgents.agents.nous_hermes_pro_agent_parts import NousHermesProTokenizer, NousHermesProToolCallHandler
from ToolAgents.agents.llama31_agent_parts import Llama31Tokenizer, Llama31ToolCallHandler
from ToolAgents.interfaces.llm_tokenizer import TemplateTokenizer
from ToolAgents.interfaces.llm_tool_call import LLMToolCallHandler, TemplateToolCallHandler
from ToolAgents.utilities.chat_history import AdvancedChatFormatter


class TemplateAgent(HostedToolAgent):

    def __init__(self, provider: HostedLLMProvider, advanced_chat_formatter: AdvancedChatFormatter,
                 tool_call_handler: LLMToolCallHandler = None,
                 generation_prompt: str = None, debug_output: bool = False):
        super().__init__(provider, TemplateTokenizer(advanced_chat_formatter, generation_prompt=generation_prompt),
                         tool_call_handler if tool_call_handler is not None else TemplateToolCallHandler(debug_mode=debug_output),
                         debug_output)

    def last_response_contains_tool_calls(self) -> bool:
        pass


class MistralAgent(HostedToolAgent):
    def last_response_contains_tool_calls(self) -> bool:
        pass

    def __init__(self, provider: HostedLLMProvider, tokenizer_file: str = None, tokenizer_version: MistralTokenizerVersion = MistralTokenizerVersion.v7, debug_output: bool = False):
        super().__init__(provider, MistralTokenizer(tokenizer_file), MistralToolCallHandler(debug_output),
                         debug_output)


class NousHermesProAgent(HostedToolAgent):
    def last_response_contains_tool_calls(self) -> bool:
        pass

    def __init__(self, provider: HostedLLMProvider, debug_output: bool = False):
        super().__init__(provider, NousHermesProTokenizer(), NousHermesProToolCallHandler(debug_output),
                         debug_output)


class Llama31Agent(HostedToolAgent):
    def last_response_contains_tool_calls(self) -> bool:
        pass

    def __init__(self, provider, debug_output=False):
        super().__init__(provider=provider, tokenizer=Llama31Tokenizer(),
                         tool_call_handler=Llama31ToolCallHandler(debug_output),
                         debug_output=debug_output)
