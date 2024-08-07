from ToolAgents.agents.llm_tool_agent import LLMToolAgent

from ToolAgents.interfaces import LLMProvider

from ToolAgents.agents.mistral_agent_parts import MistralTokenizer, MistralToolCallHandler
from ToolAgents.agents.nous_hermes_pro_agent_parts import NousHermesProTokenizer, NousHermesProToolCallHandler
from ToolAgents.agents.llama31_agent_parts import Llama31Tokenizer, Llama31ToolCallHandler


class MistralAgent(LLMToolAgent):
    def __init__(self, llm_provider: LLMProvider, tokenizer_file: str = None, debug_output: bool = False):
        super().__init__(llm_provider, MistralTokenizer(tokenizer_file), MistralToolCallHandler(debug_output),
                         debug_output)


class NousHermesProAgent(LLMToolAgent):
    def __init__(self, llm_provider: LLMProvider, debug_output: bool = False):
        super().__init__(llm_provider, NousHermesProTokenizer(), NousHermesProToolCallHandler(debug_output),
                         debug_output)


class Llama31Agent(LLMToolAgent):

    def __init__(self, provider, debug_output=False):
        super().__init__(provider=provider, tokenizer=Llama31Tokenizer(), tool_call_handler=Llama31ToolCallHandler(),
                         debug_output=debug_output)
