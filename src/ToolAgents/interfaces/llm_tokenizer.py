import abc
from typing import List, Dict, Any

from transformers import AutoTokenizer


from ToolAgents.function_tool import ToolRegistry
from ToolAgents.utilities.chat_history import AdvancedChatFormatter


class LLMTokenizer(abc.ABC):
    @abc.abstractmethod
    def apply_template(self, messages: List[Dict[str, str]], tools: ToolRegistry) -> str:
        pass

    @abc.abstractmethod
    def tokenize(self, text: str) -> List[int]:
        pass


class HuggingFaceTokenizer(LLMTokenizer):
    def __init__(self, huggingface_tokenizer_model: str):
        self.tokenizer = AutoTokenizer.from_pretrained(huggingface_tokenizer_model)

    def apply_template(self, messages: List[Dict[str, str]], tools: ToolRegistry) -> str:
        return self.tokenizer.apply_chat_template(
            conversation=messages,
            tools=tools.get_openai_tools() if len(tools.tools) > 0 else None,
            tokenize=False,
            add_generation_prompt=True
        )

    def tokenize(self, text: str) -> List[int]:
        return self.tokenizer.encode(text, add_special_tokens=False)


class TemplateTokenizer(LLMTokenizer):
    def __init__(self, advanced_chat_formatter: AdvancedChatFormatter, generation_prompt: str = None):
        self.chat_formatter = advanced_chat_formatter
        self.generation_prompt = generation_prompt

    def apply_template(self, messages: List[Dict[str, str]], tools: ToolRegistry) -> str:
        return self.chat_formatter.format_messages(messages=messages, tools=tools.get_tools()) + (self.generation_prompt if self.generation_prompt else "")

    def tokenize(self, text: str) -> List[int]:
        raise NotImplemented()
