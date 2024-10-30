import json
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Optional
from copy import deepcopy

from openai import OpenAI
from transformers import AutoTokenizer

from ToolAgents import ToolRegistry
from ToolAgents.interfaces.llm_provider import LLMSamplingSettings, HostedLLMProvider
from ToolAgents.interfaces.llm_tokenizer import LLMTokenizer


@dataclass
class VLLMServerSamplingSettings(LLMSamplingSettings):
    best_of: Optional[int] = None
    use_beam_search: bool = False
    top_k: float = -1
    top_p: float = 1
    min_p: float = 0.0
    temperature: float = 0.7
    max_tokens: int = 16
    repetition_penalty: Optional[float] = 1.0
    length_penalty: Optional[float] = 1.0
    early_stopping: Optional[bool] = False
    ignore_eos: Optional[bool] = False
    min_tokens: Optional[int] = 0
    stop_token_ids: Optional[List[int]] = field(default_factory=list)
    skip_special_tokens: Optional[bool] = True
    spaces_between_special_tokens: Optional[bool] = True
    stream: bool = False

    def save_to_file(self, settings_file: str):
        with open(settings_file, 'w') as f:
            json.dump(self.as_dict(), f, indent=2)

    def load_from_file(self, settings_file: str):
        with open(settings_file, 'r') as f:
            data = json.load(f)
        for key, value in data.items():
            setattr(self, key, value)

    def as_dict(self):
        return asdict(self)

    def set_stop_tokens(self, tokens: List[str], tokenizer: LLMTokenizer):
        self.stop_token_ids = [tokenizer.tokenize(token) for token in tokens]

    def set_max_new_tokens(self, max_new_tokens: int):
        self.max_tokens = max_new_tokens

    def set(self, setting_key: str, setting_value: str):
        if hasattr(self, setting_key):
            setattr(self, setting_key, setting_value)
        else:
            raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{setting_key}'")

    def neutralize_sampler(self, sampler_name: str):
        if sampler_name == "temperature":
            self.temperature = 1.0
        elif sampler_name == "top_k":
            self.top_k = -1
        elif sampler_name == "top_p":
            self.top_p = 1.0
        elif sampler_name == "min_p":
            self.min_p = 0.0
        else:
            raise ValueError(f"Unknown sampler: {sampler_name}")

    def neutralize_all_samplers(self):
        self.temperature = 1.0
        self.top_k = -1
        self.top_p = 1.0
        self.min_p = 0.0


class VLLMServerProvider(HostedLLMProvider):
    def __init__(self, base_url: str, model: str, huggingface_model: str, api_key: str = None):
        self.tokenizer = AutoTokenizer.from_pretrained(huggingface_model)
        self.client = OpenAI(
            base_url=base_url,
            api_key=api_key if api_key else "xxx-xxxxxxxx",
        )
        self.model = model

    def get_default_settings(self):
        return VLLMServerSamplingSettings()

    def create_completion(self, prompt: str, settings: VLLMServerSamplingSettings, tool_registry: ToolRegistry = None):
        return self._create_completion_or_chat(prompt=prompt, settings=settings)

    def create_chat_completion(self, messages: List[Dict[str, str]], settings: VLLMServerSamplingSettings, tool_registry: ToolRegistry = None):
        return self._create_completion_or_chat(messages=messages, settings=settings)

    def _create_completion_or_chat(self, settings: VLLMServerSamplingSettings, prompt: str = None,
                                   messages: List[Dict[str, str]] = None, tool_registry: ToolRegistry = None):
        settings = deepcopy(settings.as_dict())
        is_chat = messages is not None
        grammar = None
        if tool_registry is not None and tool_registry.guided_sampling_enabled:
            grammar = tool_registry.get_guided_sampling_json_schema()
        completion_settings = {
            "model": self.model,
            "top_p": settings.get('top_p'),
            "stream": settings.get('stream', False),
            "temperature": settings.get('temperature'),
            "max_tokens": settings.get('max_tokens'),
        }

        extra_body = deepcopy(settings)
        for key in ['top_p', 'stream', 'temperature', 'max_tokens']:
            extra_body.pop(key, None)

        if grammar is not None:
            extra_body["guided_json"] = grammar

        completion_settings['extra_body'] = extra_body

        if is_chat:
            completion_settings['messages'] = messages
            method = self.client.chat.completions.create
        else:
            completion_settings['prompt'] = prompt
            method = self.client.completions.create

        if settings.get('stream', False):
            result = method(**completion_settings)

            def generate_chunks(res):
                for chunk in res:
                    if is_chat:
                        content = chunk.choices[0].delta.content
                    else:
                        content = chunk.choices[0].text
                    if content is not None:
                        yield {"choices": [{"text": content}]}

            return generate_chunks(result)
        else:
            result = method(**completion_settings)
            if is_chat:
                content = result.choices[0].message.content
            else:
                content = result.choices[0].text
            return {"choices": [{"text": content}]}
