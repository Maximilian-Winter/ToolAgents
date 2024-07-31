from dataclasses import dataclass, field
from typing import List, Dict, Optional
from copy import deepcopy
from openai import OpenAI
from transformers import AutoTokenizer


@dataclass
class VLLMServerSamplingSettings:
    """
    VLLMServerSamplingSettings dataclass
    """

    best_of: Optional[int] = None
    use_beam_search = False
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

    def is_streaming(self):
        return self.stream

    @staticmethod
    def load_from_dict(settings: dict) -> "VLLMServerSamplingSettings":
        """
        Load the settings from a dictionary.

        Args:
            settings (dict): The dictionary containing the settings.

        Returns:
            LlamaCppSamplingSettings: The loaded settings.
        """
        return VLLMServerSamplingSettings(**settings)

    def as_dict(self) -> dict:
        """
        Convert the settings to a dictionary.

        Returns:
            dict: The dictionary representation of the settings.
        """
        return self.__dict__


class VLLMServerProvider:
    def __init__(self, base_url: str, model: str, huggingface_model: str, api_key: str = None):
        self.tokenizer = AutoTokenizer.from_pretrained(huggingface_model)
        self.client = OpenAI(
            base_url=base_url,
            api_key=api_key if api_key else "xxx-xxxxxxxx",
        )
        self.model = model

    def get_default_settings(self):
        return VLLMServerSamplingSettings()

    def create_completion(self, prompt: str, settings: VLLMServerSamplingSettings):
        return self._create_completion_or_chat(prompt=prompt, settings=settings)

    def create_chat_completion(self, messages: List[Dict[str, str]], settings: VLLMServerSamplingSettings):
        return self._create_completion_or_chat(messages=messages, settings=settings)

    def _create_completion_or_chat(self, settings: VLLMServerSamplingSettings, prompt: str = None,
                                   messages: List[Dict[str, str]] = None):
        settings = deepcopy(settings.as_dict())
        is_chat = messages is not None

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

        completion_settings['extra_body'] = extra_body

        if is_chat:
            completion_settings['messages'] = messages
            method = self.client.chat.completions.create
        else:
            completion_settings['prompt'] = prompt
            method = self.client.completions.create

        if settings.get('stream', False):
            result = method(**completion_settings)

            def generate_chunks():
                for chunk in result:
                    if is_chat:
                        content = chunk.choices[0].delta.content
                    else:
                        content = chunk.choices[0].text
                    if content is not None:
                        yield {"choices": [{"text": content}]}

            return generate_chunks()
        else:
            result = method(**completion_settings)
            if is_chat:
                content = result.choices[0].message.content
            else:
                content = result.choices[0].text
            return {"choices": [{"text": content}]}

    def tokenize(self, prompt: str) -> list[int]:
        return self.tokenizer.encode(text=prompt)
