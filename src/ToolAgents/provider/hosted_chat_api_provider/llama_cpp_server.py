from dataclasses import dataclass, asdict
from typing import List, Dict, Union
import json
import requests
from copy import deepcopy

from ToolAgents import ToolRegistry
from ToolAgents.interfaces.llm_provider import LLMSamplingSettings, HostedLLMProvider
from ToolAgents.interfaces.llm_tokenizer import LLMTokenizer


@dataclass
class LlamaCppSamplingSettings(LLMSamplingSettings):
    temperature: float = 1.0
    top_k: int = 0
    top_p: float = 1.0
    min_p: float = 0.0
    n_predict: int = -1
    n_keep: int = 0
    stream: bool = True
    stop: List[str] = None
    tfs_z: float = 1.0
    typical_p: float = 1.0
    repeat_penalty: float = 1.1
    repeat_last_n: int = -1
    penalize_nl: bool = False
    presence_penalty: float = 0.0
    frequency_penalty: float = 0.0
    penalty_prompt: Union[None, str, List[int]] = None
    mirostat_mode: int = 0
    mirostat_tau: float = 5.0
    mirostat_eta: float = 0.1
    cache_prompt: bool = True
    seed: int = -1
    ignore_eos: bool = False
    samplers: List[str] = None

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

    def set_stop_tokens(self, tokens: List[str], tokenizer: LLMTokenizer = None):
        self.stop = tokens

    def set_max_new_tokens(self, max_new_tokens: int):
        self.n_predict = max_new_tokens

    def set(self, setting_key: str, setting_value: str):
        if hasattr(self, setting_key):
            setattr(self, setting_key, setting_value)
        else:
            raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{setting_key}'")

    def neutralize_sampler(self, sampler_name: str):
        if sampler_name == "temperature":
            self.temperature = 1.0
        elif sampler_name == "top_k":
            self.top_k = 0
        elif sampler_name == "top_p":
            self.top_p = 1.0
        elif sampler_name == "min_p":
            self.min_p = 0.0
        elif sampler_name == "tfs_z":
            self.tfs_z = 1.0
        elif sampler_name == "typical_p":
            self.typical_p = 1.0
        else:
            raise ValueError(f"Unknown sampler: {sampler_name}")

    def neutralize_all_samplers(self):
        self.temperature = 1.0
        self.top_k = 0
        self.top_p = 1.0
        self.min_p = 0.0
        self.tfs_z = 1.0
        self.typical_p = 1.0


class LlamaCppServerProvider(HostedLLMProvider):
    def __init__(self, server_address: str, api_key: str = None):
        self.server_address = server_address
        self.server_completion_endpoint = f"{self.server_address}/completion"
        self.server_chat_completion_endpoint = f"{self.server_address}/v1/chat/completions"
        self.server_tokenize_endpoint = f"{self.server_address}/tokenize"
        self.api_key = api_key
        self.default_settings = LlamaCppSamplingSettings()

    def get_default_settings(self) -> LlamaCppSamplingSettings:
        return self.default_settings

    def set_default_settings(self, settings: LlamaCppSamplingSettings):
        self.default_settings = settings

    def create_completion(self, prompt: str, settings: LlamaCppSamplingSettings, tool_registry: ToolRegistry = None):
        settings = deepcopy(settings.as_dict())
        headers = self._get_headers()
        data = self._prepare_data(settings, prompt=prompt)

        if tool_registry is not None and tool_registry.guided_sampling_enabled:
            data["grammar"] = tool_registry.get_guided_sampling_grammar()

        if settings.get('stream', False):
            return self._get_response_stream(headers, data, self.server_completion_endpoint)

        response = requests.post(self.server_completion_endpoint, headers=headers, json=data)
        data = response.json()
        return {"choices": [{"text": data["content"]}]}

    def create_chat_completion(self, messages: List[Dict[str, str]], settings: LlamaCppSamplingSettings, tool_registry: ToolRegistry = None):
        settings = deepcopy(settings.as_dict())
        headers = self._get_headers()
        data = self._prepare_data(settings, messages=messages)

        if tool_registry is not None and tool_registry.guided_sampling_enabled:
            data["grammar"] = tool_registry.get_guided_sampling_grammar()

        if settings.get('stream', False):
            return self._get_response_stream(headers, data, self.server_chat_completion_endpoint)

        response = requests.post(self.server_chat_completion_endpoint, headers=headers, json=data)
        return response.json()

    def _get_headers(self):
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers

    def _prepare_data(self, settings, **kwargs):
        data = deepcopy(settings)
        data.update(kwargs)

        # Adjust some key names to match the API expectations
        if 'mirostat_mode' in data:
            data['mirostat'] = data.pop('mirostat_mode')
        if 'additional_stop_sequences' in data:
            data['stop'] = data.pop('additional_stop_sequences')
        if 'max_tokens' in data:
            data['n_predict'] = data.pop('max_tokens')

        # Set default samplers if not provided
        if 'samplers' not in data or data['samplers'] is None:
            data['samplers'] = ["top_k", "tfs_z", "typical_p", "top_p", "min_p", "temperature"]

        return data

    def _get_response_stream(self, headers, data, endpoint_address):
        response = requests.post(endpoint_address, headers=headers, json=data, stream=True)
        response.raise_for_status()

        def generate_text_chunks():
            decoded_chunk = ""
            for chunk in response.iter_lines():
                if chunk:
                    decoded_chunk += chunk.decode("utf-8")
                    if decoded_chunk.strip().startswith("error:"):
                        raise RuntimeError(decoded_chunk)
                    new_data = json.loads(decoded_chunk.replace("data:", ""))
                    returned_data = {
                        "choices": [{"text": new_data["content"]}]} if "choices" not in new_data else new_data
                    yield returned_data
                    decoded_chunk = ""

        return generate_text_chunks()
