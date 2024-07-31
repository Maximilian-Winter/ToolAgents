from dataclasses import dataclass

import requests
from typing import List, Dict, Union
from copy import deepcopy
import json


@dataclass
class LlamaCppSamplingSettings:
    temperature: float = 0.3
    top_k: int = 0
    top_p: float = 1.0
    min_p: float = 0.0
    max_tokens: int = -1
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

    def is_streaming(self):
        return self.stream

    @staticmethod
    def load_from_dict(settings: dict) -> "LlamaCppSamplingSettings":
        return LlamaCppSamplingSettings(**settings)

    def as_dict(self) -> dict:
        return self.__dict__

    def to_dict(self) -> dict:
        return {
            'temperature': self.temperature,
            'top_k': self.top_k,
            'top_p': self.top_p,
            'min_p': self.min_p,
            'max_tokens': self.max_tokens,
            'n_keep': self.n_keep,
            'stream': self.stream,
            'stop': self.stop,
            'tfs_z': self.tfs_z,
            'typical_p': self.typical_p,
            'repeat_penalty': self.repeat_penalty,
            'repeat_last_n': self.repeat_last_n,
            'penalize_nl': self.penalize_nl,
            'presence_penalty': self.presence_penalty,
            'frequency_penalty': self.frequency_penalty,
            'penalty_prompt': self.penalty_prompt,
            'mirostat_mode': self.mirostat_mode,
            'mirostat_tau': self.mirostat_tau,
            'mirostat_eta': self.mirostat_eta,
            'cache_prompt': self.cache_prompt,
            'seed': self.seed,
            'ignore_eos': self.ignore_eos,
            'samplers': self.samplers
        }


class LlamaCppServerProvider:
    def __init__(self, server_address: str, api_key: str = None):
        self.server_address = server_address
        self.server_completion_endpoint = f"{self.server_address}/completion"
        self.server_chat_completion_endpoint = f"{self.server_address}/v1/chat/completions"
        self.server_tokenize_endpoint = f"{self.server_address}/tokenize"
        self.api_key = api_key

    def get_default_settings(self):
        return LlamaCppSamplingSettings()

    def create_completion(self, prompt: str, settings: LlamaCppSamplingSettings):
        settings = deepcopy(settings.as_dict())
        headers = self._get_headers()
        data = self._prepare_data(settings, prompt=prompt)

        if settings.get('stream', False):
            return self._get_response_stream(headers, data, self.server_completion_endpoint)

        response = requests.post(self.server_completion_endpoint, headers=headers, json=data)
        data = response.json()
        return {"choices": [{"text": data["content"]}]}

    def create_chat_completion(self, messages: List[Dict[str, str]], settings: LlamaCppSamplingSettings):
        settings = deepcopy(settings.as_dict())
        headers = self._get_headers()
        data = self._prepare_data(settings, messages=messages)

        if settings.get('stream', False):
            return self._get_response_stream(headers, data, self.server_chat_completion_endpoint)

        response = requests.post(self.server_chat_completion_endpoint, headers=headers, json=data)
        return response.json()

    def tokenize(self, prompt: str) -> list[int]:
        headers = self._get_headers()
        response = requests.post(self.server_tokenize_endpoint, headers=headers, json={"content": prompt})
        if response.status_code == 200:
            return response.json()["tokens"]
        else:
            raise Exception(
                f"Tokenization request failed. Status code: {response.status_code}\nResponse: {response.text}")

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
