from dataclasses import dataclass, field

import requests
import json
from typing import List, Dict, Optional
from copy import deepcopy


@dataclass
class TGIServerSamplingSettings:
    """
    TGIServerSamplingSettings dataclass
    """

    best_of: Optional[int] = field(default=None, metadata={"minimum": 0})
    decoder_input_details: bool = False
    details: bool = True
    do_sample: bool = False
    frequency_penalty: Optional[float] = field(
        default=None, metadata={"exclusiveMinimum": -2}
    )
    grammar: Optional[dict] = None
    max_new_tokens: Optional[int] = field(default=None, metadata={"minimum": 0})
    repetition_penalty: Optional[float] = field(
        default=None, metadata={"exclusiveMinimum": 0}
    )
    return_full_text: Optional[bool] = field(default=None)
    seed: Optional[int] = field(default=None, metadata={"minimum": 0})
    stop: Optional[List[str]] = field(default_factory=list)
    temperature: Optional[float] = field(default=None, metadata={"exclusiveMinimum": 0})
    top_k: Optional[int] = field(default=None, metadata={"exclusiveMinimum": 0})
    top_n_tokens: Optional[int] = field(
        default=None, metadata={"minimum": 0, "exclusiveMinimum": 0}
    )
    top_p: Optional[float] = field(
        default=None, metadata={"maximum": 1, "exclusiveMinimum": 0}
    )
    truncate: Optional[int] = field(default=None, metadata={"minimum": 0})
    typical_p: Optional[float] = field(
        default=None, metadata={"maximum": 1, "exclusiveMinimum": 0}
    )
    watermark: bool = False
    stream: bool = False

    def is_streaming(self):
        return self.stream

    @staticmethod
    def load_from_dict(settings: dict) -> "TGIServerSamplingSettings":
        """
        Load the settings from a dictionary.

        Args:
            settings (dict): The dictionary containing the settings.

        Returns:
            LlamaCppSamplingSettings: The loaded settings.
        """
        return TGIServerSamplingSettings(**settings)

    def as_dict(self) -> dict:
        """
        Convert the settings to a dictionary.

        Returns:
            dict: The dictionary representation of the settings.
        """
        return self.__dict__

class TGIServerProvider:
    def __init__(self, server_address: str, api_key: str = None):
        self.server_address = server_address
        self.server_completion_endpoint = f"{self.server_address}/generate"
        self.server_streaming_completion_endpoint = f"{self.server_address}/generate_stream"
        self.server_chat_completion_endpoint = f"{self.server_address}/v1/chat/completions"
        self.server_tokenize_endpoint = f"{self.server_address}/tokenize"
        self.api_key = api_key

    def get_provider_default_settings(self):
        return TGIServerSamplingSettings()

    def create_completion(self, prompt: str, settings: TGIServerSamplingSettings):
        settings = deepcopy(settings.as_dict())
        headers = self._get_headers()
        data = {
            "parameters": settings,
            "inputs": prompt
        }

        if settings.get('stream', False):
            return self._get_response_stream(headers, data, self.server_streaming_completion_endpoint)

        response = requests.post(self.server_completion_endpoint, headers=headers, json=data)
        data = response.json()
        return {"choices": [{"text": data["generated_text"]}]}

    def create_chat_completion(self, messages: List[Dict[str, str]], settings: TGIServerSamplingSettings):
        settings = deepcopy(settings.as_dict())
        headers = self._get_headers()
        data = deepcopy(settings)
        data["messages"] = messages
        data["model"] = "tgi"

        if settings.get('stream', False):
            return self._get_response_stream(headers, data, self.server_chat_completion_endpoint)

        response = requests.post(self.server_chat_completion_endpoint, headers=headers, json=data)
        return response.json()

    def tokenize(self, prompt: str) -> list[int]:
        headers = self._get_headers()
        response = requests.post(self.server_tokenize_endpoint, headers=headers, json={"inputs": prompt})
        if response.status_code == 200:
            tokens = response.json()
            return [tok["id"] for tok in tokens]
        else:
            raise Exception(
                f"Tokenization request failed. Status code: {response.status_code}\nResponse: {response.text}")

    def _get_headers(self):
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers

    def _get_response_stream(self, headers, data, endpoint_address):
        response = requests.post(endpoint_address, headers=headers, json=data, stream=True)
        response.raise_for_status()

        def generate_text_chunks():
            try:
                for chunk in response.iter_lines():
                    if chunk:
                        decoded_chunk = chunk.decode("utf-8")
                        new_data = json.loads(decoded_chunk.replace("data:", ""))
                        if "token" in new_data and new_data["token"]["text"] is not None:
                            yield {"choices": [{"text": new_data["token"]["text"]}]}
                        elif "choices" in new_data and new_data["choices"][0]["delta"] is not None and "content" in \
                                new_data["choices"][0]["delta"]:
                            yield {"choices": [{"text": new_data["choices"][0]["delta"]["content"]}]}
            except requests.exceptions.RequestException as e:
                print(f"Request failed: {e}")

        return generate_text_chunks()
