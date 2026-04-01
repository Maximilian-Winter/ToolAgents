import json

from ToolAgents.provider import ChatAPIProvider, OpenAIChatAPI, AnthropicChatAPI, GroqChatAPI, MistralChatAPI
from virtual_game_master import VirtualGameMasterConfig


class VirtualGameMasterChatAPISelector:
    def __init__(self, config: VirtualGameMasterConfig):
        self.config = config

    def get_api(self) -> ChatAPIProvider:
        if self.config.API_TYPE == "openai":
            api = OpenAIChatAPI(api_key=self.config.API_KEY, base_url=self.config.API_URL, model=self.config.MODEL)
        elif self.config.API_TYPE == "anthropic":
            api = AnthropicChatAPI(self.config.API_KEY, self.config.MODEL)
        elif self.config.API_TYPE == "groq":
            api = GroqChatAPI(self.config.API_KEY, self.config.MODEL)
        elif self.config.API_TYPE == "mistral":
            api = MistralChatAPI(self.config.API_KEY, self.config.MODEL)
        else:
            raise ValueError(f"Unsupported API type: {self.config.API_TYPE}")

        # Set common settings
        api.settings.temperature = self.config.TEMPERATURE
        api.settings.top_p = self.config.TOP_P
        api.settings.max_tokens = self.config.MAX_TOKENS

        # Set additional settings for specific APIs
        if self.config.API_TYPE in ["openrouter", "openrouter_custom", "llamacpp", "llamacpp_custom"]:
            api.settings.top_k = self.config.TOP_K
            api.settings.min_p = self.config.MIN_P

        if self.config.API_TYPE in ["openrouter", "openrouter_custom"]:
            api.settings.stop = json.loads(self.config.STOP_SEQUENCES)

        if self.config.API_TYPE in ["llamacpp", "llamacpp_custom"]:
            api.settings.tfs_z = self.config.TFS_Z
            api.settings.additional_stop_sequences = json.loads(self.config.STOP_SEQUENCES)

        if self.config.API_TYPE == "groq":
            api.settings.stop = json.loads(self.config.STOP_SEQUENCES)

        return api
