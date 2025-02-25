import dataclasses
from typing import Union

from pydantic import BaseModel

from ToolAgents.agents import ChatToolAgent
from ToolAgents.messages import MessageTemplate, ChatMessage
from ToolAgents.provider import ProviderSettings

@dataclasses.dataclass
class TextTransformer:
    agent: ChatToolAgent
    settings: ProviderSettings
    prompt_template: MessageTemplate
    system_prompt: str = None

    def transform(self, document: str, **kwargs) -> str:
        if self.system_prompt is None:
            messages = [ChatMessage.create_user_message(self.prompt_template.generate_message_content(document=document, **kwargs))]
        else:
            messages = [
                ChatMessage.create_system_message(self.system_prompt),
                ChatMessage.create_user_message(self.prompt_template.generate_message_content(document=document, **kwargs))
            ]

        response = self.agent.get_response(messages=messages, settings=self.settings)
        return response.response