from enum import Enum
from types import NoneType
from typing import List

from pydantic import BaseModel, Field

from ToolAgents.data_models.tools import AgentTool
from ToolAgents.data_models.messages import ChatMessage


class Agent(BaseModel):
    id: str = Field(..., description="Unique identifier for the agent.")
    name: str = Field(..., description="Name of the agent.")
    description: str = Field(..., description="Description of the agent.")
    system_message: ChatMessage = Field(..., description="System message of the agent.")
    tools: List[AgentTool] = Field(..., description="Tools associated with the agent.")