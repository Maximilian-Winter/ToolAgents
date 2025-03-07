from enum import Enum
from types import NoneType
from typing import List

from pydantic import BaseModel, Field


class ToolType(Enum):
    python = "python310"
    python_313 = "python313"
    javascript = "javascript"
    c = "c"
    cpp = "cpp"
    executable = "executable"


class ToolDependencyType(Enum):
    python_package = "python_package"
    nodejs_package = "nodejs_package"
    shared_library = "shared_library"
    static_library = "static_library"

class ToolRuntime(Enum):
    function_tool = "function_tool"
    nodejs = "nodejs"

class ToolDependency(BaseModel):
    dependency_type: ToolDependencyType = Field(..., description="The dependency type.")
    dependency_value: str = Field(..., description="The dependency value. Name of a package or library.")


class ToolBuildConfiguration(BaseModel):
    docker_base_image: str = Field(..., description="The docker base image to run the tool from.")
    dependencies: List[ToolDependency] = Field(..., description="Tool build dependencies.")
    build_command: str = Field(..., description="The build command to build the tool.")


class ToolRuntimeConfiguration(BaseModel):
    docker_base_image: str = Field(..., description="The docker base image to run the tool from.")
    dependencies: List[ToolDependency] = Field(..., description="Tool runtime dependencies.")
    run_command: str = Field(..., description="The run command to execute the tool.")


class AgentTool(BaseModel):
    id: str = Field(..., description="Unique identifier for the tool.")
    name: str = Field(..., description="Name of the tool.")

    tool_type: ToolType = Field(..., description="Type of the tool.")
    tool_path: str = Field(..., description="Path to the file or folder defining the tool.")
    tool_runtime_configuration: ToolRuntimeConfiguration = Field(default_factory=NoneType, description="Tool runtime configuration.")
    tool_build_configuration: ToolBuildConfiguration = Field(default_factory=NoneType,
                                                             description="Tool build configuration.")


