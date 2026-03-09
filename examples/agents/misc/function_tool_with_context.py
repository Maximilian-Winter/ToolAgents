from enum import Enum

from pydantic import BaseModel, Field


class FileOpenMode(Enum):
    READ = "read"
    WRITE = "write"
    APPEND = "append"

class FileTool(BaseModel):
    file_path: str = Field(..., description="Absolute File path")
    file_mode: FileOpenMode = Field(..., description="The Mode in which the file will be open. Options: read, write, append")

