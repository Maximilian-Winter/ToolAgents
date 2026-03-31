import enum


class MessageType(str, enum.Enum):
    statement = "statement"
    proposal = "proposal"
    objection = "objection"
    consensus = "consensus"
    question = "question"
    answer = "answer"


class IssueState(str, enum.Enum):
    open = "open"
    closed = "closed"


class Priority(str, enum.Enum):
    critical = "critical"
    high = "high"
    medium = "medium"
    low = "low"
    none = "none"


class TaskStatus(str, enum.Enum):
    pending = "pending"
    active = "active"
    completed = "completed"
    cancelled = "cancelled"
