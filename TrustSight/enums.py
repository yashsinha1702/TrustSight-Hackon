from enum import Enum

class Priority(Enum):
    """Priority levels for fraud detection request processing"""
    CRITICAL = 1
    HIGH = 2
    MEDIUM = 3
    LOW = 4
    SKIP = 5