from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, Any, Optional
from uuid import uuid4
from .enums import DecisionStatus, VerificationMethod
from .config import MemoryBankConfig

@dataclass
class Decision:
    """Data model for fraud detection decisions"""
    decision_id: str = field(default_factory=lambda: str(uuid4()))
    timestamp: datetime = field(default_factory=datetime.utcnow)
    detection_type: str = ""
    entity_type: str = ""
    entity_id: str = ""
    fraud_score: float = 0.0
    confidence_score: float = 0.0
    evidence: Dict[str, Any] = field(default_factory=dict)
    status: DecisionStatus = DecisionStatus.PENDING_VERIFICATION
    action_taken: str = ""
    ttl_days: int = 30
    
    def calculate_ttl(self, config: MemoryBankConfig) -> int:
        if self.confidence_score >= 0.95:
            return config.TTL_HIGH_CONFIDENCE
        elif self.confidence_score >= 0.80:
            return config.TTL_MEDIUM_CONFIDENCE
        else:
            return config.TTL_LOW_CONFIDENCE

@dataclass
class VerificationSignal:
    """Data model for signals used in decision verification"""
    method: VerificationMethod
    timestamp: datetime = field(default_factory=datetime.utcnow)
    result: str = ""
    confidence: float = 0.0
    details: Dict[str, Any] = field(default_factory=dict)

@dataclass
class Appeal:
    """Data model for seller appeals against fraud decisions"""
    appeal_id: str = field(default_factory=lambda: str(uuid4()))
    decision_id: str = ""
    seller_id: str = ""
    timestamp: datetime = field(default_factory=datetime.utcnow)
    reason: str = ""
    evidence: Dict[str, Any] = field(default_factory=dict)
    status: str = "pending"
    outcome: Optional[str] = None