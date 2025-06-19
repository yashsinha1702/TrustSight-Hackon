from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Any
from uuid import uuid4
from enums import ActionPhase, AlertPriority, AlertRecipient

@dataclass
class ActionRequest:
    request_id: str = field(default_factory=lambda: str(uuid4()))
    fraud_type: str = ""
    entity_type: str = ""
    entity_id: str = ""
    fraud_score: float = 0.0
    confidence: float = 0.0
    evidence: Dict[str, Any] = field(default_factory=dict)
    network_size: int = 1
    affected_entities: List[str] = field(default_factory=list)
    decision_id: Optional[str] = None
    timestamp: datetime = field(default_factory=datetime.now)

@dataclass
class ActionResult:
    action_id: str = field(default_factory=lambda: str(uuid4()))
    request_id: str = ""
    actions_taken: List[Dict[str, Any]] = field(default_factory=list)
    phase: ActionPhase = ActionPhase.SOFT
    reversible: bool = True
    rollback_info: Optional[Dict] = None
    alerts_sent: List[Dict] = field(default_factory=list)
    trust_score_impact: float = 0.0
    success: bool = True
    error_message: Optional[str] = None
    timestamp: datetime = field(default_factory=datetime.now)

@dataclass
class TrustScore:
    overall_score: float = 100.0
    dimensions: Dict[str, float] = field(default_factory=lambda: {
        "review_authenticity": 100.0,
        "product_legitimacy": 100.0,
        "seller_reliability": 100.0,
        "pricing_fairness": 100.0,
        "network_isolation": 100.0
    })
    history: List[Dict] = field(default_factory=list)
    last_updated: datetime = field(default_factory=datetime.now)

@dataclass
class Alert:
    alert_id: str = field(default_factory=lambda: str(uuid4()))
    priority: AlertPriority = AlertPriority.MEDIUM
    recipient_type: AlertRecipient = AlertRecipient.INTERNAL_TEAM
    recipient_id: Optional[str] = None
    title: str = ""
    message: str = ""
    action_required: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)