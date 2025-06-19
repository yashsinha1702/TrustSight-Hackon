from datetime import datetime
from typing import Dict, Any, Optional
from pydantic import BaseModel, Field

class AppealRequest(BaseModel):
    """Pydantic model for appeal submission requests"""
    decision_id: str = Field(..., description="Decision ID to appeal")
    seller_id: str = Field(..., description="Seller ID submitting appeal")
    reason: str = Field(..., description="Reason for appeal")
    evidence: Dict[str, Any] = Field(default_factory=dict, description="Supporting evidence")

class DecisionQuery(BaseModel):
    """Pydantic model for decision query filters"""
    entity_id: Optional[str] = None
    entity_type: Optional[str] = None
    date_from: Optional[datetime] = None
    date_to: Optional[datetime] = None
    status: Optional[str] = None