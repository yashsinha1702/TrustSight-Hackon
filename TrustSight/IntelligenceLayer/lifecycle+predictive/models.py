from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Any
from .enums import LifecycleStage, FraudPattern, PredictionConfidence

@dataclass
class LifecycleEvent:
    """Data model for events in entity lifecycle progression"""
    event_id: str
    entity_type: str
    entity_id: str
    event_type: str
    timestamp: datetime
    metadata: Dict[str, Any] = field(default_factory=dict)
    anomaly_score: float = 0.0

@dataclass
class LifecycleProfile:
    """Data model for complete lifecycle profile of an entity"""
    entity_type: str
    entity_id: str
    current_stage: LifecycleStage
    creation_date: datetime
    events: List[LifecycleEvent] = field(default_factory=list)
    metrics: Dict[str, Any] = field(default_factory=dict)
    anomalies: List[Dict] = field(default_factory=list)
    risk_score: float = 0.0
    last_updated: datetime = field(default_factory=datetime.now)

@dataclass
class FraudPrediction:
    """Data model for prediction of future fraud patterns"""
    prediction_id: str
    entity_type: str
    entity_id: str
    fraud_pattern: FraudPattern
    probability: float
    confidence: PredictionConfidence
    predicted_timeframe: str
    risk_factors: List[str] = field(default_factory=list)
    recommended_actions: List[str] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.now)

@dataclass
class SeasonalPattern:
    """Data model for seasonal fraud patterns and characteristics"""
    pattern_name: str
    season: str
    fraud_multiplier: float
    common_tactics: List[str] = field(default_factory=list)
    high_risk_categories: List[str] = field(default_factory=list)