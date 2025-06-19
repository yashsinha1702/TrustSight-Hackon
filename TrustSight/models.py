from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Any
from uuid import uuid4
from .enums import Priority

@dataclass
class DetectionRequest:
    """Data model for unified fraud detection requests"""
    request_id: str = field(default_factory=lambda: str(uuid4()))
    entity_type: str = ""
    entity_id: str = ""
    entity_data: Dict[str, Any] = field(default_factory=dict)
    priority: Priority = Priority.MEDIUM
    timestamp: datetime = field(default_factory=datetime.now)
    source: str = "api"
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class DetectionResult:
    """Data model for individual detector results"""
    detector_name: str
    fraud_score: float
    confidence: float
    fraud_types: List[str] = field(default_factory=list)
    evidence: List[Dict] = field(default_factory=list)
    processing_time: float = 0.0
    error: Optional[str] = None

@dataclass
class IntegratedDetectionResult:
    """Data model for combined results from all detectors"""
    request_id: str
    entity_type: str
    entity_id: str
    overall_fraud_score: float
    trust_score: float
    detection_results: Dict[str, DetectionResult] = field(default_factory=dict)
    cross_intel_triggered: bool = False
    cross_intel_result: Optional[Dict] = None
    recommendations: List[str] = field(default_factory=list)
    priority_actions: List[Dict] = field(default_factory=list)
    processing_time: float = 0.0
    timestamp: datetime = field(default_factory=datetime.now)