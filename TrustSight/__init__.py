"""
TrustSight Integration Layer - The Orchestration Engine
Unifies all detection models, handles Kafka streams, and routes to Cross Intelligence
"""

import logging

logging.basicConfig(level=logging.INFO)

from .config import IntegrationConfig
from .enums import Priority
from .models import DetectionRequest, DetectionResult, IntegratedDetectionResult
from .detector_interfaces import (
    DetectorInterface,
    ReviewFraudDetectorInterface,
    SellerNetworkDetectorInterface,
    ListingFraudDetectorInterface,
    CounterfeitDetectorInterface
)
from .integration_engine import TrustSightIntegrationEngine
from .kafka_processor import KafkaStreamProcessor
from .api_models import DetectionRequestAPI, BatchDetectionRequestAPI
from .api import app

__all__ = [
    'IntegrationConfig',
    'Priority',
    'DetectionRequest',
    'DetectionResult',
    'IntegratedDetectionResult',
    'DetectorInterface',
    'ReviewFraudDetectorInterface',
    'SellerNetworkDetectorInterface',
    'ListingFraudDetectorInterface',
    'CounterfeitDetectorInterface',
    'TrustSightIntegrationEngine',
    'KafkaStreamProcessor',
    'DetectionRequestAPI',
    'BatchDetectionRequestAPI',
    'app'
]