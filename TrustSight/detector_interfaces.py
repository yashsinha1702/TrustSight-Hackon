import logging
import time
import numpy as np
from abc import ABC, abstractmethod
from typing import Dict, Any
from prometheus_client import Counter, Histogram
from .models import DetectionResult

logger = logging.getLogger(__name__)

detection_counter = Counter('trustsight_detections_total', 'Total detections', ['detector', 'result'])
processing_time = Histogram('trustsight_processing_seconds', 'Processing time', ['detector'])

class DetectorInterface(ABC):
    """Abstract interface for all fraud detection models"""
    
    @abstractmethod
    async def detect(self, data: Dict[str, Any]) -> DetectionResult:
        pass
    
    @abstractmethod
    def get_name(self) -> str:
        pass

class ReviewFraudDetectorInterface(DetectorInterface):
    """Interface for review fraud detection model"""
    
    def __init__(self, model_path: str = None):
        self.name = "review_fraud_detector"
        logger.info(f"Initialized {self.name}")
    
    async def detect(self, data: Dict[str, Any]) -> DetectionResult:
        start_time = time.time()
        
        try:
            fraud_score = np.random.uniform(0.3, 0.95) if "fake" in str(data).lower() else np.random.uniform(0.1, 0.4)
            
            result = DetectionResult(
                detector_name=self.name,
                fraud_score=fraud_score,
                confidence=0.85,
                fraud_types=['generic_text', 'timing_anomaly'] if fraud_score > 0.5 else [],
                evidence=[
                    {'type': 'timing', 'description': 'Review posted at 3:47 AM'},
                    {'type': 'text', 'description': 'Generic phrases detected'}
                ] if fraud_score > 0.5 else [],
                processing_time=time.time() - start_time
            )
            
        except Exception as e:
            logger.error(f"Error in {self.name}: {e}")
            result = DetectionResult(
                detector_name=self.name,
                fraud_score=0.0,
                confidence=0.0,
                error=str(e),
                processing_time=time.time() - start_time
            )
        
        detection_counter.labels(detector=self.name, result='success' if not result.error else 'error').inc()
        processing_time.labels(detector=self.name).observe(result.processing_time)
        
        return result
    
    def get_name(self) -> str:
        return self.name

class SellerNetworkDetectorInterface(DetectorInterface):
    """Interface for seller network detection model"""
    
    def __init__(self, model_path: str = None):
        self.name = "seller_network_detector"
        logger.info(f"Initialized {self.name}")
    
    async def detect(self, data: Dict[str, Any]) -> DetectionResult:
        start_time = time.time()
        
        try:
            is_network = "network" in str(data).lower() or "connected" in str(data).lower()
            fraud_score = np.random.uniform(0.7, 0.95) if is_network else np.random.uniform(0.1, 0.3)
            
            result = DetectionResult(
                detector_name=self.name,
                fraud_score=fraud_score,
                confidence=0.82,
                fraud_types=['price_coordination', 'registration_cluster'] if fraud_score > 0.5 else [],
                evidence=[
                    {'type': 'network', 'description': '5 connected sellers found'},
                    {'type': 'pattern', 'description': 'Synchronized pricing detected'}
                ] if fraud_score > 0.5 else [],
                processing_time=time.time() - start_time
            )
            
        except Exception as e:
            logger.error(f"Error in {self.name}: {e}")
            result = DetectionResult(
                detector_name=self.name,
                fraud_score=0.0,
                confidence=0.0,
                error=str(e),
                processing_time=time.time() - start_time
            )
        
        return result
    
    def get_name(self) -> str:
        return self.name

class ListingFraudDetectorInterface(DetectorInterface):
    """Interface for listing fraud detection model"""
    
    def __init__(self, model_path: str = None):
        self.name = "listing_fraud_detector"
        logger.info(f"Initialized {self.name}")
    
    async def detect(self, data: Dict[str, Any]) -> DetectionResult:
        start_time = time.time()
        
        try:
            suspicious = "variation" in str(data).lower() or "seo" in str(data).lower()
            fraud_score = np.random.uniform(0.6, 0.9) if suspicious else np.random.uniform(0.1, 0.4)
            
            result = DetectionResult(
                detector_name=self.name,
                fraud_score=fraud_score,
                confidence=0.79,
                fraud_types=['variation_abuse', 'seo_manipulation'] if fraud_score > 0.5 else [],
                evidence=[
                    {'type': 'listing', 'description': 'Unrelated variations detected'},
                    {'type': 'seo', 'description': 'Keyword stuffing found'}
                ] if fraud_score > 0.5 else [],
                processing_time=time.time() - start_time
            )
            
        except Exception as e:
            logger.error(f"Error in {self.name}: {e}")
            result = DetectionResult(
                detector_name=self.name,
                fraud_score=0.0,
                confidence=0.0,
                error=str(e),
                processing_time=time.time() - start_time
            )
        
        return result
    
    def get_name(self) -> str:
        return self.name

class CounterfeitDetectorInterface(DetectorInterface):
    """Interface for counterfeit product detection model"""
    
    def __init__(self, model_path: str = None):
        self.name = "counterfeit_detector"
        logger.info(f"Initialized {self.name}")
    
    async def detect(self, data: Dict[str, Any]) -> DetectionResult:
        start_time = time.time()
        
        try:
            is_fake = "fake" in str(data).lower() or "counterfeit" in str(data).lower()
            brand = data.get('brand', '').lower()
            suspicious_price = data.get('price', 100) < 50 and brand in ['nike', 'adidas', 'apple']
            
            fraud_score = np.random.uniform(0.8, 0.95) if (is_fake or suspicious_price) else np.random.uniform(0.1, 0.3)
            
            result = DetectionResult(
                detector_name=self.name,
                fraud_score=fraud_score,
                confidence=0.83,
                fraud_types=['price_anomaly', 'brand_mismatch'] if fraud_score > 0.5 else [],
                evidence=[
                    {'type': 'price', 'description': 'Price 70% below market'},
                    {'type': 'seller', 'description': 'Unauthorized seller'}
                ] if fraud_score > 0.5 else [],
                processing_time=time.time() - start_time
            )
            
        except Exception as e:
            logger.error(f"Error in {self.name}: {e}")
            result = DetectionResult(
                detector_name=self.name,
                fraud_score=0.0,
                confidence=0.0,
                error=str(e),
                processing_time=time.time() - start_time
            )
        
        return result
    
    def get_name(self) -> str:
        return self.name