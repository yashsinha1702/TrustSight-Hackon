import asyncio
import json
import logging
import pickle
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from typing import Dict, List, Optional, Any
import redis
from kafka import KafkaProducer
from prometheus_client import Gauge
from cross_intelligence import CrossIntelligenceEngine, FraudSignal, SignalType

from .config import IntegrationConfig
from .enums import Priority
from .models import DetectionRequest, DetectionResult, IntegratedDetectionResult
from .detector_interfaces import (
    ReviewFraudDetectorInterface,
    SellerNetworkDetectorInterface,
    ListingFraudDetectorInterface,
    CounterfeitDetectorInterface
)

logger = logging.getLogger(__name__)

active_investigations = Gauge('trustsight_active_investigations', 'Active investigations')

class TrustSightIntegrationEngine:
    """Central orchestration engine that coordinates all detectors and routes to Cross Intelligence"""
    
    def __init__(self, config: IntegrationConfig = None):
        self.config = config or IntegrationConfig()
        
        self._initialize_detectors()
        self._initialize_cross_intelligence()
        self._initialize_cache()
        self._initialize_kafka()
        self._initialize_priority_queue()
        
        self.executor = ThreadPoolExecutor(max_workers=self.config.MAX_PARALLEL_DETECTIONS)
        
        self.active_requests = {}
        self.completed_requests = deque(maxlen=1000)
        
        logger.info("TrustSight Integration Engine initialized successfully!")
    
    def _initialize_detectors(self):
        self.detectors = {
            'review': ReviewFraudDetectorInterface(self.config.MODEL_PATHS.get('review_fraud')),
            'seller': SellerNetworkDetectorInterface(self.config.MODEL_PATHS.get('seller_network')),
            'listing': ListingFraudDetectorInterface(self.config.MODEL_PATHS.get('listing_fraud')),
            'counterfeit': CounterfeitDetectorInterface(self.config.MODEL_PATHS.get('counterfeit'))
        }
        
        self.entity_detector_map = {
            'product': ['counterfeit', 'listing'],
            'review': ['review'],
            'seller': ['seller'],
            'listing': ['listing', 'counterfeit'],
            'all': ['review', 'seller', 'listing', 'counterfeit']
        }
    
    def _initialize_cross_intelligence(self):
        self.cross_intelligence = CrossIntelligenceEngine()
        logger.info("Cross Intelligence Engine connected")
    
    def _initialize_cache(self):
        try:
            self.redis_client = redis.Redis(
                host=self.config.REDIS_HOST,
                port=self.config.REDIS_PORT,
                db=self.config.REDIS_DB,
                decode_responses=False
            )
            self.redis_client.ping()
            logger.info("Redis cache connected")
        except Exception as e:
            logger.warning(f"Redis not available, using in-memory cache: {e}")
            self.redis_client = None
            self.memory_cache = {}
    
    def _initialize_kafka(self):
        try:
            self.kafka_producer = KafkaProducer(
                bootstrap_servers=self.config.KAFKA_BOOTSTRAP_SERVERS,
                value_serializer=lambda v: json.dumps(v).encode('utf-8'),
                key_serializer=lambda k: k.encode('utf-8') if k else None
            )
            logger.info("Kafka producer initialized")
        except Exception as e:
            logger.warning(f"Kafka not available: {e}")
            self.kafka_producer = None
    
    def _initialize_priority_queue(self):
        self.priority_queue = asyncio.PriorityQueue(maxsize=self.config.PRIORITY_QUEUE_SIZE)
        self.queue_processor_task = None
    
    async def process_detection_request(self, request: DetectionRequest) -> IntegratedDetectionResult:
        start_time = time.time()
        request_id = request.request_id
        
        logger.info(f"Processing detection request {request_id} for {request.entity_type}:{request.entity_id}")
        
        cached_result = await self._check_cache(request)
        if cached_result:
            logger.info(f"Cache hit for {request_id}")
            return cached_result
        
        self.active_requests[request_id] = request
        active_investigations.inc()
        
        try:
            detectors_to_run = self._get_relevant_detectors(request.entity_type)
            
            detection_results = await self._run_detectors_parallel(request, detectors_to_run)
            
            overall_fraud_score = self._calculate_overall_fraud_score(detection_results)
            trust_score = self._calculate_trust_score(overall_fraud_score)
            
            cross_intel_result = None
            cross_intel_triggered = False
            
            if overall_fraud_score >= self.config.CROSS_INTEL_THRESHOLD:
                cross_intel_triggered = True
                cross_intel_result = await self._trigger_cross_intelligence(request, detection_results)
            
            recommendations = self._generate_recommendations(
                request, detection_results, overall_fraud_score, cross_intel_result
            )
            
            priority_actions = self._determine_priority_actions(
                overall_fraud_score, detection_results, cross_intel_result
            )
            
            result = IntegratedDetectionResult(
                request_id=request_id,
                entity_type=request.entity_type,
                entity_id=request.entity_id,
                overall_fraud_score=overall_fraud_score,
                trust_score=trust_score,
                detection_results=detection_results,
                cross_intel_triggered=cross_intel_triggered,
                cross_intel_result=cross_intel_result,
                recommendations=recommendations,
                priority_actions=priority_actions,
                processing_time=time.time() - start_time
            )
            
            await self._cache_result(request, result)
            
            await self._publish_results(result)
            
            self.completed_requests.append(request_id)
            del self.active_requests[request_id]
            active_investigations.dec()
            
            logger.info(f"Completed detection request {request_id} in {result.processing_time:.2f}s")
            
            return result
            
        except Exception as e:
            logger.error(f"Error processing request {request_id}: {e}")
            del self.active_requests[request_id]
            active_investigations.dec()
            raise
    
    def _get_relevant_detectors(self, entity_type: str) -> List[str]:
        return self.entity_detector_map.get(entity_type, ['all'])
    
    async def _run_detectors_parallel(self, request: DetectionRequest, 
                                    detector_names: List[str]) -> Dict[str, DetectionResult]:
        results = {}
        
        tasks = []
        for detector_name in detector_names:
            if detector_name in self.detectors:
                detector = self.detectors[detector_name]
                task = asyncio.create_task(detector.detect(request.entity_data))
                tasks.append((detector_name, task))
        
        for detector_name, task in tasks:
            try:
                result = await task
                results[detector_name] = result
            except Exception as e:
                logger.error(f"Detector {detector_name} failed: {e}")
                results[detector_name] = DetectionResult(
                    detector_name=detector_name,
                    fraud_score=0.0,
                    confidence=0.0,
                    error=str(e)
                )
        
        return results
    
    def _calculate_overall_fraud_score(self, detection_results: Dict[str, DetectionResult]) -> float:
        if not detection_results:
            return 0.0
        
        weights = {
            'counterfeit': 1.5,
            'review': 1.2,
            'seller': 1.3,
            'listing': 1.0
        }
        
        total_weighted_score = 0.0
        total_weight = 0.0
        
        for detector_name, result in detection_results.items():
            if not result.error and result.confidence > 0:
                weight = weights.get(detector_name, 1.0) * result.confidence
                total_weighted_score += result.fraud_score * weight
                total_weight += weight
        
        if total_weight > 0:
            return min(total_weighted_score / total_weight, 1.0)
        
        return 0.0
    
    def _calculate_trust_score(self, fraud_score: float) -> float:
        if fraud_score < 0.2:
            trust_score = 95 - (fraud_score * 50)
        elif fraud_score < 0.5:
            trust_score = 85 - ((fraud_score - 0.2) * 100)
        elif fraud_score < 0.8:
            trust_score = 55 - ((fraud_score - 0.5) * 100)
        else:
            trust_score = 25 - ((fraud_score - 0.8) * 100)
        
        return max(0, min(100, trust_score))
    
    async def _trigger_cross_intelligence(self, request: DetectionRequest,
                                        detection_results: Dict[str, DetectionResult]) -> Dict:
        logger.info(f"Triggering Cross Intelligence for {request.entity_id}")
        
        highest_score = 0.0
        signal_type = SignalType.SUSPICIOUS_SELLER
        
        for detector_name, result in detection_results.items():
            if result.fraud_score > highest_score:
                highest_score = result.fraud_score
                if detector_name == 'review':
                    signal_type = SignalType.FAKE_REVIEW
                elif detector_name == 'counterfeit':
                    signal_type = SignalType.COUNTERFEIT_PRODUCT
                elif detector_name == 'seller':
                    signal_type = SignalType.SUSPICIOUS_SELLER
                elif detector_name == 'listing':
                    signal_type = SignalType.LISTING_FRAUD
        
        fraud_signal = FraudSignal(
            signal_id=f"SIGNAL_{request.request_id}",
            signal_type=signal_type,
            entity_id=request.entity_id,
            confidence=highest_score,
            timestamp=datetime.now(),
            metadata={
                'entity_type': request.entity_type,
                'detection_results': {k: v.fraud_score for k, v in detection_results.items()}
            },
            source_detector='integration_layer'
        )
        
        try:
            investigation = await self.cross_intelligence.trace_fraud_network(fraud_signal)
            
            viz_data = self.cross_intelligence.create_visualization_data(investigation)
            
            return {
                'investigation_id': investigation.investigation_id,
                'network_type': investigation.network_type.value if investigation.network_type else 'unknown',
                'network_size': investigation.network_graph.number_of_nodes(),
                'financial_impact': investigation.financial_impact,
                'confidence': investigation.confidence_score,
                'key_players': investigation.key_players[:5],
                'visualization': viz_data,
                'expansion_path': investigation.expansion_path[-3:]
            }
            
        except Exception as e:
            logger.error(f"Cross Intelligence failed: {e}")
            return {'error': str(e)}
    
    def _generate_recommendations(self, request: DetectionRequest,
                                detection_results: Dict[str, DetectionResult],
                                overall_fraud_score: float,
                                cross_intel_result: Optional[Dict]) -> List[str]:
        recommendations = []
        
        if overall_fraud_score >= 0.8:
            recommendations.append("IMMEDIATE ACTION: High fraud probability detected")
            recommendations.append("Recommend immediate listing suspension pending investigation")
        elif overall_fraud_score >= 0.5:
            recommendations.append("WARNING: Moderate fraud risk detected")
            recommendations.append("Recommend enhanced monitoring and manual review")
        elif overall_fraud_score >= 0.3:
            recommendations.append("CAUTION: Some fraud indicators present")
            recommendations.append("Recommend adding to watchlist")
        
        for detector_name, result in detection_results.items():
            if result.fraud_score > 0.7:
                if detector_name == 'counterfeit':
                    recommendations.append("Notify brand owner of potential counterfeit")
                elif detector_name == 'review':
                    recommendations.append("Flag all reviews from identified suspicious reviewers")
                elif detector_name == 'seller':
                    recommendations.append("Investigate all products from this seller")
                elif detector_name == 'listing':
                    recommendations.append("Review listing history for manipulation patterns")
        
        if cross_intel_result and 'network_size' in cross_intel_result:
            if cross_intel_result['network_size'] > 50:
                recommendations.append(f"NETWORK DETECTED: {cross_intel_result['network_size']} connected entities found")
                recommendations.append("Recommend bulk action on entire network")
            
            if cross_intel_result.get('financial_impact', 0) > 100000:
                recommendations.append(f"HIGH VALUE FRAUD: Estimated impact ${cross_intel_result['financial_impact']:,.2f}")
                recommendations.append("Escalate to senior investigation team")
        
        return recommendations
    
    def _determine_priority_actions(self, overall_fraud_score: float,
                                  detection_results: Dict[str, DetectionResult],
                                  cross_intel_result: Optional[Dict]) -> List[Dict]:
        actions = []
        
        if overall_fraud_score >= 0.9:
            actions.append({
                'action': 'suspend_listing',
                'priority': 'CRITICAL',
                'automated': True,
                'reason': 'Fraud score exceeds critical threshold'
            })
            
        if overall_fraud_score >= 0.8:
            actions.append({
                'action': 'hold_payments',
                'priority': 'HIGH',
                'automated': True,
                'reason': 'High fraud probability detected'
            })
        
        if cross_intel_result and cross_intel_result.get('network_size', 0) > 20:
            actions.append({
                'action': 'bulk_investigation',
                'priority': 'HIGH',
                'automated': False,
                'targets': cross_intel_result.get('key_players', []),
                'reason': f"Part of {cross_intel_result['network_size']}-node fraud network"
            })
        
        high_risk_detectors = [
            name for name, result in detection_results.items()
            if result.fraud_score > 0.8
        ]
        
        if 'counterfeit' in high_risk_detectors:
            actions.append({
                'action': 'brand_notification',
                'priority': 'HIGH',
                'automated': True,
                'reason': 'Counterfeit product detected'
            })
        
        if 'review' in high_risk_detectors:
            actions.append({
                'action': 'review_removal',
                'priority': 'MEDIUM',
                'automated': False,
                'reason': 'Fake review pattern detected'
            })
        
        return actions
    
    async def _check_cache(self, request: DetectionRequest) -> Optional[IntegratedDetectionResult]:
        cache_key = f"detection:{request.entity_type}:{request.entity_id}"
        
        try:
            if self.redis_client:
                cached_data = self.redis_client.get(cache_key)
                if cached_data:
                    return pickle.loads(cached_data)
            elif hasattr(self, 'memory_cache'):
                return self.memory_cache.get(cache_key)
        except Exception as e:
            logger.warning(f"Cache read error: {e}")
        
        return None
    
    async def _cache_result(self, request: DetectionRequest, result: IntegratedDetectionResult):
        cache_key = f"detection:{request.entity_type}:{request.entity_id}"
        
        try:
            if self.redis_client:
                self.redis_client.setex(
                    cache_key,
                    self.config.CACHE_TTL,
                    pickle.dumps(result)
                )
            elif hasattr(self, 'memory_cache'):
                self.memory_cache[cache_key] = result
        except Exception as e:
            logger.warning(f"Cache write error: {e}")
    
    async def _publish_results(self, result: IntegratedDetectionResult):
        if not self.kafka_producer:
            return
        
        try:
            if result.overall_fraud_score > 0.5:
                self.kafka_producer.send(
                    self.config.KAFKA_TOPICS['output']['fraud-detections'],
                    key=result.entity_id,
                    value={
                        'request_id': result.request_id,
                        'entity_type': result.entity_type,
                        'entity_id': result.entity_id,
                        'fraud_score': result.overall_fraud_score,
                        'trust_score': result.trust_score,
                        'timestamp': result.timestamp.isoformat(),
                        'cross_intel_triggered': result.cross_intel_triggered
                    }
                )
            
            self.kafka_producer.send(
                self.config.KAFKA_TOPICS['output']['trust-scores'],
                key=result.entity_id,
                value={
                    'entity_id': result.entity_id,
                    'trust_score': result.trust_score,
                    'timestamp': result.timestamp.isoformat()
                }
            )
            
            for action in result.priority_actions:
                if action['priority'] in ['CRITICAL', 'HIGH']:
                    self.kafka_producer.send(
                        self.config.KAFKA_TOPICS['output']['alerts'],
                        value={
                            'request_id': result.request_id,
                            'action': action,
                            'timestamp': datetime.now().isoformat()
                        }
                    )
            
        except Exception as e:
            logger.error(f"Kafka publish error: {e}")
    
    def calculate_priority(self, request: DetectionRequest) -> Priority:
        if request.entity_data.get('brand', '').lower() in ['nike', 'apple', 'louis vuitton', 'rolex']:
            return Priority.HIGH
        
        if request.entity_data.get('price', 0) > 500:
            return Priority.HIGH
        
        if request.entity_type == 'seller' and request.entity_data.get('account_age_days', 365) < 30:
            return Priority.HIGH
        
        if request.metadata.get('anomaly_detected'):
            return Priority.MEDIUM
        
        return Priority.LOW