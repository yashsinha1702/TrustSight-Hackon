"""
TRUSTSIGHT INTEGRATION LAYER - COMPLETE VERSION
This is the central orchestration engine that connects all components
"""

import asyncio
import json
import logging
import time
import torch
import joblib
from abc import ABC, abstractmethod
from collections import defaultdict, deque
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Any, Set, Tuple
from uuid import uuid4

import numpy as np
import pandas as pd
from aiokafka import AIOKafkaConsumer, AIOKafkaProducer
from kafka import KafkaProducer
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import uvicorn

# Import all components
from cross_intelligence import CrossIntelligenceEngine, FraudSignal, SignalType
from decision_memory_bank import FraudDecisionMemoryBank
from action_layer import TrustSightActionLayer, ActionRequest
from lifecycle_predictive import EnhancedCrossIntelligence

# Import detector modules (these would be your actual detector implementations)
#from review import FraudDetectionInference as ReviewDetector
#from counterfeit import HybridCounterfeitDetector
#from seller import SellerNetworkDetectionModel
#from listing import ListingFraudDetector

# Import Redis for caching
import redis
import pickle
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
from dataclasses import dataclass, field
from typing import Dict, Optional

from dataclasses import dataclass, field
from typing import Dict, Optional, Any

@dataclass
class ModelConfig:
    """Configuration for TrustSight Review Fraud Detector"""
    model_name: str = "roberta-base"
    hidden_size: int = 768
    num_fraud_types: int = 6
    max_length: int = 256
    dropout_rate: float = 0.1
    learning_rate: float = 2e-5
    batch_size: int = 32
    num_epochs: int = 10
    warmup_steps: int = 500
    log_every_n_steps: int = 10
    validate_every_n_steps: int = 100
    save_every_n_epochs: int = 1
    use_wandb: bool = False
    task_weights: Optional[Dict[str, float]] = field(default_factory=dict)

@dataclass
class SellerNetworkConfig:
    """Configuration for Seller Network Detection Model"""
    embedding_dim: int = 128
    hidden_dim: int = 256
    num_heads: int = 8
    num_gcn_layers: int = 3
    dropout_rate: float = 0.1
    learning_rate: float = 1e-5
    batch_size: int = 32
    num_epochs: int = 1
    warmup_steps: int = 500
    task_weights: Optional[Dict[str, float]] = field(default_factory=dict)


# ============= Fix for the _loss module error =============
# This is a workaround for the sklearn _loss module issue
import sys
import types

# Create a dummy _loss module to prevent import errors
_loss = types.ModuleType('_loss')

# Add the loss functions that sklearn models expect
loss_functions = [
    'CyHalfBinomialLoss',
    'CyHalfPoissonLoss', 
    'CyHalfSquaredError',
    'CyHalfMultinomialLoss',
    'CyAbsoluteError',
    'CyPinballLoss',
    'CyHalfGammaLoss',
    'CyHalfTweedieLoss',
    'CyHalfTweedieLossIdentity',
    'CyExponentialLoss'
]

# Create dummy classes for each loss function
for loss_name in loss_functions:
    setattr(_loss, loss_name, type(loss_name, (), {}))

# Add to sys.modules
sys.modules['_loss'] = _loss

# ============= Prometheus Metrics =============

from prometheus_client import Counter, Histogram, Gauge

# Define metrics
detection_counter = Counter('trustsight_detections_total', 'Total detections', ['detector', 'result'])
processing_time = Histogram('trustsight_processing_seconds', 'Processing time', ['detector'])
active_investigations = Gauge('trustsight_active_investigations', 'Active investigations')
kafka_messages_processed = Counter('trustsight_kafka_messages', 'Kafka messages processed', ['topic'])
cache_hits = Counter('trustsight_cache_hits', 'Cache hit rate', ['operation'])

# ============= Configuration =============

class IntegrationConfig:
    """Configuration for Integration Layer"""
    
    # Model Paths - Update these with your actual model paths
    MODEL_PATHS = {
        'review_fraud': {
            'model': 'C:/Users/syash/Desktop/trustsightnotclean/best_model_f1_1.0000.pt',
            'tokenizer': 'C:/Users/syash/Desktop/trustsightnotclean/models/review/tokenizer.pkl',
            'config': 'C:/Users/syash/Desktop/trustsightnotclean/models/review/model_config.json'
        },
        'counterfeit': {
            'text_model': 'models/counterfeit/text_model.pt',
            'image_model': 'models/counterfeit/image_model.pt',
            'hybrid_model': 'models/counterfeit/hybrid_model.pkl',
            'brand_embeddings': 'models/counterfeit/brand_embeddings.pkl'
        },
        'seller_network': {
            'gcn_model': 'C:/Users/syash/Desktop/trustsightnotclean/seller_network_best_f1_0.5707.pt',  # Fixed path
            'feature_extractor': 'models/seller/feature_extractor.pkl',
            'graph_embeddings': 'models/seller/graph_embeddings.pkl'
        },
        'listing_fraud': {
            'mismatch_model': 'C:/Users/syash/Desktop/trustsightnotclean/mismatch_detector_model.pkl',
            'seo_model': 'C:/Users/syash/Desktop/trustsightnotclean/seo_detector_model.pkl',
            'ensemble_model': 'C:/Users/syash/Desktop/trustsightnotclean/listing_fraud_ensemble_model.pkl'
        }
    }
    
    # Device configuration for PyTorch models
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Detection thresholds
    CROSS_INTEL_THRESHOLD = 0.5  # Trigger cross intelligence above this
    ACTION_THRESHOLD = 0.6       # Trigger actions above this
    HIGH_PRIORITY_THRESHOLD = 0.8
    
    # Processing
    MAX_PARALLEL_DETECTIONS = 10
    CACHE_TTL_SECONDS = 3600  # 1 hour
    BATCH_SIZE = 100
    PRIORITY_QUEUE_SIZE = 10000
    
    # Redis Configuration
    REDIS_HOST = 'localhost'
    REDIS_PORT = 6379
    REDIS_DB = 0
    REDIS_DECODE_RESPONSES = False
    
    # Kafka configuration
    KAFKA_BOOTSTRAP_SERVERS = 'localhost:9092'
    KAFKA_TOPICS = {
        'product-events': 'trustsight-product-events',
        'review-events': 'trustsight-review-events',
        'seller-events': 'trustsight-seller-events',
        'listing-events': 'trustsight-listing-events',
        'fraud-detections': 'trustsight-fraud-detections'
    }
    
    # API configuration
    API_HOST = '0.0.0.0'
    API_PORT = 8000
    
    # Component weights for overall fraud score
    DETECTOR_WEIGHTS = {
        'review': 0.3,
        'counterfeit': 0.3,
        'seller': 0.2,
        'listing': 0.2
    }

# ============= Data Classes =============

class Priority(Enum):
    CRITICAL = 1
    HIGH = 2
    MEDIUM = 3
    LOW = 4
    BATCH = 5

@dataclass
class DetectionRequest:
    """Request for fraud detection"""
    entity_type: str  # product, review, seller, listing
    entity_id: str
    entity_data: Dict[str, Any]
    priority: Priority = Priority.MEDIUM
    source: str = 'api'  # api, kafka, batch
    metadata: Dict[str, Any] = field(default_factory=dict)
    request_id: str = field(default_factory=lambda: str(uuid4()))
    timestamp: datetime = field(default_factory=datetime.now)

@dataclass
class DetectionResult:
    """Result from individual detector"""
    detector_name: str
    fraud_score: float  # 0-1
    confidence: float   # 0-1
    fraud_types: List[str] = field(default_factory=list)
    evidence: List[Dict] = field(default_factory=list)
    processing_time: float = 0.0
    error: Optional[str] = None

@dataclass
class IntegratedDetectionResult:
    """Combined result from all detectors"""
    request_id: str
    entity_type: str
    entity_id: str
    overall_fraud_score: float
    trust_score: float  # 0-100
    detection_results: Dict[str, DetectionResult] = field(default_factory=dict)
    cross_intel_triggered: bool = False
    cross_intel_result: Optional[Dict] = None
    lifecycle_analysis: Optional[Dict] = None
    predictions: Optional[List[Dict]] = None
    decision_id: Optional[str] = None
    actions_taken: List[Dict] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)
    priority_actions: List[Dict] = field(default_factory=list)
    processing_time: float = 0.0
    timestamp: datetime = field(default_factory=datetime.now)

# ============= Detector Interfaces =============

class DetectorInterface(ABC):
    """Abstract interface for all detectors"""
    
    @abstractmethod
    async def detect(self, data: Dict[str, Any]) -> DetectionResult:
        pass
    
    @abstractmethod
    def get_name(self) -> str:
        pass
    
    @abstractmethod
    def load_model(self):
        pass

# ============= Real Detector Implementations =============

# ============= Complete Review Fraud Detector =============
from transformers import AutoTokenizer
import torch
import numpy as np
from datetime import datetime
import os

class ReviewFraudDetectorInterface(DetectorInterface):
    """Interface for review fraud detector with real model loading"""
    
    def __init__(self, model_paths: Dict[str, str] = None, config: IntegrationConfig = None):
        self.name = "review_fraud_detector"
        self.config = config or IntegrationConfig()
        self.model_paths = model_paths or self.config.MODEL_PATHS['review_fraud']
        self.model = None
        self.tokenizer = None
        self.device = self.config.DEVICE
        self.model_config = ModelConfig()
        self.load_model()
        
        mode = "model-based" if self.model is not None else "rule-based"
        logger.info(f"Initialized {self.name} in {mode} mode on device: {self.device}")
    
    def get_name(self) -> str:
        """Return the detector name"""
        return self.name
    
    def load_model(self):
        """Load the trained review fraud detection model"""
        try:
            from transformers import AutoTokenizer
            self.tokenizer = AutoTokenizer.from_pretrained('roberta-base')
            logger.info("Loaded RoBERTa tokenizer")
            
            # Load model checkpoint
            if os.path.exists(self.model_paths['model']):
                checkpoint = torch.load(self.model_paths['model'], map_location=self.device)
                
                # Extract config
                if 'config' in checkpoint:
                    self.model_config = checkpoint['config']
                    logger.info(f"Loaded config from checkpoint, best F1: {checkpoint.get('best_f1', 'N/A')}")
                
                # Import the CORRECT model class from review.py
                from review import TrustSightReviewFraudDetector
                
                # Create model instance
                self.model = TrustSightReviewFraudDetector(self.model_config)
                
                # Load the state dict
                self.model.load_state_dict(checkpoint['model_state_dict'])
                self.model.to(self.device)
                self.model.eval()
                
                logger.info("Successfully loaded TrustSightReviewFraudDetector model")
                
        except ImportError as e:
            logger.error(f"Cannot import TrustSightReviewFraudDetector from review.py: {e}")
            logger.error("Please ensure review.py is in the same directory")
            self.model = None
            self.tokenizer = None
        except Exception as e:
            logger.error(f"Failed to load review fraud model: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            self.model = None
            self.tokenizer = None
    
    async def detect(self, data: Dict[str, Any]) -> DetectionResult:
        start_time = time.time()
        
        try:
            if not self.model or not self.tokenizer:
                logger.error("Model or tokenizer not loaded")
                return DetectionResult(
                    detector_name=self.name,
                    fraud_score=0.5,
                    confidence=0.0,
                    error="Model not loaded",
                    processing_time=time.time() - start_time
                )
            
            # Transform input data to match training format
            review_features = self._transform_to_training_format(data)
            
            # Extract text for tokenization
            review_text = review_features.get('review_text', '')
            
            # Tokenize text
            tokenized = self.tokenizer(
                review_text,
                padding='max_length',
                truncation=True,
                max_length=256,
                return_tensors='pt'
            )
            
            # Create the batch format your model expects
            batch = {
                'input_ids': tokenized['input_ids'].to(self.device),
                'attention_mask': tokenized['attention_mask'].to(self.device),
                'rating': torch.tensor([review_features.get('rating', 3.0)], dtype=torch.float32).to(self.device),
                'verified_purchase': torch.tensor([review_features.get('verified_purchase', 0)], dtype=torch.float32).to(self.device),
                'hours_since_delivery': torch.tensor([review_features.get('hours_since_delivery', 24.0)], dtype=torch.float32).to(self.device),
                'reviewer_account_age_days': torch.tensor([review_features.get('reviewer_account_age_days', 365)], dtype=torch.float32).to(self.device),
                'reviewer_total_reviews': torch.tensor([review_features.get('reviewer_total_reviews', 10)], dtype=torch.float32).to(self.device),
                'reviewer_verified_purchases_pct': torch.tensor([review_features.get('reviewer_verified_purchases_pct', 0.8)], dtype=torch.float32).to(self.device),
                'reviews_per_day_avg': torch.tensor([review_features.get('reviews_per_day_avg', 0.1)], dtype=torch.float32).to(self.device)
            }
            
            # Add labels as your model expects them during training (dummy labels for inference)
            batch['labels'] = {
                'is_fraud': torch.tensor([0], dtype=torch.float32).to(self.device),
                'generic_text': torch.tensor([0], dtype=torch.float32).to(self.device),
                'timing_anomaly': torch.tensor([0], dtype=torch.float32).to(self.device),
                'bot_reviewer': torch.tensor([0], dtype=torch.float32).to(self.device),
                'incentivized': torch.tensor([0], dtype=torch.float32).to(self.device),
                'network_fraud': torch.tensor([0], dtype=torch.float32).to(self.device)
            }
            
            # Run model inference
            with torch.no_grad():
                outputs = self.model(batch)
            
            # Extract fraud probability and predictions
            fraud_prob = outputs['fraud_probability'].item()
            
            # Generate evidence based on model outputs
            evidence = []
            fraud_types = []
            
            if fraud_prob > 0.7:
                evidence.append({
                    'type': 'ai_model_detection',
                    'description': 'Advanced AI model detected fraudulent patterns',
                    'confidence': fraud_prob
                })
                
                # Check specific fraud type predictions
                if outputs.get('generic_text_pred', torch.tensor([0.0]))[0].item() > 0.5:
                    fraud_types.append('generic_text')
                    evidence.append({
                        'type': 'generic_text',
                        'description': 'Generic or template-like review detected'
                    })
                
                if outputs.get('timing_anomaly_pred', torch.tensor([0.0]))[0].item() > 0.5:
                    fraud_types.append('timing_anomaly')
                    evidence.append({
                        'type': 'timing_anomaly',
                        'description': 'Suspicious timing pattern detected'
                    })
                
                if outputs.get('bot_reviewer_pred', torch.tensor([0.0]))[0].item() > 0.5:
                    fraud_types.append('bot_reviewer')
                    evidence.append({
                        'type': 'bot_reviewer',
                        'description': 'Bot-like reviewer behavior detected'
                    })
                
                if outputs.get('network_fraud_pred', torch.tensor([0.0]))[0].item() > 0.5:
                    fraud_types.append('network_fraud')
                    evidence.append({
                        'type': 'network_fraud',
                        'description': 'Part of coordinated fraud network'
                    })
                
                if outputs.get('incentivized_pred', torch.tensor([0.0]))[0].item() > 0.5:
                    fraud_types.append('incentivized')
                    evidence.append({
                        'type': 'incentivized',
                        'description': 'Potentially incentivized review'
                    })
            
            return DetectionResult(
                detector_name=self.name,
                fraud_score=fraud_prob,
                confidence=0.95,  # High confidence in model
                fraud_types=fraud_types if fraud_types else ['clean'],
                evidence=evidence,
                processing_time=time.time() - start_time
            )
            
        except Exception as e:
            logger.error(f"Error in {self.name}: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            return DetectionResult(
                detector_name=self.name,
                fraud_score=0.5,  # Neutral score on error
                confidence=0.0,   # No confidence
                error=str(e),
                processing_time=time.time() - start_time
            )
    
    def _transform_to_training_format(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Transform API input to match training data format"""
        now = datetime.now()
        
        # Extract or calculate features matching your training data
        review_features = {
            'review_id': data.get('review_id', f"R_{uuid4().hex[:12]}"),
            'product_id': data.get('product_id', 'Unknown'),
            'reviewer_id': data.get('reviewer_id', data.get('user_id', 'Unknown')),
            'rating': float(data.get('rating', 3)),
            'review_text': data.get('review_text', ''),
            'verified_purchase': 1 if data.get('verified_purchase', False) else 0,
            'review_timestamp': data.get('review_timestamp', now.isoformat()),
            'delivery_timestamp': data.get('delivery_timestamp', now.isoformat()),
            'reviewer_account_age_days': data.get('reviewer_account_age_days', 365),
            'reviewer_total_reviews': data.get('reviewer_total_reviews', 10),
            'reviewer_verified_purchases_pct': data.get('reviewer_verified_purchases_pct', 0.8),
            'reviews_per_day_avg': data.get('reviews_per_day_avg', 0.1),
        }
        
        # Calculate hours since delivery
        if 'hours_since_delivery' not in data:
            try:
                delivery = datetime.fromisoformat(review_features['delivery_timestamp'].replace('Z', '+00:00'))
                review = datetime.fromisoformat(review_features['review_timestamp'].replace('Z', '+00:00'))
                hours_diff = (review - delivery).total_seconds() / 3600
                review_features['hours_since_delivery'] = max(0, hours_diff)
            except:
                review_features['hours_since_delivery'] = 24  # Default to 1 day
        else:
            review_features['hours_since_delivery'] = data['hours_since_delivery']
        
        return review_features


# ============= Complete Seller Network Detector =============
class SellerNetworkDetectorInterface(DetectorInterface):
    """Interface for seller network detector with real model loading"""
    
    def __init__(self, model_paths: Dict[str, str] = None, config: IntegrationConfig = None):
        self.name = "seller_network_detector"
        self.config = config or IntegrationConfig()
        self.model_paths = model_paths or self.config.MODEL_PATHS['seller_network']
        self.gcn_model = None
        self.feature_extractor = None
        self.graph_embeddings = None
        self.device = self.config.DEVICE
        self.model_config = SellerNetworkConfig()
        self.load_model()
        
        mode = "model-based" if self.gcn_model is not None else "rule-based"
        logger.info(f"Initialized {self.name} in {mode} mode")
    
    def get_name(self) -> str:
        """Return the detector name"""
        return self.name
    
    def load_model(self):
        """Load the trained seller network detection models"""
        try:
            # Load GCN model
            if os.path.exists(self.model_paths.get('gcn_model', '')):
                self.gcn_model = torch.load(self.model_paths['gcn_model'], map_location=self.device)
                if hasattr(self.gcn_model, 'eval'):
                    self.gcn_model.eval()
                logger.info("Loaded GCN model")
            else:
                logger.warning(f"GCN model not found: {self.model_paths.get('gcn_model', '')}")
            
            # Load feature extractor
            if os.path.exists(self.model_paths.get('feature_extractor', '')):
                self.feature_extractor = joblib.load(self.model_paths['feature_extractor'])
                logger.info("Loaded feature extractor")
            
            # Load graph embeddings
            if os.path.exists(self.model_paths.get('graph_embeddings', '')):
                with open(self.model_paths['graph_embeddings'], 'rb') as f:
                    self.graph_embeddings = pickle.load(f)
                logger.info("Loaded graph embeddings")
                
        except Exception as e:
            logger.error(f"Failed to load seller network models: {str(e)}")
            self.gcn_model = None
    
    async def detect(self, data: Dict[str, Any]) -> DetectionResult:
        start_time = time.time()
        
        try:
            if not self.gcn_model:
                logger.error("GCN model not loaded")
                return DetectionResult(
                    detector_name=self.name,
                    fraud_score=0.5,
                    confidence=0.0,
                    error="Model not loaded",
                    processing_time=time.time() - start_time
                )
            
            # Transform to training format
            seller_features = self._transform_to_training_format(data)
            
            # Extract features for the model
            feature_vector = self._extract_feature_vector(seller_features)
            
            # Create node features
            node_features = torch.FloatTensor(feature_vector).unsqueeze(0).to(self.device)
            
            # For single seller detection, create a simple graph
            # If we have connected sellers, create a more complex graph
            connected_sellers = data.get('connected_sellers', [])
            if connected_sellers:
                # Create adjacency matrix for GCN
                num_nodes = 1 + len(connected_sellers)
                edge_list = []
                for i, _ in enumerate(connected_sellers):
                    edge_list.extend([[0, i+1], [i+1, 0]])  # Bidirectional edges
                
                if edge_list:
                    edge_index = torch.LongTensor(edge_list).t().to(self.device)
                else:
                    edge_index = torch.LongTensor([[0], [0]]).to(self.device)
                
                # Repeat features for all nodes (simplified)
                node_features = node_features.repeat(num_nodes, 1)
            else:
                edge_index = torch.LongTensor([[0], [0]]).to(self.device)  # Self-loop
            
            with torch.no_grad():
                # Run GCN
                output = self.gcn_model(node_features, edge_index)
                if output.dim() > 1:
                    fraud_prob = torch.sigmoid(output[0]).item()
                else:
                    fraud_prob = torch.sigmoid(output).item()
            
            # Generate evidence
            evidence = []
            fraud_types = []
            
            if fraud_prob > 0.7:
                evidence.append({
                    'type': 'network_detection',
                    'description': 'GCN model detected suspicious network patterns',
                    'confidence': fraud_prob
                })
                
                # Check specific indicators
                if len(connected_sellers) > 5:
                    evidence.append({
                        'type': 'large_network',
                        'description': f"Connected to {len(connected_sellers)} other sellers"
                    })
                    fraud_types.append('coordinated_network')
                
                if seller_features.get('same_day_registrations', 0) > 3:
                    evidence.append({
                        'type': 'registration_cluster',
                        'description': f"{seller_features['same_day_registrations']} sellers registered same day"
                    })
                    fraud_types.append('bulk_registration')
                
                if seller_features['financial_indicators'].get('unusual_refund_spike', False):
                    evidence.append({
                        'type': 'financial_anomaly',
                        'description': 'Unusual refund spike detected'
                    })
                    fraud_types.append('exit_scam_risk')
            
            return DetectionResult(
                detector_name=self.name,
                fraud_score=fraud_prob,
                confidence=0.9,
                fraud_types=fraud_types if fraud_types else ['network_fraud'],
                evidence=evidence,
                processing_time=time.time() - start_time
            )
            
        except Exception as e:
            logger.error(f"Error in {self.name}: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            return DetectionResult(
                detector_name=self.name,
                fraud_score=0.5,
                confidence=0.0,
                error=str(e),
                processing_time=time.time() - start_time
            )
    
    def _transform_to_training_format(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Transform to match training data structure"""
        return {
            'seller_id': data.get('seller_id', ''),
            'business_name': data.get('business_name', ''),
            'registration_date': data.get('registration_date', ''),
            'account_age_days': data.get('account_age_days', 365),
            'same_day_registrations': data.get('same_date_registrations', 0),  # Map field name
            'business_address': data.get('business_address', {
                'city': 'Unknown',
                'state': 'Unknown',
                'country': 'US'
            }),
            'seller_metrics': {
                'total_products': data.get('total_products', 10),
                'active_products': data.get('active_products', 5),
                'avg_product_price': data.get('avg_product_price', 50),
                'total_reviews': data.get('total_reviews', 100),
                'avg_rating': data.get('avg_rating', 4.0),
                'response_time_hours': data.get('response_time_hours', 24),
                'fulfillment_rate': data.get('fulfillment_rate', 0.9),
                'return_rate': data.get('return_rate', 0.05),
                'customer_complaints': data.get('customer_complaints', 0)
            },
            'network_features': {
                'same_day_registrations': data.get('same_date_registrations', []),
                'address_similarity_score': data.get('address_similarity_score', 0.0),
                'customer_overlap_score': data.get('customer_overlap_score', 0.0),
                'reviewer_overlap_score': data.get('reviewer_overlap_score', 0.0)
            },
            'financial_indicators': {
                'revenue_growth_rate': data.get('revenue_growth_rate', 1.0),
                'unusual_refund_spike': data.get('unusual_refund_spike', False)
            }
        }
    
    def _extract_feature_vector(self, seller_data: Dict) -> np.ndarray:
        """Extract numerical features for the model"""
        features = []
        
        # Basic features
        features.append(seller_data.get('account_age_days', 0) / 365.0)
        features.append(seller_data.get('same_day_registrations', 0) / 10.0)
        
        # Seller metrics
        metrics = seller_data.get('seller_metrics', {})
        features.append(metrics.get('total_products', 0) / 1000.0)
        features.append(metrics.get('active_products', 0) / 100.0)
        features.append(metrics.get('avg_product_price', 0) / 500.0)
        features.append(metrics.get('avg_rating', 0) / 5.0)
        features.append(metrics.get('fulfillment_rate', 0))
        features.append(metrics.get('return_rate', 0))
        features.append(metrics.get('customer_complaints', 0) / 100.0)
        
        # Network features
        network = seller_data.get('network_features', {})
        features.append(network.get('address_similarity_score', 0))
        features.append(network.get('customer_overlap_score', 0))
        features.append(network.get('reviewer_overlap_score', 0))
        
        # Financial indicators
        financial = seller_data.get('financial_indicators', {})
        features.append(financial.get('revenue_growth_rate', 0) / 10.0)
        features.append(1.0 if financial.get('unusual_refund_spike', False) else 0.0)
        
        # Pad to expected size if needed (based on your model)
        while len(features) < self.model_config.embedding_dim:
            features.append(0.0)
        
        return np.array(features[:self.model_config.embedding_dim])


# ============= Complete Listing Fraud Detector =============
class ListingFraudDetectorInterface(DetectorInterface):
    """Interface for listing fraud detector with real model loading"""
    
    def __init__(self, model_paths: Dict[str, str] = None, config: IntegrationConfig = None):
        self.name = "listing_fraud_detector"
        self.config = config or IntegrationConfig()
        self.model_paths = model_paths or self.config.MODEL_PATHS['listing_fraud']
        self.mismatch_model = None
        self.seo_model = None
        self.ensemble_model = None
        self.load_model()
        logger.info(f"Initialized {self.name}")
    
    def get_name(self) -> str:
        """Return the detector name"""
        return self.name
    
    def load_model(self):
        """Load the trained listing fraud detection models"""
        try:
            import warnings
            warnings.filterwarnings('ignore', category=UserWarning, module='sklearn')
            
            # Load mismatch detector
            try:
                if os.path.exists(self.model_paths.get('mismatch_model', '')):
                    self.mismatch_model = joblib.load(self.model_paths['mismatch_model'])
                    logger.info("Loaded mismatch detection model")
            except Exception as e:
                logger.warning(f"Failed to load mismatch model: {e}")
                self.mismatch_model = None
            
            # Load SEO detector
            try:
                if os.path.exists(self.model_paths.get('seo_model', '')):
                    self.seo_model = joblib.load(self.model_paths['seo_model'])
                    logger.info("Loaded SEO manipulation model")
            except Exception as e:
                logger.warning(f"Failed to load SEO model: {e}")
                self.seo_model = None
            
            # Load ensemble model
            try:
                if os.path.exists(self.model_paths.get('ensemble_model', '')):
                    self.ensemble_model = joblib.load(self.model_paths['ensemble_model'])
                    logger.info("Loaded listing ensemble model")
            except Exception as e:
                logger.warning(f"Failed to load ensemble model: {e}")
                self.ensemble_model = None
                
            if not any([self.mismatch_model, self.seo_model, self.ensemble_model]):
                logger.warning("No listing models loaded successfully")
                
        except Exception as e:
            logger.error(f"Failed to load listing fraud models: {e}")
    
    async def detect(self, data: Dict[str, Any]) -> DetectionResult:
        start_time = time.time()
        
        try:
            if not self.ensemble_model:
                logger.error("Ensemble model not loaded")
                return DetectionResult(
                    detector_name=self.name,
                    fraud_score=0.5,
                    confidence=0.0,
                    error="Model not loaded",
                    processing_time=time.time() - start_time
                )
            
            # Extract features
            features = self._extract_listing_features(data)
            
            # Use ensemble model
            fraud_prob = self.ensemble_model.predict_proba(features)[0][1]
            
            # Generate evidence
            evidence = []
            fraud_types = []
            
            if fraud_prob > 0.6:
                evidence.append({
                    'type': 'ensemble_detection',
                    'description': 'Multiple fraud indicators detected',
                    'confidence': fraud_prob
                })
                
                # Check specific fraud types based on input data
                title = data.get('title', '')
                variations = data.get('variations', [])
                reviews = data.get('reviews', [])
                
                # Check for SEO manipulation
                seo_keywords = ['best', 'cheap', 'discount', 'sale', 'guarantee', 'premium', 'amazing']
                seo_count = sum(1 for kw in seo_keywords if kw in title.lower())
                if seo_count > 3:
                    fraud_types.append('seo_manipulation')
                    evidence.append({
                        'type': 'seo_manipulation', 
                        'description': f'Excessive keywords detected ({seo_count} found)'
                    })
                
                # Check for variation abuse
                if variations:
                    categories = set()
                    for var in variations:
                        if isinstance(var, dict):
                            var_name = var.get('name', '').lower()
                            # Detect category from variation name
                            if 'phone' in var_name or 'case' in var_name:
                                categories.add('phone_accessories')
                            elif 'shoe' in var_name or 'size' in var_name:
                                categories.add('footwear')
                            elif 'watch' in var_name:
                                categories.add('watches')
                            else:
                                categories.add('other')
                    
                    if len(categories) > 2:
                        fraud_types.append('variation_abuse')
                        evidence.append({
                            'type': 'variation_abuse',
                            'description': f'Unrelated variations across {len(categories)} categories'
                        })
                
                # Check for review mismatch
                if reviews:
                    mismatched = 0
                    for review in reviews[:10]:
                        review_text = review.get('text', '').lower()
                        if not any(word in review_text for word in title.lower().split()[:3]):
                            mismatched += 1
                    
                    if mismatched > 5:
                        fraud_types.append('review_mismatch')
                        evidence.append({
                            'type': 'review_mismatch',
                            'description': f'{mismatched}/10 reviews appear to be for different products'
                        })
            
            return DetectionResult(
                detector_name=self.name,
                fraud_score=fraud_prob,
                confidence=0.85,
                fraud_types=fraud_types if fraud_types else ['listing_fraud'],
                evidence=evidence,
                processing_time=time.time() - start_time
            )
            
        except Exception as e:
            logger.error(f"Error in {self.name}: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            return DetectionResult(
                detector_name=self.name,
                fraud_score=0.5,
                confidence=0.0,
                error=str(e),
                processing_time=time.time() - start_time
            )
    
    def _extract_listing_features(self, data: Dict[str, Any]) -> np.ndarray:
        """Extract features matching the ensemble model's training (9 features)"""
        features = []
        
        # If pre-calculated features are provided, use them
        if 'feature_mismatch_score' in data:
            features = [
                data.get('feature_mismatch_score', 0),
                data.get('feature_evolution_score', 0),
                data.get('feature_seo_score', 0),
                data.get('feature_variation_score', 0),
                data.get('feature_hijack_score', 0),
                data.get('feature_review_count', 0),
                data.get('feature_price_variance', 0),
                data.get('feature_seller_age', 0),
                data.get('feature_category_competitiveness', 0)
            ]
        else:
            # Calculate features from raw data
            title = data.get('title', '')
            description = data.get('description', '')
            price = float(data.get('price', 0))
            variations = data.get('variations', [])
            reviews = data.get('reviews', [])
            category = data.get('category', '').lower()
            
            # Feature 1: Mismatch score
            mismatch_score = 0.0
            if reviews:
                for i, review in enumerate(reviews[:10]):
                    review_text = review.get('text', '').lower()
                    # Check if review mentions the product
                    title_words = title.lower().split()[:3]  # First 3 words of title
                    if not any(word in review_text for word in title_words):
                        mismatch_score += 0.1
            features.append(min(mismatch_score, 1.0))
            
            # Feature 2: Evolution score (default to mid-range)
            features.append(0.5)
            
            # Feature 3: SEO score
            seo_keywords = ['best', 'cheap', 'discount', 'sale', 'guarantee', 'premium', 'amazing', 'free', 'limited']
            seo_score = sum(1 for kw in seo_keywords if kw in title.lower()) / len(seo_keywords)
            features.append(min(seo_score * 2, 1.0))  # Scale up
            
            # Feature 4: Variation score
            variation_score = 0.0
            if variations:
                categories = set()
                for var in variations:
                    if isinstance(var, dict):
                        var_name = var.get('name', '').lower()
                        # Simple category detection
                        if any(color in var_name for color in ['black', 'white', 'red', 'blue', 'green']):
                            categories.add('color')
                        elif any(size in var_name for size in ['small', 'medium', 'large', 'xl', 'xxl']):
                            categories.add('size')
                        else:
                            categories.add(var_name.split()[0] if var_name else 'unknown')
                
                # More categories = more suspicious
                variation_score = (len(categories) - 1) / 4.0
            features.append(min(variation_score, 1.0))
            
            # Feature 5: Hijack score (default low)
            features.append(0.1)
            
            # Feature 6: Review count (normalized)
            review_count_norm = min(len(reviews) / 100.0, 1.0)
            features.append(review_count_norm)
            
            # Feature 7: Price variance (default mid-range)
            features.append(0.5)
            
            # Feature 8: Seller age (normalized, default 1 year)
            seller_age = data.get('seller_age_days', 365) / 1000.0
            features.append(min(seller_age, 1.0))
            
            # Feature 9: Category competitiveness
            competitive_categories = ['electronics', 'beauty', 'health', 'supplements', 'phone', 'computer']
            competitiveness = 0.8 if any(cat in category for cat in competitive_categories) else 0.3
            features.append(competitiveness)
        
        return np.array(features).reshape(1, -1)
class CounterfeitDetectorInterface(DetectorInterface):
    """Interface for counterfeit detector with real model loading"""
    
    def __init__(self, model_paths: Dict[str, str] = None, config: IntegrationConfig = None):
        self.name = "counterfeit_detector"
        self.config = config or IntegrationConfig()
        self.model_paths = model_paths or self.config.MODEL_PATHS['counterfeit']
        self.text_model = None
        self.image_model = None
        self.hybrid_model = None
        self.brand_embeddings = None
        self.device = self.config.DEVICE
        self.load_model()
        logger.info(f"Initialized {self.name}")
    
    def load_model(self):
        """Load the trained counterfeit detection models"""
        try:
            import os
            
            # Load text model
            if os.path.exists(self.model_paths['text_model']):
                self.text_model = torch.load(self.model_paths['text_model'], map_location=self.device)
                self.text_model.eval()
                logger.info("Loaded counterfeit text model")
            
            # Load image model
            if os.path.exists(self.model_paths['image_model']):
                self.image_model = torch.load(self.model_paths['image_model'], map_location=self.device)
                self.image_model.eval()
                logger.info("Loaded counterfeit image model")
            
            # Load hybrid model
            if os.path.exists(self.model_paths['hybrid_model']):
                self.hybrid_model = joblib.load(self.model_paths['hybrid_model'])
                logger.info("Loaded hybrid counterfeit model")
            
            # Load brand embeddings
            if os.path.exists(self.model_paths['brand_embeddings']):
                with open(self.model_paths['brand_embeddings'], 'rb') as f:
                    self.brand_embeddings = pickle.load(f)
                logger.info("Loaded brand embeddings")
                
        except Exception as e:
            logger.error(f"Failed to load counterfeit models: {e}")
            logger.info("Falling back to rule-based detection")
    
    async def detect(self, data: Dict[str, Any]) -> DetectionResult:
        start_time = time.time()
        
        try:
            evidence = []
            fraud_types = []
            
            # Extract features
            price = data.get('price', 0)
            brand = data.get('brand', '').lower()
            title = data.get('title', '')
            images = data.get('images', [])
            seller_info = data.get('seller_info', {})
            
            if self.hybrid_model and self.brand_embeddings:
                # Use trained models
                features = self._extract_features(data)
                fraud_prob = self.hybrid_model.predict_proba([features])[0][1]
                
                # Add evidence based on model insights
                if fraud_prob > 0.7:
                    evidence.append({
                        'type': 'model_detection',
                        'description': 'Multiple counterfeit indicators detected',
                        'confidence': fraud_prob
                    })
                    fraud_types.append('counterfeit_product')
                    
            else:
                # Fallback to rule-based detection
                fraud_score = 0.0
                
                # Price anomaly detection
                brand_avg_prices = {
                    'nike': 80, 'adidas': 75, 'apple': 500,
                    'gucci': 800, 'louis vuitton': 1200
                }
                
                if brand in brand_avg_prices:
                    expected_price = brand_avg_prices[brand]
                    price_ratio = price / expected_price
                    
                    if price_ratio < 0.3:  # 70% below expected
                        fraud_score += 0.4
                        evidence.append({
                            'type': 'price_anomaly',
                            'description': f'Price ${price} is {(1-price_ratio)*100:.0f}% below market average',
                            'severity': 'high'
                        })
                        fraud_types.append('price_manipulation')
                
                # Seller authority check
                if not seller_info.get('authorized_seller', False):
                    fraud_score += 0.3
                    evidence.append({
                        'type': 'unauthorized_seller',
                        'description': 'Seller not authorized for this brand'
                    })
                    fraud_types.append('unauthorized_distribution')
                
                # Text analysis
                suspicious_keywords = ['replica', 'inspired', 'style', 'type', 'oem']
                if any(keyword in title.lower() for keyword in suspicious_keywords):
                    fraud_score += 0.2
                    evidence.append({
                        'type': 'suspicious_keywords',
                        'description': 'Title contains counterfeit indicators'
                    })
                
                fraud_prob = min(fraud_score, 0.95)
            
            result = DetectionResult(
                detector_name=self.name,
                fraud_score=fraud_prob,
                confidence=0.88,
                fraud_types=fraud_types,
                evidence=evidence,
                processing_time=time.time() - start_time
            )
            
            # Track metrics
            detection_counter.labels(
                detector=self.name,
                result='fraud' if result.fraud_score > 0.5 else 'clean'
            ).inc()
            processing_time.labels(detector=self.name).observe(result.processing_time)
            
            return result
            
        except Exception as e:
            logger.error(f"Error in {self.name}: {e}")
            return DetectionResult(
                detector_name=self.name,
                fraud_score=0.0,
                confidence=0.0,
                error=str(e),
                processing_time=time.time() - start_time
            )
    
    def _extract_features(self, data: Dict[str, Any]) -> np.ndarray:
        """Extract features for model inference"""
        features = []
        
        # Price features
        price = data.get('price', 0)
        features.append(price)
        
        # Brand encoding (simplified)
        brand = data.get('brand', '').lower()
        brand_vec = self.brand_embeddings.get(brand, np.zeros(50))
        features.extend(brand_vec)
        
        # Text features (simplified)
        title_len = len(data.get('title', ''))
        features.append(title_len)
        
        # Seller features
        seller_rating = data.get('seller_info', {}).get('rating', 0)
        features.append(seller_rating)
        
        return np.array(features)
    
    def get_name(self) -> str:
        return self.name


# ============= Integration Engine =============

class TrustSightIntegrationEngine:
    """
    Central orchestration engine for TrustSight
    Coordinates all detectors and routes to Cross Intelligence
    """
    
    def __init__(self, config: IntegrationConfig = None):
        self.config = config or IntegrationConfig()
        
        # Initialize components
        self._initialize_detectors()
        self._initialize_intelligence_engines()
        self._initialize_decision_memory()
        self._initialize_action_layer()
        self._initialize_cache()
        self._initialize_kafka()
        self._initialize_priority_queue()
        
        # Thread pool for parallel processing
        self.executor = ThreadPoolExecutor(max_workers=self.config.MAX_PARALLEL_DETECTIONS)
        
        # Tracking
        self.active_requests = {}
        self.completed_requests = deque(maxlen=1000)
        
        # Entity to detector mapping
        self.entity_detector_map = {
            'product': ['counterfeit', 'listing'],
            'review': ['review'],
            'seller': ['seller'],
            'listing': ['listing', 'counterfeit'],
            'all': ['review', 'seller', 'listing', 'counterfeit']
        }
        
        logger.info("TrustSight Integration Engine initialized with real models!")
    
    def _initialize_detectors(self):
        """Initialize all fraud detectors with real models"""
        self.detectors = {
            'review': ReviewFraudDetectorInterface(
                self.config.MODEL_PATHS.get('review_fraud'),
                self.config
            ),
            'counterfeit': CounterfeitDetectorInterface(
                self.config.MODEL_PATHS.get('counterfeit'),
                self.config
            ),
            'seller': SellerNetworkDetectorInterface(
                self.config.MODEL_PATHS.get('seller_network'),
                self.config
            ),
            'listing': ListingFraudDetectorInterface(
                self.config.MODEL_PATHS.get('listing_fraud'),
                self.config
            )
        }
        logger.info(f"Initialized {len(self.detectors)} detectors with trained models")
    
    def _initialize_intelligence_engines(self):
        """Initialize intelligence components"""
        # Cross Intelligence
        self.cross_intelligence = CrossIntelligenceEngine()
        
        # Enhanced Intelligence (Lifecycle + Predictive)
        self.enhanced_intel = EnhancedCrossIntelligence(self.cross_intelligence)
        
        logger.info("Intelligence engines initialized")
    
    def _initialize_decision_memory(self):
        """Initialize Decision Memory Bank"""
        self.decision_memory_bank = FraudDecisionMemoryBank()
        logger.info("Decision Memory Bank initialized")
    
    def _initialize_action_layer(self):
        """Initialize Action Layer"""
        self.action_layer = TrustSightActionLayer(self.decision_memory_bank)
        logger.info("Action Layer initialized")
    
    def _initialize_cache(self):
        """Initialize caching layer with Redis and fallback"""
        try:
            self.redis_client = redis.Redis(
                host=self.config.REDIS_HOST,
                port=self.config.REDIS_PORT,
                db=self.config.REDIS_DB,
                decode_responses=self.config.REDIS_DECODE_RESPONSES
            )
            # Test connection
            self.redis_client.ping()
            logger.info("Redis cache connected successfully")
        except Exception as e:
            logger.warning(f"Redis not available, falling back to in-memory cache: {e}")
            self.redis_client = None
        
        # Always initialize memory cache as fallback
        self.memory_cache = {}
        self.cache_timestamps = {}
    
    def _initialize_kafka(self):
        """Initialize Kafka connections"""
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
        
        self.kafka_consumers = {}
    
    def _initialize_priority_queue(self):
        """Initialize priority processing queue"""
        self.priority_queue = asyncio.PriorityQueue(maxsize=self.config.PRIORITY_QUEUE_SIZE)
        self.queue_processor_task = None
    
    async def process_detection_request(self, request: DetectionRequest) -> IntegratedDetectionResult:
        """
        Main processing pipeline for fraud detection
        """
        start_time = time.time()
        logger.info(f"Processing detection request: {request.request_id}")
        
        # Track active request
        self.active_requests[request.request_id] = request
        active_investigations.inc()
        
        try:
            # Step 1: Check cache
            cache_key = f"{request.entity_type}:{request.entity_id}"
            cached_result = await self._get_cached_result(cache_key)
            if cached_result:
                logger.info(f"Cache hit for {cache_key}")
                cache_hits.labels(operation='detection').inc()
                return cached_result
            
            # Step 2: Run all relevant detectors in parallel
            detection_results = await self._run_detectors(request)
            
            # Step 3: Calculate overall fraud score
            overall_fraud_score = self._calculate_overall_score(detection_results)
            trust_score = self._calculate_trust_score(overall_fraud_score)
            
            # Step 4: Check if Cross Intelligence should be triggered
            cross_intel_result = None
            cross_intel_triggered = False
            
            if overall_fraud_score > self.config.CROSS_INTEL_THRESHOLD:
                cross_intel_result = await self._trigger_cross_intelligence(request, detection_results)
                cross_intel_triggered = True
            
            # Step 5: Enhance with Lifecycle and Predictive Intelligence
            lifecycle_analysis = None
            predictions = None
            
            if cross_intel_triggered:
                enhanced_result = await self.enhanced_intel.analyze_with_intelligence(
                    request.entity_type,
                    request.entity_id,
                    {
                        'fraud_score': overall_fraud_score,
                        'detection_results': {k: v.fraud_score for k, v in detection_results.items()},
                        'cross_intel_result': cross_intel_result
                    }
                )
                lifecycle_analysis = enhanced_result.get('lifecycle_analysis')
                predictions = enhanced_result.get('fraud_predictions')
            
            # Step 6: Record decision in Memory Bank
            decision_id = None
            if overall_fraud_score > 0.3:  # Record significant detections
                fraud_signal = {
                    'signal_type': self._get_signal_type(request.entity_type),
                    'entity_type': request.entity_type,
                    'entity_id': request.entity_id
                }
                
                detection_result = {
                    'overall_fraud_score': overall_fraud_score,
                    'confidence': max(d.confidence for d in detection_results.values()),
                    'primary_evidence': self._extract_evidence(detection_results),
                    'cross_intel_result': cross_intel_result,
                    'lifecycle_analysis': lifecycle_analysis,
                    'predictions': predictions
                }
                
                decision_id = await self.decision_memory_bank.record_decision(
                    fraud_signal, detection_result
                )
            
            # Step 7: Trigger Action Layer if needed
            actions_taken = []
            if overall_fraud_score > self.config.ACTION_THRESHOLD:
                action_request = ActionRequest(
                    request_id=request.request_id,
                    entity_type=request.entity_type,
                    entity_id=request.entity_id,
                    entity_data=request.entity_data,
                    fraud_score=overall_fraud_score,
                    confidence=max(d.confidence for d in detection_results.values()),
                    evidence=self._extract_evidence(detection_results),
                    detection_timestamp=datetime.now(),
                    network_indicators=cross_intel_result.get('network_indicators', []) if cross_intel_result else [],
                    lifecycle_stage=lifecycle_analysis.get('current_stage') if lifecycle_analysis else None,
                    predicted_risks=predictions if predictions else []
                )
                
                action_result = await self.action_layer.process_fraud_detection(action_request)
                actions_taken = action_result.actions_taken
            
            # Step 8: Generate recommendations
            recommendations = self._generate_recommendations(
                request, detection_results, overall_fraud_score, cross_intel_result
            )
            
            # Step 9: Determine priority actions
            priority_actions = self._determine_priority_actions(
                overall_fraud_score, detection_results, cross_intel_result
            )
            
            # Create final result
            result = IntegratedDetectionResult(
                request_id=request.request_id,
                entity_type=request.entity_type,
                entity_id=request.entity_id,
                overall_fraud_score=overall_fraud_score,
                trust_score=trust_score,
                detection_results=detection_results,
                cross_intel_triggered=cross_intel_triggered,
                cross_intel_result=cross_intel_result,
                lifecycle_analysis=lifecycle_analysis,
                predictions=predictions,
                decision_id=decision_id,
                actions_taken=actions_taken,
                recommendations=recommendations,
                priority_actions=priority_actions,
                processing_time=time.time() - start_time
            )
            
            # Cache result
            await self._cache_result(cache_key, result)
            
            # Move to completed
            del self.active_requests[request.request_id]
            self.completed_requests.append(request.request_id)
            active_investigations.dec()
            
            # Publish to Kafka if configured
            await self._publish_results(result)
            
            logger.info(f"Detection complete: {request.request_id} - Score: {overall_fraud_score:.2%}")
            
            return result
            
        except Exception as e:
            logger.error(f"Error processing request {request.request_id}: {e}")
            del self.active_requests[request.request_id]
            active_investigations.dec()
            raise
    
    async def _run_detectors(self, request: DetectionRequest) -> Dict[str, DetectionResult]:
        """Run all relevant detectors in parallel"""
        detector_tasks = {}
        
        # Determine which detectors to run based on entity type
        detectors_to_run = self._get_relevant_detectors(request.entity_type)
        
        # Create tasks for each detector
        for detector_name in detectors_to_run:
            detector = self.detectors.get(detector_name)
            if detector:
                detector_tasks[detector_name] = asyncio.create_task(
                    detector.detect(request.entity_data)
                )
        
        # Wait for all detectors to complete
        results = {}
        for detector_name, task in detector_tasks.items():
            try:
                results[detector_name] = await task
            except Exception as e:
                logger.error(f"Detector {detector_name} failed: {e}")
                results[detector_name] = DetectionResult(
                    detector_name=detector_name,
                    fraud_score=0.0,
                    confidence=0.0,
                    error=str(e)
                )
        
        return results
    
    def _get_relevant_detectors(self, entity_type: str) -> List[str]:
        """Determine which detectors to run based on entity type"""
        return self.entity_detector_map.get(entity_type, self.entity_detector_map['all'])
    
    def _calculate_overall_score(self, detection_results: Dict[str, DetectionResult]) -> float:
        """Calculate weighted overall fraud score"""
        total_weighted_score = 0.0
        total_weight = 0.0
        
        for detector_name, result in detection_results.items():
            if not result.error:
                weight = self.config.DETECTOR_WEIGHTS.get(detector_name, 0.25)
                total_weighted_score += result.fraud_score * weight
                total_weight += weight
        
        if total_weight > 0:
            return min(total_weighted_score / total_weight, 1.0)
        
        return 0.0
    
    def _calculate_trust_score(self, fraud_score: float) -> float:
        """Convert fraud score to trust score (0-100)"""
        # Non-linear transformation for better UX
        if fraud_score < 0.2:
            trust_score = 95 - (fraud_score * 50)  # 95-85
        elif fraud_score < 0.5:
            trust_score = 85 - ((fraud_score - 0.2) * 100)  # 85-55
        elif fraud_score < 0.8:
            trust_score = 55 - ((fraud_score - 0.5) * 100)  # 55-25
        else:
            trust_score = 25 - ((fraud_score - 0.8) * 100)  # 25-5
        
        return max(0, min(100, trust_score))
    
    async def _trigger_cross_intelligence(self, request: DetectionRequest,
                                        detection_results: Dict[str, DetectionResult]) -> Dict:
        """Trigger Cross Intelligence Engine for network analysis"""
        logger.info(f"Triggering Cross Intelligence for {request.entity_id}")
        
        # Create fraud signal
        highest_score = 0.0
        signal_type = SignalType.SUSPICIOUS_SELLER  # Default
        
        for detector_name, result in detection_results.items():
            if result.fraud_score > highest_score:
                highest_score = result.fraud_score
                # Map detector to signal type
                signal_type_map = {
                    'review': SignalType.FAKE_REVIEW,
                    'counterfeit': SignalType.COUNTERFEIT_PRODUCT,
                    'seller': SignalType.SUSPICIOUS_SELLER,
                    'listing': SignalType.LISTING_FRAUD
                }
                signal_type = signal_type_map.get(detector_name, SignalType.NETWORK_PATTERN)
        
        # Create fraud signal
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
        
        # Run Cross Intelligence investigation
        try:
            investigation = await self.cross_intelligence.trace_fraud_network(fraud_signal)
            
            # Get visualization data
            viz_data = self.cross_intelligence.create_visualization_data(investigation)
            
            # Get recommended actions
            actions = self.cross_intelligence.recommend_actions(investigation)
            
            return {
                'investigation_id': investigation.investigation_id,
                'network_size': investigation.network_graph.number_of_nodes(),
                'network_type': investigation.network_type.value if investigation.network_type else 'unknown',
                'financial_impact': investigation.financial_impact,
                'confidence_score': investigation.confidence_score,
                'key_players': investigation.key_players,
                'visualization_data': viz_data,
                'recommended_actions': actions,
                'network_indicators': self._extract_network_indicators(investigation)
            }
            
        except Exception as e:
            logger.error(f"Cross Intelligence failed: {e}")
            return {
                'error': str(e),
                'network_size': 0,
                'financial_impact': 0
            }
    
    def _extract_network_indicators(self, investigation) -> List[str]:
        """Extract network indicators from investigation"""
        indicators = []
        
        if investigation.network_type:
            indicators.append(f"Network Type: {investigation.network_type.value}")
        
        if investigation.network_graph.number_of_nodes() > 10:
            indicators.append("Large fraud network detected")
        
        if investigation.financial_impact > 10000:
            indicators.append(f"High financial impact: ${investigation.financial_impact:,.2f}")
        
        return indicators
    
    def _get_signal_type(self, entity_type: str) -> str:
        """Map entity type to signal type"""
        mapping = {
            'product': 'counterfeit_product',
            'review': 'fake_review',
            'seller': 'suspicious_seller',
            'listing': 'listing_fraud'
        }
        return mapping.get(entity_type, 'unknown')
    
    def _extract_evidence(self, detection_results: Dict[str, DetectionResult]) -> List[Dict]:
        """Extract top evidence from all detectors"""
        all_evidence = []
        
        for detector_name, result in detection_results.items():
            if not result.error:
                for evidence in result.evidence:
                    evidence['source'] = detector_name
                    evidence['score'] = result.fraud_score
                    all_evidence.append(evidence)
        
        # Sort by score and return top evidence
        all_evidence.sort(key=lambda x: x['score'], reverse=True)
        return all_evidence[:5]
    
    def _generate_recommendations(self, request: DetectionRequest, detection_results: Dict,
                                overall_fraud_score: float, cross_intel_result: Dict) -> List[str]:
        """Generate actionable recommendations based on detection results"""
        recommendations = []
        
        # Based on overall fraud score
        if overall_fraud_score > 0.8:
            recommendations.append("Immediate action required - high fraud confidence")
            recommendations.append("Consider temporary suspension pending investigation")
        elif overall_fraud_score > 0.6:
            recommendations.append("Enhanced monitoring recommended")
            recommendations.append("Request additional verification from entity")
        elif overall_fraud_score > 0.4:
            recommendations.append("Add to watchlist for continued monitoring")
        
        # Based on specific detectors
        for detector_name, result in detection_results.items():
            if result.fraud_score > 0.7:
                if detector_name == 'review':
                    recommendations.append("Review all recent reviews from this source")
                elif detector_name == 'counterfeit':
                    recommendations.append("Verify product authenticity with brand owner")
                elif detector_name == 'seller':
                    recommendations.append("Investigate connected seller accounts")
                elif detector_name == 'listing':
                    recommendations.append("Review listing compliance and variations")
        
        # Based on Cross Intelligence
        if cross_intel_result and cross_intel_result.get('network_size', 0) > 5:
            recommendations.append(f"Investigate {cross_intel_result['network_size']} connected entities")
            recommendations.append("Consider bulk action on entire network")
        
        return list(set(recommendations))  # Remove duplicates
    
    def _determine_priority_actions(self, fraud_score: float, detection_results: Dict,
                                  cross_intel_result: Dict) -> List[Dict]:
        """Determine priority actions based on detection results"""
        actions = []
        
        if fraud_score > self.config.HIGH_PRIORITY_THRESHOLD:
            actions.append({
                'action': 'IMMEDIATE_REVIEW',
                'priority': 'CRITICAL',
                'reason': 'High fraud score detected',
                'deadline': 'Within 1 hour'
            })
        
        if cross_intel_result and cross_intel_result.get('network_size', 0) > 10:
            actions.append({
                'action': 'NETWORK_INVESTIGATION',
                'priority': 'HIGH',
                'reason': f"Large network detected ({cross_intel_result['network_size']} nodes)",
                'deadline': 'Within 24 hours'
            })
        
        # Check for specific fraud types
        for detector_name, result in detection_results.items():
            if result.fraud_score > 0.7:
                for fraud_type in result.fraud_types:
                    if fraud_type in ['exit_scam', 'review_bomb', 'account_takeover']:
                        actions.append({
                            'action': f'PREVENT_{fraud_type.upper()}',
                            'priority': 'CRITICAL',
                            'reason': f'{fraud_type} pattern detected',
                            'deadline': 'Immediate'
                        })
        
        return sorted(actions, key=lambda x: 1 if x['priority'] == 'CRITICAL' else 2)
    
    # ============= Caching Methods =============
    
    async def _get_cached_result(self, cache_key: str) -> Optional[IntegratedDetectionResult]:
        """Get cached result from Redis or memory"""
        # Try Redis first
        if self.redis_client:
            try:
                cached_data = self.redis_client.get(cache_key)
                if cached_data:
                    return pickle.loads(cached_data)
            except Exception as e:
                logger.warning(f"Redis read error: {e}")
        
        # Fallback to memory cache
        if cache_key in self.memory_cache:
            timestamp = self.cache_timestamps.get(cache_key, 0)
            if time.time() - timestamp < self.config.CACHE_TTL_SECONDS:
                return self.memory_cache[cache_key]
            else:
                # Expired - remove from cache
                del self.memory_cache[cache_key]
                del self.cache_timestamps[cache_key]
        
        return None
    
    async def _cache_result(self, cache_key: str, result: IntegratedDetectionResult):
        """Cache detection result in Redis and memory"""
        # Cache in Redis if available
        if self.redis_client:
            try:
                self.redis_client.setex(
                    cache_key,
                    self.config.CACHE_TTL_SECONDS,
                    pickle.dumps(result)
                )
            except Exception as e:
                logger.warning(f"Redis write error: {e}")
        
        # Always cache in memory as fallback
        self.memory_cache[cache_key] = result
        self.cache_timestamps[cache_key] = time.time()
    
    # ============= Kafka Methods =============
    
    async def _publish_results(self, result: IntegratedDetectionResult):
        """Publish detection result to Kafka"""
        if self.kafka_producer is None:
            return  # Kafka not configured
        
        try:
            message = {
                'request_id': result.request_id,
                'entity_type': result.entity_type,
                'entity_id': result.entity_id,
                'fraud_score': result.overall_fraud_score,
                'trust_score': result.trust_score,
                'timestamp': result.timestamp.isoformat(),
                'actions_taken': result.actions_taken,
                'cross_intel_triggered': result.cross_intel_triggered,
                'detection_results': {
                    k: {
                        'score': v.fraud_score,
                        'confidence': v.confidence,
                        'fraud_types': v.fraud_types
                    } for k, v in result.detection_results.items()
                }
            }
            
            # Send to Kafka
            self.kafka_producer.send(
                self.config.KAFKA_TOPICS['fraud-detections'],
                key=result.entity_id,
                value=message
            )
            
            kafka_messages_processed.labels(topic='fraud-detections').inc()
            
        except Exception as e:
            logger.error(f"Failed to publish to Kafka: {e}")
    
    def calculate_priority(self, request: DetectionRequest) -> Priority:
        """Calculate request priority based on various factors"""
        # High-value brands get higher priority
        high_value_brands = ['apple', 'nike', 'adidas', 'gucci', 'rolex', 'louis vuitton']
        if any(brand in str(request.entity_data).lower() for brand in high_value_brands):
            return Priority.HIGH
        
        # Price anomalies get high priority
        price = request.entity_data.get('price', 0)
        if price > 1000:
            return Priority.HIGH
        
        # Recent suspicious activity
        if request.metadata.get('recent_fraud_detected', False):
            return Priority.CRITICAL
        
        # Source-based priority
        if request.source == 'kafka':
            return Priority.MEDIUM
        elif request.source == 'batch':
            return Priority.BATCH
        
        return Priority.MEDIUM

# ============= Kafka Stream Processor =============

class KafkaStreamProcessor:
    """Process real-time events from Kafka"""
    
    def __init__(self, integration_engine: TrustSightIntegrationEngine):
        self.engine = integration_engine
        self.config = integration_engine.config
        self.running = False
    
    async def start(self):
        """Start processing Kafka streams"""
        self.running = True
        tasks = []
        
        for topic_name, topic in self.config.KAFKA_TOPICS.items():
            if topic_name != 'fraud-detections':  # Don't consume our own output
                task = asyncio.create_task(self._consume_topic(topic_name, topic))
                tasks.append(task)
        
        logger.info("Kafka stream processors started")
        await asyncio.gather(*tasks)
    
    async def stop(self):
        """Stop processing"""
        self.running = False
    
    async def _consume_topic(self, topic_name: str, topic: str):
        """Consume messages from a specific topic"""
        try:
            consumer = AIOKafkaConsumer(
                topic,
                bootstrap_servers=self.config.KAFKA_BOOTSTRAP_SERVERS,
                value_deserializer=lambda m: json.loads(m.decode('utf-8')),
                group_id=f'trustsight-{topic_name}'
            )
            
            await consumer.start()
            logger.info(f"Started consuming from {topic}")
            
            try:
                async for message in consumer:
                    if not self.running:
                        break
                    
                    # Process message
                    await self._process_message(topic_name, message.value)
                    
            finally:
                await consumer.stop()
                
        except Exception as e:
            logger.error(f"Failed to initialize Kafka consumer for {topic}: {e}")
    
    async def _process_message(self, topic_name: str, message: Dict):
        """Process individual Kafka message"""
        try:
            # Map topic to entity type
            entity_type_map = {
                'product-events': 'product',
                'review-events': 'review',
                'seller-events': 'seller',
                'listing-events': 'listing'
            }
            
            entity_type = entity_type_map.get(topic_name, 'unknown')
            
            # Create detection request
            request = DetectionRequest(
                entity_type=entity_type,
                entity_id=message.get('entity_id', ''),
                entity_data=message,
                source='kafka',
                metadata={'kafka_topic': topic_name}
            )
            
            # Calculate priority
            request.priority = self.engine.calculate_priority(request)
            
            # Add to processing queue
            await self.engine.priority_queue.put((request.priority.value, request))
            
            # Track Kafka metrics
            kafka_messages_processed.labels(topic=topic_name).inc()
            
        except Exception as e:
            logger.error(f"Error processing Kafka message: {e}")

# ============= REST API =============

app = FastAPI(title="TrustSight Integration API", version="1.0.0")

# Global engine instance
engine = None

# Pydantic models for API
class DetectionRequestAPI(BaseModel):
    entity_type: str = Field(..., description="Type of entity: product, review, seller, listing")
    entity_id: str = Field(..., description="Unique identifier for the entity")
    entity_data: Dict[str, Any] = Field(..., description="Entity data for detection")
    priority: str = Field(default="MEDIUM", description="Processing priority")

class BatchDetectionRequestAPI(BaseModel):
    requests: List[DetectionRequestAPI]
    process_async: bool = Field(default=True, description="Process asynchronously")

@app.on_event("startup")
async def startup_event():
    """Initialize engine on startup"""
    global engine
    engine = TrustSightIntegrationEngine()
    
    # Start queue processor
    engine.queue_processor_task = asyncio.create_task(process_priority_queue(engine))
    
    # Start Kafka processors
    kafka_processor = KafkaStreamProcessor(engine)
    asyncio.create_task(kafka_processor.start())
    
    logger.info("API server started with all components connected")

@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    if engine and engine.queue_processor_task:
        engine.queue_processor_task.cancel()

# ============= API Endpoints =============

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    redis_status = "healthy"
    if engine and engine.redis_client:
        try:
            engine.redis_client.ping()
        except:
            redis_status = "unhealthy"
    else:
        redis_status = "not_configured"
    
    detector_status = {}
    if engine:
        for name, detector in engine.detectors.items():
            # Check if models are loaded
            model_loaded = False
            if hasattr(detector, 'model') and detector.model is not None:
                model_loaded = True
            elif hasattr(detector, 'ensemble_model') and detector.ensemble_model is not None:
                model_loaded = True
            
            detector_status[name] = {
                'active': True,
                'model_loaded': model_loaded
            }
    
    return {
        "status": "healthy",
        "active_requests": len(engine.active_requests) if engine else 0,
        "detectors": detector_status,
        "components": {
            "cross_intelligence": "active",
            "decision_memory": "active",
            "action_layer": "active",
            "lifecycle_predictive": "active",
            "redis_cache": redis_status
        }
    }

@app.post("/detect", response_model=Dict)
async def detect_fraud(request: DetectionRequestAPI):
    """Single entity fraud detection"""
    detection_request = DetectionRequest(
        entity_type=request.entity_type,
        entity_id=request.entity_id,
        entity_data=request.entity_data,
        priority=Priority[request.priority],
        source='api'
    )
    
    result = await engine.process_detection_request(detection_request)
    
    return {
        "request_id": result.request_id,
        "entity_id": result.entity_id,
        "trust_score": result.trust_score,
        "fraud_score": result.overall_fraud_score,
        "fraud_detected": result.overall_fraud_score > 0.5,
        "cross_intel_triggered": result.cross_intel_triggered,
        "network_size": result.cross_intel_result.get('network_size', 0) if result.cross_intel_result else 0,
        "lifecycle_stage": result.lifecycle_analysis.get('current_stage') if result.lifecycle_analysis else None,
        "predictions": result.predictions[:3] if result.predictions else [],
        "decision_id": result.decision_id,
        "actions_taken": result.actions_taken,
        "recommendations": result.recommendations,
        "priority_actions": result.priority_actions,
        "processing_time": result.processing_time,
        "detector_results": {
            k: {
                "score": v.fraud_score,
                "confidence": v.confidence,
                "fraud_types": v.fraud_types,
                "evidence": v.evidence[:2]  # Top 2 evidence items
            } for k, v in result.detection_results.items()
        }
    }

@app.post("/batch-detect", response_model=Dict)
async def batch_detect_fraud(batch_request: BatchDetectionRequestAPI):
    """Batch fraud detection"""
    results = []
    
    for request in batch_request.requests:
        detection_request = DetectionRequest(
            entity_type=request.entity_type,
            entity_id=request.entity_id,
            entity_data=request.entity_data,
            priority=Priority[request.priority],
            source='api'
        )
        
        if batch_request.process_async:
            # Add to queue for async processing
            await engine.priority_queue.put((detection_request.priority.value, detection_request))
            results.append({
                "request_id": detection_request.request_id,
                "status": "queued",
                "priority": detection_request.priority.name
            })
        else:
            # Process synchronously
            result = await engine.process_detection_request(detection_request)
            results.append({
                "request_id": result.request_id,
                "entity_id": result.entity_id,
                "trust_score": result.trust_score,
                "fraud_score": result.overall_fraud_score,
                "fraud_detected": result.overall_fraud_score > 0.5
            })
    
    return {
        "batch_size": len(batch_request.requests),
        "processing_mode": "async" if batch_request.process_async else "sync",
        "results": results
    }

@app.get("/status/{request_id}")
async def get_request_status(request_id: str):
    """Get status of a specific detection request"""
    if request_id in engine.active_requests:
        return {
            "request_id": request_id,
            "status": "processing",
            "entity_type": engine.active_requests[request_id].entity_type,
            "entity_id": engine.active_requests[request_id].entity_id
        }
    elif request_id in engine.completed_requests:
        return {
            "request_id": request_id,
            "status": "completed"
        }
    else:
        raise HTTPException(status_code=404, detail="Request not found")

# Add this to your integration.py file, replacing the existing /metrics endpoint

@app.get("/metrics")
async def get_metrics():
    """Get system metrics"""
    # Calculate metrics from engine state
    active_count = len(engine.active_requests) if engine else 0
    completed_count = len(engine.completed_requests) if engine else 0
    queue_size = engine.priority_queue.qsize() if engine and hasattr(engine, 'priority_queue') else 0
    
    # Calculate detector metrics
    detector_metrics = {}
    if engine and hasattr(engine, 'detectors'):
        for name, detector in engine.detectors.items():
            detector_metrics[name] = "active"
    
    # Calculate cache metrics
    cache_size = 0
    cache_hits = 0
    cache_misses = 0
    
    if engine and engine.redis_client:
        try:
            # Get Redis info
            info = engine.redis_client.info()
            cache_size = info.get('used_memory_human', '0')
            
            # Get cache stats if stored
            hits = engine.redis_client.get('cache_hits')
            misses = engine.redis_client.get('cache_misses')
            
            if hits:
                cache_hits = int(hits)
            if misses:
                cache_misses = int(misses)
        except:
            pass
    
    # Check if Redis is connected
    redis_connected = False
    if engine and engine.redis_client:
        try:
            engine.redis_client.ping()
            redis_connected = True
        except:
            redis_connected = False
    
    return {
        "active_requests": active_count,
        "completed_requests": completed_count,
        "queue_size": queue_size,
        "cache_size": cache_size,
        "cache_hits": cache_hits,
        "cache_misses": cache_misses,
        "cache_hit_rate": cache_hits / (cache_hits + cache_misses) if (cache_hits + cache_misses) > 0 else 0,
        "detectors": detector_metrics,
        "redis_connected": redis_connected,
        "timestamp": datetime.now().isoformat()
    }

# ============= Priority Queue Processor =============

async def process_priority_queue(engine: TrustSightIntegrationEngine):
    """Process requests from priority queue"""
    logger.info("Priority queue processor started")
    
    while True:
        try:
            # Get next request from queue
            priority, request = await engine.priority_queue.get()
            
            # Process request
            asyncio.create_task(engine.process_detection_request(request))
            
        except asyncio.CancelledError:
            break
        except Exception as e:
            logger.error(f"Queue processor error: {e}")
            await asyncio.sleep(1)

# ============= Main =============

if __name__ == "__main__":
    import sys
    import os
    
    if len(sys.argv) > 1 and sys.argv[1] == "demo":
        # Run demo
        async def run_demo():
            engine = TrustSightIntegrationEngine()
            
            # Demo requests to test different detectors
            demo_requests = [
                # Fake review detection
                DetectionRequest(
                    entity_type='review',
                    entity_id='REVIEW_FAKE_001',
                    entity_data={
                        'review_text': 'Great product! Highly recommend! Five stars!',
                        'rating': 5,
                        'verified_purchase': False,
                        'reviewer_id': 'USER_12345'
                    }
                ),
                # Counterfeit product detection
                DetectionRequest(
                    entity_type='product',
                    entity_id='NIKE_FAKE_001',
                    entity_data={
                        'title': 'Nike Air Max Best Price Cheap',
                        'brand': 'Nike',
                        'price': 29.99,
                        'seller_info': {
                            'authorized_seller': False,
                            'rating': 3.2
                        }
                    }
                ),
                # Seller network detection
                DetectionRequest(
                    entity_type='seller',
                    entity_id='SELLER_NETWORK_001',
                    entity_data={
                        'seller_id': 'SELLER_123',
                        'registration_date': '2024-01-01',
                        'business_info': {
                            'address': '123 Industrial Park Unit 5'
                        },
                        'connected_sellers': ['SELLER_124', 'SELLER_125', 'SELLER_126'],
                        'same_date_registrations': 8,
                        'similar_addresses': 5
                    }
                ),
                # Listing fraud detection
                DetectionRequest(
                    entity_type='listing',
                    entity_id='LISTING_FRAUD_001',
                    entity_data={
                        'title': 'Best Nike Adidas Puma Shoes Cheap Price Sale Discount',
                        'category': 'Shoes',
                        'variations': [
                            {'name': 'Size 8 Black'},
                            {'name': 'iPhone Case'},  # Unrelated
                            {'name': 'Watch Band'}     # Unrelated
                        ],
                        'reviews': [
                            {'text': 'Great phone case!', 'rating': 5},
                            {'text': 'Love this watch!', 'rating': 5}
                        ]
                    }
                )
            ]
            
            print("\n" + "="*60)
            print("TRUSTSIGHT INTEGRATION ENGINE DEMO")
            print("="*60 + "\n")
            
            for request in demo_requests:
                print(f"\nProcessing {request.entity_type} - {request.entity_id}...")
                result = await engine.process_detection_request(request)
                
                print(f"  Trust Score: {result.trust_score:.1f}/100")
                print(f"  Fraud Score: {result.overall_fraud_score:.2%}")
                print(f"  Cross Intel: {'Yes' if result.cross_intel_triggered else 'No'}")
                
                if result.cross_intel_result and result.cross_intel_result.get('network_size', 0) > 0:
                    print(f"  Network Size: {result.cross_intel_result['network_size']} entities")
                
                if result.actions_taken:
                    print(f"  Actions: {len(result.actions_taken)} taken")
                
                print(f"  Processing Time: {result.processing_time:.2f}s")
                
                # Show detector results
                print("\n  Detector Results:")
                for detector_name, detector_result in result.detection_results.items():
                    if not detector_result.error:
                        print(f"    - {detector_name}: {detector_result.fraud_score:.2%} confidence: {detector_result.confidence:.2%}")
                        if detector_result.fraud_types:
                            print(f"      Types: {', '.join(detector_result.fraud_types)}")
            
            print("\n" + "="*60)
            print("Demo complete!")
        
        asyncio.run(run_demo())
    else:
        # Run API server
        uvicorn.run(
            app,
            host=IntegrationConfig.API_HOST,
            port=IntegrationConfig.API_PORT,
            log_level="info"
        )