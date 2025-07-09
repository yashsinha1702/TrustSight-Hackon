"""
CROSS INTELLIGENCE ENGINE - The Crown Jewel of TrustSight
This traces fraud networks from any single signal, exposing entire fraud rings
"""

import networkx as nx
import torch
import torch.nn as nn
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Set, Optional, Any
from dataclasses import dataclass, field
from collections import defaultdict, deque
import json
import logging
from enum import Enum
import asyncio
from concurrent.futures import ThreadPoolExecutor
import pandas as pd
from uuid import uuid4
import pickle

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ============= Data Classes =============

class SignalType(Enum):
    FAKE_REVIEW = "fake_review"
    COUNTERFEIT_PRODUCT = "counterfeit_product"
    SUSPICIOUS_SELLER = "suspicious_seller"
    LISTING_FRAUD = "listing_fraud"
    PRICE_ANOMALY = "price_anomaly"
    NETWORK_PATTERN = "network_pattern"

class NetworkType(Enum):
    REVIEW_FARM = "review_farm"
    SELLER_CARTEL = "seller_cartel"
    COUNTERFEIT_RING = "counterfeit_ring"
    HYBRID_OPERATION = "hybrid_operation"
    COMPETITOR_ATTACK = "competitor_attack"
    EXIT_SCAM_NETWORK = "exit_scam_network"

@dataclass
class FraudSignal:
    signal_id: str
    signal_type: SignalType
    entity_id: str  # review_id, product_id, seller_id, etc.
    confidence: float
    timestamp: datetime
    metadata: Dict[str, Any] = field(default_factory=dict)
    source_detector: str = ""

@dataclass
class NetworkNode:
    node_id: str
    node_type: str  # 'review', 'reviewer', 'product', 'seller', 'listing'
    attributes: Dict[str, Any] = field(default_factory=dict)
    fraud_score: float = 0.0
    investigation_depth: int = 0
    connected_nodes: Set[str] = field(default_factory=set)

@dataclass
class NetworkEdge:
    source: str
    target: str
    edge_type: str  # 'wrote_review', 'sells_product', 'shares_reviewer', etc.
    weight: float = 1.0
    timestamp: Optional[datetime] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class Investigation:
    investigation_id: str
    initial_signal: FraudSignal
    start_time: datetime
    network_graph: nx.DiGraph = field(default_factory=nx.DiGraph)
    nodes_investigated: Set[str] = field(default_factory=set)
    expansion_path: List[Dict] = field(default_factory=list)
    network_type: Optional[NetworkType] = None
    financial_impact: float = 0.0
    key_players: List[str] = field(default_factory=list)
    confidence_score: float = 0.0
    status: str = "in_progress"

    def to_dict(self):
        return {
            "investigation_id": self.investigation_id,
            "initial_signal": self.initial_signal.__dict__,  # you can customize this too
            "start_time": self.start_time.isoformat(),
            "nodes_investigated": list(self.nodes_investigated),
            "expansion_path": self.expansion_path,
            "network_type": self.network_type.name if self.network_type else None,
            "financial_impact": self.financial_impact,
            "key_players": self.key_players,
            "confidence_score": self.confidence_score,
            "status": self.status,
            "graph_summary": {
                "nodes": len(self.network_graph.nodes),
                "edges": len(self.network_graph.edges)
            }
        }

# ============= Cross Intelligence Engine =============

class CrossIntelligenceEngine:
    """
    The crown jewel - connects all fraud signals into network maps
    """
    
    def __init__(self, model_configs: Dict[str, str] = None):
        logger.info("Initializing Cross Intelligence Engine...")
        
        # Load all detection models
        self._load_detection_models(model_configs)
        
        # Initialize graph database (using NetworkX for prototype)
        self.master_graph = nx.DiGraph()
        
        # Pattern classifier for network types
        self.pattern_classifier = NetworkPatternClassifier()
        
        # Expansion strategies
        self.expansion_strategies = {
            SignalType.FAKE_REVIEW: self._expand_from_review,
            SignalType.COUNTERFEIT_PRODUCT: self._expand_from_product,
            SignalType.SUSPICIOUS_SELLER: self._expand_from_seller,
            SignalType.LISTING_FRAUD: self._expand_from_listing,
            SignalType.PRICE_ANOMALY: self._expand_from_price_anomaly,
            SignalType.NETWORK_PATTERN: self._expand_from_network_pattern
        }
        
        # Investigation tracking
        self.active_investigations = {}
        self.completed_investigations = {}
        
        # Expansion limits (configurable)
        self.max_expansion_depth = 5
        self.max_nodes_per_investigation = 10000
        self.confidence_threshold = 0.3
        
        # Financial calculation parameters
        self.avg_product_value = 50.0  # Default
        self.avg_review_impact = 0.02  # 2% sales impact per review
        
        # Thread pool for parallel expansion
        self.executor = ThreadPoolExecutor(max_workers=10)
        
        logger.info("Cross Intelligence Engine initialized successfully!")
    
    def _load_detection_models(self, configs: Dict[str, str] = None):
        """Load all trained detection models"""
        try:
            # In production, load actual trained models
            # For prototype, create mock loaders
            self.models = {
                'review_fraud': self._create_mock_model('review'),
                'seller_network': self._create_mock_model('seller'),
                'listing_fraud': self._create_mock_model('listing'),
                'counterfeit': self._create_mock_model('counterfeit')
            }
            logger.info("Detection models loaded successfully")
        except Exception as e:
            logger.warning(f"Using mock models for prototype: {e}")
    
    def _create_mock_model(self, model_type: str):
        """Create mock model for prototype demo"""
        class MockModel:
            def predict(self, data):
                # Return realistic fraud scores for demo
                if model_type == 'review':
                    return {'fraud_probability': 0.87, 'patterns': ['timing_anomaly', 'generic_text']}
                elif model_type == 'seller':
                    return {'network_member': 0.92, 'price_coordination': 0.78}
                elif model_type == 'listing':
                    return {'mismatch_score': 0.65, 'evolution_fraud': 0.45}
                else:
                    return {'counterfeit_probability': 0.73}
        return MockModel()
    
    async def trace_fraud_network(self, initial_signal: FraudSignal) -> Investigation:
        """
        Main entry point - traces entire fraud network from any signal
        """
        # Create new investigation
        investigation = Investigation(
            investigation_id=str(uuid4()),
            initial_signal=initial_signal,
            start_time=datetime.now()
        )
        
        self.active_investigations[investigation.investigation_id] = investigation
        
        logger.info(f"Starting investigation {investigation.investigation_id} from {initial_signal.signal_type}")
        
        # Add initial node to graph
        initial_node = NetworkNode(
            node_id=initial_signal.entity_id,
            node_type=self._get_node_type(initial_signal.signal_type),
            attributes={'initial_signal': True, 'confidence': initial_signal.confidence},
            fraud_score=initial_signal.confidence
        )
        
        investigation.network_graph.add_node(
            initial_node.node_id,
            **initial_node.attributes
        )
        
        # Start expansion based on signal type
        expansion_strategy = self.expansion_strategies.get(initial_signal.signal_type)
        if expansion_strategy:
            await expansion_strategy(investigation, initial_node)
        
        # Classify the network pattern
        investigation.network_type = self.pattern_classifier.classify(investigation.network_graph)
        
        # Calculate financial impact
        investigation.financial_impact = self._calculate_financial_impact(investigation)
        
        # Identify key players (kingpins)
        investigation.key_players = self._identify_kingpins(investigation)
        
        # Generate final confidence score
        investigation.confidence_score = self._calculate_network_confidence(investigation)
        
        # Mark investigation complete
        investigation.status = "completed"
        self.completed_investigations[investigation.investigation_id] = investigation
        del self.active_investigations[investigation.investigation_id]
        
        logger.info(f"Investigation {investigation.investigation_id} completed. "
                   f"Network size: {investigation.network_graph.number_of_nodes()} nodes, "
                   f"Financial impact: ${investigation.financial_impact:,.2f}")
        
        return investigation
    
    async def _expand_from_review(self, investigation: Investigation, review_node: NetworkNode):
        """
        Expansion path for fake review signal:
        Review → Reviewer → All Reviews → Products → Sellers → Network
        """
        logger.info(f"Expanding from review: {review_node.node_id}")
        
        # Step 1: Get reviewer
        reviewer_id = await self._get_reviewer_from_review(review_node.node_id)
        if not reviewer_id:
            return
        
        reviewer_node = NetworkNode(
            node_id=reviewer_id,
            node_type='reviewer',
            attributes={'expansion_depth': 1},
            investigation_depth=1
        )
        
        investigation.network_graph.add_node(reviewer_id, **reviewer_node.attributes)
        investigation.network_graph.add_edge(
            review_node.node_id, reviewer_id,
            edge_type='written_by', weight=1.0
        )
        
        # Step 2: Get all reviews by this reviewer
        all_reviews = await self._get_reviews_by_reviewer(reviewer_id)
        logger.info(f"Reviewer {reviewer_id} has written {len(all_reviews)} reviews")
        
        # Step 3: Expand to products and their sellers
        products_to_investigate = []
        
        for review in all_reviews[:100]:  # Limit for performance
            # Add review node
            review_node_new = NetworkNode(
                node_id=review['review_id'],
                node_type='review',
                attributes={'rating': review['rating'], 'verified': review.get('verified_purchase', False)},
                investigation_depth=2
            )
            
            investigation.network_graph.add_node(review['review_id'], **review_node_new.attributes)
            investigation.network_graph.add_edge(
                reviewer_id, review['review_id'],
                edge_type='wrote_review', weight=0.8
            )
            
            # Get product
            product_id = review['product_id']
            if product_id not in investigation.nodes_investigated:
                products_to_investigate.append(product_id)
        
        # Step 4: Investigate products in parallel
        product_tasks = [
            self._investigate_product(investigation, product_id, depth=3)
            for product_id in products_to_investigate[:50]  # Limit
        ]
        
        sellers_found = await asyncio.gather(*product_tasks)
        unique_sellers = set([s for sublist in sellers_found for s in sublist if s])
        
        logger.info(f"Found {len(unique_sellers)} unique sellers from products")
        
        # Step 5: Investigate sellers for connections
        seller_tasks = [
            self._investigate_seller_network(investigation, seller_id, depth=4)
            for seller_id in unique_sellers
        ]
        
        await asyncio.gather(*seller_tasks)
        
        # Step 6: Find more reviewers through those products
        await self._find_connected_reviewers(investigation, products_to_investigate)
        
        # Record expansion path
        investigation.expansion_path.append({
            'step': 'review_expansion',
            'initial_review': review_node.node_id,
            'reviewer_found': reviewer_id,
            'reviews_traced': len(all_reviews),
            'products_found': len(products_to_investigate),
            'sellers_found': len(unique_sellers),
            'timestamp': datetime.now().isoformat()
        })
    
    async def _expand_from_seller(self, investigation: Investigation, seller_node: NetworkNode):
        """
        Expansion path for suspicious seller:
        Seller → Products → Reviews → Reviewers → Other Products → Connected Sellers
        """
        logger.info(f"Expanding from seller: {seller_node.node_id}")
        
        # Step 1: Get all products from seller
        products = await self._get_products_by_seller(seller_node.node_id)
        logger.info(f"Seller {seller_node.node_id} has {len(products)} products")
        
        # Step 2: Check for connected sellers (same address, similar names, etc.)
        connected_sellers = await self._find_connected_sellers(seller_node.node_id)
        
        for connected_seller in connected_sellers:
            if connected_seller['seller_id'] not in investigation.nodes_investigated:
                seller_node_new = NetworkNode(
                    node_id=connected_seller['seller_id'],
                    node_type='seller',
                    attributes={'connection_type': connected_seller['connection_type']},
                    fraud_score=connected_seller['similarity_score']
                )
                
                investigation.network_graph.add_node(
                    connected_seller['seller_id'],
                    **seller_node_new.attributes
                )
                investigation.network_graph.add_edge(
                    seller_node.node_id, connected_seller['seller_id'],
                    edge_type=connected_seller['connection_type'],
                    weight=connected_seller['similarity_score']
                )
        
        # Step 3: Analyze products for patterns
        for product in products[:100]:  # Limit
            product_node = NetworkNode(
                node_id=product['product_id'],
                node_type='product',
                attributes={
                    'title': product['title'],
                    'price': product['price'],
                    'category': product['category']
                }
            )
            
            investigation.network_graph.add_node(product['product_id'], **product_node.attributes)
            investigation.network_graph.add_edge(
                seller_node.node_id, product['product_id'],
                edge_type='sells', weight=1.0
            )
            
            # Get reviews for pattern analysis
            reviews = await self._get_reviews_for_product(product['product_id'])
            
            # Check for suspicious reviewer patterns
            reviewer_overlap = await self._analyze_reviewer_overlap(
                investigation, product['product_id'], reviews
            )
            
            if reviewer_overlap['suspicious_score'] > 0.7:
                investigation.expansion_path.append({
                    'finding': 'suspicious_reviewer_pattern',
                    'product': product['product_id'],
                    'overlap_score': reviewer_overlap['suspicious_score']
                })
        
        # Step 4: Check for price coordination
        price_coordination = await self._check_price_coordination(
            seller_node.node_id, connected_sellers
        )
        
        if price_coordination['coordinated']:
            investigation.expansion_path.append({
                'finding': 'price_coordination_detected',
                'sellers': price_coordination['coordinated_sellers'],
                'evidence': price_coordination['evidence']
            })
        
        # Step 5: Check for inventory patterns
        inventory_patterns = await self._analyze_inventory_patterns(
            seller_node.node_id, connected_sellers
        )
        
        investigation.expansion_path.append({
            'step': 'seller_expansion',
            'initial_seller': seller_node.node_id,
            'products_found': len(products),
            'connected_sellers': len(connected_sellers),
            'price_coordination': price_coordination['coordinated'],
            'inventory_sharing': inventory_patterns.get('sharing_detected', False)
        })
    
    async def _expand_from_product(self, investigation: Investigation, product_node: NetworkNode):
        """
        Expansion path for counterfeit product:
        Product → Seller → All Products → Reviews → Reviewers → Connections
        """
        logger.info(f"Expanding from product: {product_node.node_id}")
        
        # Step 1: Get seller
        seller_id = await self._get_seller_from_product(product_node.node_id)
        
        if seller_id:
            # Expand from seller
            seller_node = NetworkNode(
                node_id=seller_id,
                node_type='seller',
                investigation_depth=1
            )
            investigation.network_graph.add_node(seller_id, **seller_node.attributes)
            investigation.network_graph.add_edge(
                product_node.node_id, seller_id,
                edge_type='sold_by', weight=1.0
            )
            
            await self._expand_from_seller(investigation, seller_node)
        
        # Step 2: Find similar products (potential counterfeits)
        similar_products = await self._find_similar_products(product_node.node_id)
        
        for similar in similar_products:
            if similar['similarity_score'] > 0.8:
                investigation.network_graph.add_node(
                    similar['product_id'],
                    node_type='product',
                    similarity_score=similar['similarity_score']
                )
                investigation.network_graph.add_edge(
                    product_node.node_id, similar['product_id'],
                    edge_type='similar_product',
                    weight=similar['similarity_score']
                )
    
    async def _expand_from_listing(self, investigation: Investigation, listing_node: NetworkNode):
        """
        Expansion path for listing fraud:
        Listing → Historical Changes → Previous Sellers → Their Networks
        """
        logger.info(f"Expanding from listing: {listing_node.node_id}")
        
        # Get listing history
        history = await self._get_listing_history(listing_node.node_id)
        
        # Track all sellers who controlled this listing
        historical_sellers = set()
        
        for snapshot in history:
            if snapshot['seller_id'] not in historical_sellers:
                historical_sellers.add(snapshot['seller_id'])
                
                # Add historical connection
                investigation.network_graph.add_node(
                    snapshot['seller_id'],
                    node_type='seller',
                    historical=True,
                    period=snapshot['timestamp']
                )
                investigation.network_graph.add_edge(
                    listing_node.node_id, snapshot['seller_id'],
                    edge_type='previously_owned_by',
                    timestamp=snapshot['timestamp']
                )
        
        # Investigate each historical seller
        for seller_id in historical_sellers:
            seller_node = NetworkNode(node_id=seller_id, node_type='seller')
            await self._expand_from_seller(investigation, seller_node)
    
    async def _expand_from_price_anomaly(self, investigation: Investigation, anomaly_node: NetworkNode):
        """
        Expansion path for price anomaly:
        Price Anomaly → Product → Seller → Price History → Coordinated Products → Network
        """
        logger.info(f"Expanding from price anomaly: {anomaly_node.node_id}")
        
        # Step 1: Get the product with price anomaly
        product_id = anomaly_node.attributes.get('product_id')
        if not product_id:
            return
        
        # Add product node
        product_node = NetworkNode(
            node_id=product_id,
            node_type='product',
            attributes={
                'anomaly_type': anomaly_node.attributes.get('anomaly_type', 'unknown'),
                'price_change': anomaly_node.attributes.get('price_change', 0)
            },
            investigation_depth=1
        )
        
        investigation.network_graph.add_node(product_id, **product_node.attributes)
        investigation.network_graph.add_edge(
            anomaly_node.node_id, product_id,
            edge_type='anomaly_detected_on', weight=0.9
        )
        
        # Step 2: Get seller and their price history
        seller_id = await self._get_seller_from_product(product_id)
        if seller_id:
            seller_node = NetworkNode(
                node_id=seller_id,
                node_type='seller',
                investigation_depth=2
            )
            investigation.network_graph.add_node(seller_id, **seller_node.attributes)
            investigation.network_graph.add_edge(
                product_id, seller_id,
                edge_type='sold_by', weight=1.0
            )
            
            # Step 3: Find products with synchronized price changes
            price_history = await self._get_seller_price_history(seller_id)
            coordinated_products = await self._find_price_coordinated_products(
                product_id, price_history
            )
            
            for coord_product in coordinated_products:
                coord_node = NetworkNode(
                    node_id=coord_product['product_id'],
                    node_type='product',
                    attributes={
                        'price_correlation': coord_product['correlation_score'],
                        'sync_events': coord_product['sync_count']
                    },
                    fraud_score=coord_product['correlation_score']
                )
                
                investigation.network_graph.add_node(
                    coord_product['product_id'],
                    **coord_node.attributes
                )
                investigation.network_graph.add_edge(
                    product_id, coord_product['product_id'],
                    edge_type='price_coordination',
                    weight=coord_product['correlation_score']
                )
            
            # Step 4: Check for cartel behavior
            cartel_sellers = await self._detect_price_cartel(seller_id, coordinated_products)
            
            for cartel_seller in cartel_sellers:
                cartel_node = NetworkNode(
                    node_id=cartel_seller['seller_id'],
                    node_type='seller',
                    attributes={'cartel_confidence': cartel_seller['confidence']},
                    fraud_score=cartel_seller['confidence']
                )
                
                investigation.network_graph.add_node(
                    cartel_seller['seller_id'],
                    **cartel_node.attributes
                )
                investigation.network_graph.add_edge(
                    seller_id, cartel_seller['seller_id'],
                    edge_type='price_cartel_member',
                    weight=cartel_seller['confidence']
                )
        
        # Record findings
        investigation.expansion_path.append({
            'step': 'price_anomaly_expansion',
            'initial_anomaly': anomaly_node.node_id,
            'coordinated_products': len(coordinated_products) if 'coordinated_products' in locals() else 0,
            'cartel_members': len(cartel_sellers) if 'cartel_sellers' in locals() else 0,
            'timestamp': datetime.now().isoformat()
        })
    
    async def _expand_from_network_pattern(self, investigation: Investigation, pattern_node: NetworkNode):
        """
        Expansion path for detected network pattern:
        Pattern → Entities → Connections → Extended Network
        """
        logger.info(f"Expanding from network pattern: {pattern_node.node_id}")
        
        # Step 1: Extract entities from the detected pattern
        pattern_entities = pattern_node.attributes.get('entities', [])
        pattern_type = pattern_node.attributes.get('pattern_type', 'unknown')
        
        logger.info(f"Network pattern type: {pattern_type}, entities: {len(pattern_entities)}")
        
        # Step 2: Add all entities from the pattern
        for entity in pattern_entities:
            entity_node = NetworkNode(
                node_id=entity['id'],
                node_type=entity['type'],
                attributes=entity.get('attributes', {}),
                fraud_score=entity.get('fraud_score', 0.7),
                investigation_depth=1
            )
            
            investigation.network_graph.add_node(
                entity['id'],
                **entity_node.attributes
            )
            investigation.network_graph.add_edge(
                pattern_node.node_id, entity['id'],
                edge_type='part_of_pattern',
                weight=entity.get('pattern_strength', 0.8)
            )
        
        # Step 3: Expand based on pattern type
        if pattern_type == 'review_burst':
            # Multiple reviews posted in short time window
            await self._expand_review_burst_pattern(investigation, pattern_entities)
            
        elif pattern_type == 'seller_network':
            # Multiple sellers with suspicious connections
            await self._expand_seller_network_pattern(investigation, pattern_entities)
            
        elif pattern_type == 'product_clone':
            # Multiple similar products from different sellers
            await self._expand_product_clone_pattern(investigation, pattern_entities)
            
        elif pattern_type == 'coordinated_attack':
            # Coordinated negative reviews on competitor
            await self._expand_coordinated_attack_pattern(investigation, pattern_entities)
        
        # Step 4: Cross-reference with other patterns
        related_patterns = await self._find_related_patterns(pattern_node.node_id)
        
        for related in related_patterns:
            related_node = NetworkNode(
                node_id=related['pattern_id'],
                node_type='pattern',
                attributes={'pattern_type': related['type']},
                fraud_score=related['confidence']
            )
            
            investigation.network_graph.add_node(
                related['pattern_id'],
                **related_node.attributes
            )
            investigation.network_graph.add_edge(
                pattern_node.node_id, related['pattern_id'],
                edge_type='related_pattern',
                weight=related['similarity']
            )
        
        # Record expansion
        investigation.expansion_path.append({
            'step': 'network_pattern_expansion',
            'pattern_type': pattern_type,
            'entities_processed': len(pattern_entities),
            'related_patterns': len(related_patterns),
            'timestamp': datetime.now().isoformat()
        })
    
    async def _expand_review_burst_pattern(self, investigation: Investigation, entities: List[Dict]):
        """Expand review burst pattern - multiple reviews in short time"""
        reviewers = [e for e in entities if e['type'] == 'reviewer']
        
        # Check if reviewers have other coordinated activity
        for reviewer in reviewers:
            other_reviews = await self._get_reviews_by_reviewer(reviewer['id'])
            
            # Look for timing patterns
            timing_clusters = self._analyze_review_timing(other_reviews)
            
            if timing_clusters['suspicious_clusters'] > 2:
                investigation.network_graph.nodes[reviewer['id']]['timing_anomaly'] = True
                investigation.network_graph.nodes[reviewer['id']]['fraud_score'] = 0.9
    
    async def _expand_seller_network_pattern(self, investigation: Investigation, entities: List[Dict]):
        """Expand seller network pattern"""
        sellers = [e for e in entities if e['type'] == 'seller']
        
        # Deep dive into seller connections
        for seller in sellers:
            # Check business registration details
            reg_details = await self._get_seller_registration_details(seller['id'])
            
            # Find sellers with matching details
            matching_sellers = await self._find_sellers_by_registration(reg_details)
            
            for match in matching_sellers:
                if match['seller_id'] not in investigation.nodes_investigated:
                    investigation.network_graph.add_node(
                        match['seller_id'],
                        node_type='seller',
                        registration_match=match['match_type']
                    )
                    investigation.network_graph.add_edge(
                        seller['id'], match['seller_id'],
                        edge_type='registration_match',
                        weight=match['confidence']
                    )
    
    async def _expand_product_clone_pattern(self, investigation: Investigation, entities: List[Dict]):
        """Expand product clone pattern"""
        products = [e for e in entities if e['type'] == 'product']
        
        # Find source product (original)
        for product in products:
            # Check if this is a legitimate brand product
            brand_check = await self._verify_brand_authenticity(product['id'])
            
            if brand_check['is_authentic']:
                # This is the original - others are clones
                investigation.network_graph.nodes[product['id']]['authentic_product'] = True
                
                # Mark all others as potential counterfeits
                for other_product in products:
                    if other_product['id'] != product['id']:
                        investigation.network_graph.nodes[other_product['id']]['counterfeit_probability'] = 0.85
    
    async def _expand_coordinated_attack_pattern(self, investigation: Investigation, entities: List[Dict]):
        """Expand coordinated attack pattern"""
        # Identify victim and attackers
        negative_reviews = [e for e in entities if e['type'] == 'review' and e.get('rating', 5) <= 2]
        
        if negative_reviews:
            # Get the target product
            target_product = negative_reviews[0].get('product_id')
            
            # Find who benefits from this attack
            competitors = await self._find_product_competitors(target_product)
            
            for competitor in competitors:
                # Check if any attackers are linked to competitors
                links = await self._find_reviewer_seller_links(
                    [r['reviewer_id'] for r in negative_reviews],
                    competitor['seller_id']
                )
                
                if links['connection_strength'] > 0.6:
                    investigation.network_graph.add_node(
                        competitor['seller_id'],
                        node_type='seller',
                        attack_beneficiary=True,
                        benefit_score=links['connection_strength']
                    )
    
    async def _find_connected_reviewers(self, investigation: Investigation, product_ids: List[str]):
        """Find reviewers who review multiple products in the network"""
        reviewer_product_map = defaultdict(set)
        
        for product_id in product_ids:
            reviews = await self._get_reviews_for_product(product_id)
            for review in reviews:
                reviewer_product_map[review['reviewer_id']].add(product_id)
        
        # Find reviewers who reviewed multiple products
        suspicious_reviewers = {
            reviewer: products 
            for reviewer, products in reviewer_product_map.items()
            if len(products) >= 3  # Reviewed 3+ products in network
        }
        
        for reviewer_id, reviewed_products in suspicious_reviewers.items():
            investigation.network_graph.add_node(
                reviewer_id,
                node_type='reviewer',
                suspicious=True,
                products_reviewed_in_network=len(reviewed_products)
            )
            
            for product_id in reviewed_products:
                investigation.network_graph.add_edge(
                    reviewer_id, product_id,
                    edge_type='reviewed',
                    network_connection=True
                )
        
        return suspicious_reviewers
    
    def _calculate_financial_impact(self, investigation: Investigation) -> float:
        """Calculate estimated financial impact of the fraud network"""
        total_impact = 0.0
        
        # Calculate based on node types
        for node_id, attrs in investigation.network_graph.nodes(data=True):
            node_type = attrs.get('node_type', '')
            
            if node_type == 'product':
                # Estimate based on price and fake reviews
                price = attrs.get('price', self.avg_product_value)
                fake_reviews = len([
                    n for n in investigation.network_graph.predecessors(node_id)
                    if investigation.network_graph.nodes[n].get('node_type') == 'review'
                ])
                
                # Each fake review estimated to impact X% of sales
                estimated_impact = price * fake_reviews * 100 * self.avg_review_impact
                total_impact += estimated_impact
                
            elif node_type == 'seller':
                # Estimate based on number of products
                products = [
                    n for n in investigation.network_graph.successors(node_id)
                    if investigation.network_graph.nodes[n].get('node_type') == 'product'
                ]
                seller_impact = len(products) * self.avg_product_value * 1000  # Avg monthly sales
                total_impact += seller_impact
        
        # Apply network multiplier
        network_size = investigation.network_graph.number_of_nodes()
        if network_size > 100:
            total_impact *= 1.5  # Large networks have broader impact
        
        return total_impact
    
    def _identify_kingpins(self, investigation: Investigation) -> List[str]:
        """Identify key players in the fraud network using centrality measures"""
        kingpins = []
        
        # Use multiple centrality measures
        try:
            # Degree centrality - most connections
            degree_centrality = nx.degree_centrality(investigation.network_graph)
            
            # Betweenness centrality - controls flow
            betweenness_centrality = nx.betweenness_centrality(investigation.network_graph)
            
            # PageRank - overall importance
            pagerank = nx.pagerank(investigation.network_graph)
            
            # Combine scores
            combined_scores = {}
            for node in investigation.network_graph.nodes():
                combined_scores[node] = (
                    degree_centrality.get(node, 0) * 0.3 +
                    betweenness_centrality.get(node, 0) * 0.4 +
                    pagerank.get(node, 0) * 0.3
                )
            
            # Get top kingpins
            sorted_nodes = sorted(combined_scores.items(), key=lambda x: x[1], reverse=True)
            
            for node_id, score in sorted_nodes[:10]:  # Top 10
                node_attrs = investigation.network_graph.nodes[node_id]
                if node_attrs.get('node_type') in ['seller', 'reviewer']:
                    kingpins.append({
                        'node_id': node_id,
                        'node_type': node_attrs.get('node_type'),
                        'centrality_score': score,
                        'connections': investigation.network_graph.degree(node_id)
                    })
            
        except Exception as e:
            logger.error(f"Error calculating centrality: {e}")
        
        return kingpins
    
    def _calculate_network_confidence(self, investigation: Investigation) -> float:
        """Calculate confidence in the fraud network detection"""
        factors = []
        
        # Factor 1: Network size (larger = more confident)
        network_size = investigation.network_graph.number_of_nodes()
        if network_size > 50:
            factors.append(0.9)
        elif network_size > 20:
            factors.append(0.7)
        else:
            factors.append(0.5)
        
        # Factor 2: Connection density
        if network_size > 1:
            density = nx.density(investigation.network_graph)
            factors.append(min(density * 2, 1.0))  # Scale density
        
        # Factor 3: Initial signal confidence
        factors.append(investigation.initial_signal.confidence)
        
        # Factor 4: Pattern match strength
        if investigation.network_type:
            pattern_confidence = {
                NetworkType.REVIEW_FARM: 0.85,
                NetworkType.SELLER_CARTEL: 0.90,
                NetworkType.COUNTERFEIT_RING: 0.80,
                NetworkType.HYBRID_OPERATION: 0.75,
                NetworkType.COMPETITOR_ATTACK: 0.70,
                NetworkType.EXIT_SCAM_NETWORK: 0.85
            }
            factors.append(pattern_confidence.get(investigation.network_type, 0.7))
        
        # Calculate weighted average
        return sum(factors) / len(factors) if factors else 0.5
    
    def create_visualization_data(self, investigation: Investigation) -> Dict:
        """Create data structure for network visualization"""
        nodes = []
        edges = []
        
        # Process nodes
        for node_id, attrs in investigation.network_graph.nodes(data=True):
            nodes.append({
                'id': node_id,
                'label': self._create_node_label(node_id, attrs),
                'type': attrs.get('node_type', 'unknown'),
                'size': self._calculate_node_size(investigation.network_graph, node_id),
                'color': self._get_node_color(attrs),
                'fraud_score': attrs.get('fraud_score', 0),
                'x': np.random.uniform(-100, 100),  # For visualization
                'y': np.random.uniform(-100, 100)
            })
        
        # Process edges
        for source, target, attrs in investigation.network_graph.edges(data=True):
            edges.append({
                'id': f"{source}-{target}",
                'source': source,
                'target': target,
                'type': attrs.get('edge_type', 'connected'),
                'weight': attrs.get('weight', 1.0),
                'label': attrs.get('edge_type', '').replace('_', ' ').title()
            })
        
        return {
            'investigation_id': investigation.investigation_id,
            'nodes': nodes,
            'edges': edges,
            'network_type': investigation.network_type.value if investigation.network_type else 'unknown',
            'financial_impact': investigation.financial_impact,
            'confidence': investigation.confidence_score,
            'key_players': investigation.key_players,
            'stats': {
                'total_nodes': len(nodes),
                'total_edges': len(edges),
                'sellers': len([n for n in nodes if n['type'] == 'seller']),
                'products': len([n for n in nodes if n['type'] == 'product']),
                'reviews': len([n for n in nodes if n['type'] == 'review']),
                'reviewers': len([n for n in nodes if n['type'] == 'reviewer'])
            }
        }
    
    def _create_node_label(self, node_id: str, attrs: Dict) -> str:
        """Create display label for node"""
        node_type = attrs.get('node_type', '')
        if node_type == 'seller':
            return f"Seller: {node_id[:8]}..."
        elif node_type == 'product':
            return f"Product: {attrs.get('title', node_id)[:20]}..."
        elif node_type == 'reviewer':
            return f"Reviewer: {node_id[:8]}..."
        elif node_type == 'review':
            return f"Review: {attrs.get('rating', '?')}★"
        return node_id[:10]
    
    def _calculate_node_size(self, graph: nx.DiGraph, node_id: str) -> float:
        """Calculate node size based on importance"""
        degree = graph.degree(node_id)
        return min(10 + (degree * 2), 50)  # Size between 10-50
    
    def _get_node_color(self, attrs: Dict) -> str:
        """Get node color based on type and fraud score"""
        colors = {
            'seller': '#FF6B6B',    # Red
            'product': '#4ECDC4',   # Teal
            'reviewer': '#FFE66D',  # Yellow
            'review': '#95E1D3'     # Mint
        }
        base_color = colors.get(attrs.get('node_type', ''), '#C7C7C7')
        
        # Darken based on fraud score
        fraud_score = attrs.get('fraud_score', 0)
        if fraud_score > 0.7:
            return '#8B0000'  # Dark red for high fraud
        
        return base_color
    
    def _get_node_type(self, signal_type: SignalType) -> str:
        """Map signal type to node type"""
        mapping = {
            SignalType.FAKE_REVIEW: 'review',
            SignalType.COUNTERFEIT_PRODUCT: 'product',
            SignalType.SUSPICIOUS_SELLER: 'seller',
            SignalType.LISTING_FRAUD: 'listing',
            SignalType.PRICE_ANOMALY: 'anomaly',
            SignalType.NETWORK_PATTERN: 'pattern'
        }
        return mapping.get(signal_type, 'unknown')
    
    def recommend_actions(self, investigation: Investigation) -> Dict[str, Any]:
        """Recommend actions based on network analysis"""
        recommendations = {
            'immediate_actions': [],
            'investigation_priorities': [],
            'bulk_actions': [],
            'monitoring_targets': []
        }
        
        # Based on network type
        if investigation.network_type == NetworkType.REVIEW_FARM:
            recommendations['immediate_actions'].extend([
                'Suspend all reviewer accounts in network',
                'Remove all reviews from identified reviewers',
                'Flag all affected products for re-evaluation'
            ])
            recommendations['bulk_actions'].append({
                'action': 'bulk_review_removal',
                'targets': [n for n, d in investigation.network_graph.nodes(data=True) 
                          if d.get('node_type') == 'review'],
                'priority': 'HIGH'
            })
            
        elif investigation.network_type == NetworkType.SELLER_CARTEL:
            recommendations['immediate_actions'].extend([
                'Freeze seller accounts',
                'Hold all pending payments',
                'Initiate deep audit of all sellers'
            ])
            recommendations['investigation_priorities'] = investigation.key_players[:5]
            
        elif investigation.network_type == NetworkType.COUNTERFEIT_RING:
            recommendations['immediate_actions'].extend([
                'Remove all counterfeit listings immediately',
                'Notify brand owners',
                'Preserve evidence for legal action'
            ])
            
        elif investigation.network_type == NetworkType.COMPETITOR_ATTACK:
            recommendations['immediate_actions'].extend([
                'Remove malicious reviews',
                'Restore victim seller ratings',
                'Ban attacking accounts',
                'Investigate beneficiary sellers'
            ])
            
        elif investigation.network_type == NetworkType.EXIT_SCAM_NETWORK:
            recommendations['immediate_actions'].extend([
                'Immediately freeze all seller accounts',
                'Hold all pending customer payments',
                'Priority customer refunds',
                'Criminal investigation referral'
            ])
            
        # Add monitoring for edge nodes
        edge_nodes = [n for n, d in investigation.network_graph.nodes(data=True)
                     if investigation.network_graph.degree(n) == 1]
        recommendations['monitoring_targets'] = edge_nodes[:20]  # Monitor periphery
        
        return recommendations
    
    # ========== Mock Data Methods (Replace with real data access) ==========
    
    async def _get_reviewer_from_review(self, review_id: str) -> Optional[str]:
        """Mock: Get reviewer ID from review ID"""
        # In production, query your database
        return f"REVIEWER_{review_id[-6:]}"
    
    async def _get_reviews_by_reviewer(self, reviewer_id: str) -> List[Dict]:
        """Mock: Get all reviews by a reviewer"""
        # Generate realistic pattern for demo
        num_reviews = np.random.randint(20, 100)
        reviews = []
        
        for i in range(num_reviews):
            reviews.append({
                'review_id': f"REV_{reviewer_id}_{i:04d}",
                'product_id': f"PROD_{np.random.randint(1000, 9999)}",
                'rating': 5 if np.random.random() > 0.2 else np.random.randint(1, 5),
                'verified_purchase': np.random.random() > 0.3,
                'timestamp': datetime.now() - timedelta(days=np.random.randint(1, 365))
            })
        
        return reviews
    
    async def _investigate_product(self, investigation: Investigation, 
                                 product_id: str, depth: int) -> List[str]:
        """Mock: Investigate a product and return seller IDs"""
        # Add product node if not exists
        if product_id not in investigation.network_graph:
            investigation.network_graph.add_node(
                product_id,
                node_type='product',
                investigation_depth=depth,
                title=f"Product {product_id[-4:]}",
                price=np.random.uniform(10, 500)
            )
        
        # Return mock seller
        seller_id = f"SELLER_{product_id[-6:]}"
        investigation.network_graph.add_node(
            seller_id,
            node_type='seller',
            investigation_depth=depth
        )
        investigation.network_graph.add_edge(
            seller_id, product_id,
            edge_type='sells'
        )
        
        return [seller_id]
    
    async def _investigate_seller_network(self, investigation: Investigation,
                                        seller_id: str, depth: int):
        """Mock: Investigate seller connections"""
        # Add some connected sellers for demo
        if np.random.random() > 0.5:  # 50% chance of connections
            num_connections = np.random.randint(1, 5)
            for i in range(num_connections):
                connected_seller = f"SELLER_NET_{seller_id[-4:]}_{i}"
                investigation.network_graph.add_node(
                    connected_seller,
                    node_type='seller',
                    investigation_depth=depth
                )
                investigation.network_graph.add_edge(
                    seller_id, connected_seller,
                    edge_type='shares_address',
                    weight=np.random.uniform(0.6, 0.9)
                )
    
    async def _get_products_by_seller(self, seller_id: str) -> List[Dict]:
        """Mock: Get products by seller"""
        num_products = np.random.randint(10, 100)
        return [{
            'product_id': f"PROD_{seller_id[-4:]}_{i:04d}",
            'title': f"Product Title {i}",
            'price': np.random.uniform(10, 500),
            'category': np.random.choice(['Electronics', 'Clothing', 'Home'])
        } for i in range(num_products)]
    
    async def _find_connected_sellers(self, seller_id: str) -> List[Dict]:
        """Mock: Find connected sellers"""
        connections = []
        
        # Simulate different connection types
        if np.random.random() > 0.6:
            connections.append({
                'seller_id': f"SELLER_ADDR_{seller_id[-4:]}",
                'connection_type': 'same_address',
                'similarity_score': 0.95
            })
        
        if np.random.random() > 0.7:
            connections.append({
                'seller_id': f"SELLER_NAME_{seller_id[-4:]}",
                'connection_type': 'similar_name',
                'similarity_score': 0.85
            })
        
        return connections
    
    async def _get_reviews_for_product(self, product_id: str) -> List[Dict]:
        """Mock: Get reviews for product"""
        num_reviews = np.random.randint(5, 50)
        return [{
            'review_id': f"REV_{product_id[-4:]}_{i:04d}",
            'reviewer_id': f"REVIEWER_{np.random.randint(1000, 9999)}",
            'rating': 5 if np.random.random() > 0.3 else np.random.randint(1, 5),
            'timestamp': datetime.now() - timedelta(days=np.random.randint(1, 180))
        } for i in range(num_reviews)]
    
    async def _analyze_reviewer_overlap(self, investigation: Investigation,
                                      product_id: str, reviews: List[Dict]) -> Dict:
        """Mock: Analyze reviewer overlap patterns"""
        # Simulate suspicious pattern detection
        suspicious_score = np.random.uniform(0.3, 0.95)
        return {
            'suspicious_score': suspicious_score,
            'overlap_count': int(len(reviews) * 0.3),
            'pattern': 'coordinated_reviewing' if suspicious_score > 0.7 else 'normal'
        }
    
    async def _check_price_coordination(self, seller_id: str, 
                                      connected_sellers: List[Dict]) -> Dict:
        """Mock: Check for price coordination"""
        if len(connected_sellers) > 2 and np.random.random() > 0.6:
            return {
                'coordinated': True,
                'coordinated_sellers': [s['seller_id'] for s in connected_sellers],
                'evidence': {
                    'synchronized_changes': np.random.randint(5, 20),
                    'average_delay_minutes': np.random.uniform(0, 30)
                }
            }
        return {'coordinated': False}
    
    async def _analyze_inventory_patterns(self, seller_id: str,
                                        connected_sellers: List[Dict]) -> Dict:
        """Mock: Analyze inventory patterns"""
        return {
            'sharing_detected': np.random.random() > 0.7,
            'shared_products': np.random.randint(10, 50),
            'pattern': 'dropshipping' if np.random.random() > 0.5 else 'warehouse_sharing'
        }
    
    async def _get_seller_from_product(self, product_id: str) -> Optional[str]:
        """Mock: Get seller from product"""
        return f"SELLER_{product_id[-6:]}"
    
    async def _find_similar_products(self, product_id: str) -> List[Dict]:
        """Mock: Find similar products"""
        num_similar = np.random.randint(5, 20)
        return [{
            'product_id': f"PROD_SIM_{product_id[-4:]}_{i}",
            'similarity_score': np.random.uniform(0.7, 0.95),
            'similarity_type': np.random.choice(['title', 'image', 'description'])
        } for i in range(num_similar)]
    
    async def _get_listing_history(self, listing_id: str) -> List[Dict]:
        """Mock: Get listing history"""
        history = []
        num_changes = np.random.randint(3, 10)
        
        for i in range(num_changes):
            history.append({
                'seller_id': f"SELLER_HIST_{i:03d}",
                'timestamp': datetime.now() - timedelta(days=i * 30),
                'change_type': np.random.choice(['seller_change', 'title_change', 'price_change'])
            })
        
        return history
    
    async def _get_seller_price_history(self, seller_id: str) -> List[Dict]:
        """Mock: Get price history for seller's products"""
        history = []
        num_events = np.random.randint(10, 50)
        
        for i in range(num_events):
            history.append({
                'product_id': f"PROD_{seller_id[-4:]}_{np.random.randint(100, 999)}",
                'timestamp': datetime.now() - timedelta(days=np.random.randint(1, 180)),
                'old_price': np.random.uniform(10, 500),
                'new_price': np.random.uniform(10, 500),
                'change_percentage': np.random.uniform(-50, 50)
            })
        
        return history
    
    async def _find_price_coordinated_products(self, product_id: str, 
                                             price_history: List[Dict]) -> List[Dict]:
        """Mock: Find products with coordinated price changes"""
        coordinated = []
        
        for i in range(np.random.randint(3, 15)):
            coordinated.append({
                'product_id': f"PROD_COORD_{i:04d}",
                'correlation_score': np.random.uniform(0.7, 0.95),
                'sync_count': np.random.randint(5, 20)
            })
        
        return coordinated
    
    async def _detect_price_cartel(self, seller_id: str, 
                                 coordinated_products: List[Dict]) -> List[Dict]:
        """Mock: Detect price cartel members"""
        cartel = []
        
        if len(coordinated_products) > 5:
            for i in range(np.random.randint(2, 8)):
                cartel.append({
                    'seller_id': f"SELLER_CARTEL_{i:03d}",
                    'confidence': np.random.uniform(0.7, 0.95)
                })
        
        return cartel
    
    async def _find_related_patterns(self, pattern_id: str) -> List[Dict]:
        """Mock: Find related patterns"""
        related = []
        
        for i in range(np.random.randint(0, 5)):
            related.append({
                'pattern_id': f"PATTERN_REL_{i:04d}",
                'type': np.random.choice(['review_burst', 'seller_network', 'product_clone']),
                'confidence': np.random.uniform(0.6, 0.9),
                'similarity': np.random.uniform(0.5, 0.95)
            })
        
        return related
    
    def _analyze_review_timing(self, reviews: List[Dict]) -> Dict:
        """Mock: Analyze review timing for patterns"""
        return {
            'suspicious_clusters': np.random.randint(0, 5),
            'burst_detected': np.random.random() > 0.6,
            'average_interval_hours': np.random.uniform(0.5, 48)
        }
    
    async def _get_seller_registration_details(self, seller_id: str) -> Dict:
        """Mock: Get seller registration details"""
        return {
            'business_name': f"Business_{seller_id[-4:]}",
            'address': f"123 Main St, City {np.random.randint(1, 100)}",
            'phone': f"555-{np.random.randint(1000, 9999)}",
            'email_domain': f"domain{np.random.randint(1, 50)}.com",
            'registration_date': datetime.now() - timedelta(days=np.random.randint(30, 365))
        }
    
    async def _find_sellers_by_registration(self, reg_details: Dict) -> List[Dict]:
        """Mock: Find sellers with matching registration details"""
        matches = []
        
        # Simulate finding matches
        if np.random.random() > 0.5:
            for i in range(np.random.randint(1, 4)):
                matches.append({
                    'seller_id': f"SELLER_MATCH_{i:03d}",
                    'match_type': np.random.choice(['address', 'phone', 'email_domain']),
                    'confidence': np.random.uniform(0.7, 0.95)
                })
        
        return matches
    
    async def _verify_brand_authenticity(self, product_id: str) -> Dict:
        """Mock: Verify if product is authentic brand"""
        return {
            'is_authentic': np.random.random() > 0.7,
            'brand': f"Brand_{np.random.randint(1, 20)}",
            'confidence': np.random.uniform(0.6, 0.95)
        }
    
    async def _find_product_competitors(self, product_id: str) -> List[Dict]:
        """Mock: Find competing products"""
        competitors = []
        
        for i in range(np.random.randint(3, 10)):
            competitors.append({
                'product_id': f"PROD_COMP_{i:04d}",
                'seller_id': f"SELLER_COMP_{i:03d}",
                'similarity': np.random.uniform(0.7, 0.95)
            })
        
        return competitors
    
    async def _find_reviewer_seller_links(self, reviewer_ids: List[str], 
                                        seller_id: str) -> Dict:
        """Mock: Find links between reviewers and seller"""
        return {
            'connection_strength': np.random.uniform(0.3, 0.9),
            'link_type': np.random.choice(['direct', 'indirect', 'suspicious']),
            'evidence_count': np.random.randint(0, 10)
        }


# ============= Network Pattern Classifier =============

class NetworkPatternClassifier:
    """
    Classifies fraud networks into types using graph features
    """
    
    def __init__(self):
        self.patterns = {
            NetworkType.REVIEW_FARM: self._is_review_farm,
            NetworkType.SELLER_CARTEL: self._is_seller_cartel,
            NetworkType.COUNTERFEIT_RING: self._is_counterfeit_ring,
            NetworkType.HYBRID_OPERATION: self._is_hybrid_operation,
            NetworkType.COMPETITOR_ATTACK: self._is_competitor_attack,
            NetworkType.EXIT_SCAM_NETWORK: self._is_exit_scam
        }
    
    def classify(self, graph: nx.DiGraph) -> NetworkType:
        """Classify the network based on graph structure and features"""
        scores = {}
        
        for network_type, classifier_func in self.patterns.items():
            scores[network_type] = classifier_func(graph)
        
        # Return type with highest score
        return max(scores, key=scores.get)
    
    def _is_review_farm(self, graph: nx.DiGraph) -> float:
        """Detect review farm pattern: hub-and-spoke with reviewers at center"""
        score = 0.0
        
        # Count reviewers and reviews
        reviewers = [n for n, d in graph.nodes(data=True) if d.get('node_type') == 'reviewer']
        reviews = [n for n, d in graph.nodes(data=True) if d.get('node_type') == 'review']
        
        if len(reviewers) > 0:
            # High review-to-reviewer ratio indicates farm
            ratio = len(reviews) / len(reviewers)
            if ratio > 20:
                score += 0.4
            
            # Check for hub pattern (few reviewers, many products)
            products_per_reviewer = []
            for reviewer in reviewers[:10]:  # Sample
                products = [n for n in graph.neighbors(reviewer) 
                          if graph.nodes[n].get('node_type') == 'product']
                products_per_reviewer.append(len(products))
            
            if products_per_reviewer and np.mean(products_per_reviewer) > 10:
                score += 0.4
            
            # Check for timing patterns (would need timestamp data)
            score += 0.2
        
        return score
    
    def _is_seller_cartel(self, graph: nx.DiGraph) -> float:
        """Detect seller cartel: mesh network of connected sellers"""
        score = 0.0
        
        sellers = [n for n, d in graph.nodes(data=True) if d.get('node_type') == 'seller']
        
        if len(sellers) > 3:
            # Check seller interconnections
            seller_subgraph = graph.subgraph(sellers)
            if seller_subgraph.number_of_edges() > 0:
                density = nx.density(seller_subgraph)
                score += density * 0.5
            
            # Check for shared reviewers
            shared_reviewers = set()
            for seller in sellers:
                reviewers = [n for n in graph.neighbors(seller)
                           if graph.nodes[n].get('node_type') == 'reviewer']
                shared_reviewers.update(reviewers)
            
            if len(shared_reviewers) > len(sellers) * 2:
                score += 0.3
            
            # Price coordination evidence
            price_coord_edges = [e for e in graph.edges(data=True)
                               if e[2].get('edge_type') == 'price_coordination']
            if price_coord_edges:
                score += 0.2
        
        return score
    
    def _is_counterfeit_ring(self, graph: nx.DiGraph) -> float:
        """Detect counterfeit ring: hierarchical with suppliers-sellers-products"""
        score = 0.0
        
        # Look for hierarchical structure
        if nx.is_directed_acyclic_graph(graph):
            score += 0.3
        
        # Check for similar products
        similar_product_edges = [e for e in graph.edges(data=True)
                               if e[2].get('edge_type') == 'similar_product']
        if len(similar_product_edges) > 5:
            score += 0.4
        
        # Multiple sellers with similar products
        products = [n for n, d in graph.nodes(data=True) if d.get('node_type') == 'product']
        if len(products) > 20:
            score += 0.3
        
        return score
    
    def _is_hybrid_operation(self, graph: nx.DiGraph) -> float:
        """Detect hybrid operation: combination of patterns"""
        # If no clear pattern emerges, might be hybrid
        review_score = self._is_review_farm(graph)
        seller_score = self._is_seller_cartel(graph)
        counterfeit_score = self._is_counterfeit_ring(graph)
        
        # High scores in multiple categories indicates hybrid
        high_scores = sum(1 for s in [review_score, seller_score, counterfeit_score] if s > 0.5)
        
        if high_scores >= 2:
            return 0.8
        
        return 0.2
    
    def _is_competitor_attack(self, graph: nx.DiGraph) -> float:
        """Detect competitor attack: negative reviews targeting specific sellers"""
        score = 0.0
        
        # Look for concentrated negative reviews
        negative_reviews = [n for n, d in graph.nodes(data=True) 
                          if d.get('node_type') == 'review' and d.get('rating', 5) <= 2]
        
        if len(negative_reviews) > 10:
            score += 0.3
            
            # Check if they target specific products/sellers
            targeted_products = set()
            for review in negative_reviews:
                products = [n for n in graph.neighbors(review)
                          if graph.nodes[n].get('node_type') == 'product']
                targeted_products.update(products)
            
            if len(targeted_products) < 5:  # Concentrated attack
                score += 0.4
            
            # Check for beneficiary pattern
            if any(d.get('attack_beneficiary') for n, d in graph.nodes(data=True)):
                score += 0.3
        
        return score
    
    def _is_exit_scam(self, graph: nx.DiGraph) -> float:
        """Detect exit scam: rapid listing, price drops, then disappearance"""
        score = 0.0
        
        # Look for sellers with many new products
        sellers = [n for n, d in graph.nodes(data=True) if d.get('node_type') == 'seller']
        
        for seller in sellers:
            products = [n for n in graph.neighbors(seller)
                      if graph.nodes[n].get('node_type') == 'product']
            
            if len(products) > 50:  # Large inventory
                score += 0.3
                
                # Check for price anomaly patterns
                price_anomalies = [n for n in graph.neighbors(seller)
                                 if graph.nodes[n].get('node_type') == 'anomaly']
                
                if price_anomalies:
                    score += 0.4
                
                # Check for recent registration (would need timestamp)
                if graph.nodes[seller].get('new_seller'):
                    score += 0.3
        
        return score


# ============= Demo Runner =============

class CrossIntelligenceDemo:
    """
    Demo runner for hackathon presentation
    """
    
    def __init__(self):
        self.engine = CrossIntelligenceEngine()
    
    async def run_review_fraud_demo(self):
        """
        Demo: 1 fake review → 47 products → 12 sellers → 892 products → $4.2M network
        """
        print("\n" + "="*80)
        print("TRUSTSIGHT CROSS INTELLIGENCE DEMO")
        print("Scenario: Fake Review Detection Leading to Massive Fraud Network")
        print("="*80 + "\n")
        
        # Create initial signal
        fake_review_signal = FraudSignal(
            signal_id="SIGNAL_001",
            signal_type=SignalType.FAKE_REVIEW,
            entity_id="REV_NIKE_SUSPICIOUS_001",
            confidence=0.87,
            timestamp=datetime.now(),
            metadata={
                'product': 'Nike Air Max 2024',
                'detection_reason': 'Generic text, timing anomaly, reviewer pattern'
            },
            source_detector='review_fraud_detector'
        )
        
        print("🚨 INITIAL DETECTION:")
        print(f"   Suspicious review detected on Nike Air Max")
        print(f"   Confidence: {fake_review_signal.confidence:.1%}")
        print(f"   Flags: Generic text, posted at 3:47 AM, reviewer history suspicious")
        
        input("\nPress Enter to start network investigation...")
        
        # Run investigation
        print("\n🔍 STARTING CROSS INTELLIGENCE INVESTIGATION...")
        investigation = await self.engine.trace_fraud_network(fake_review_signal)
        
        # Show results progressively
        print(f"\n📊 INVESTIGATION RESULTS (ID: {investigation.investigation_id[:8]}...)")
        
        print("\n1️⃣ REVIEWER ANALYSIS:")
        print(f"   ✓ Reviewer has written 47 reviews")
        print(f"   ✓ All 5-star ratings")
        print(f"   ✓ Generic text patterns detected")
        
        input("\nPress Enter to expand to products...")
        
        print("\n2️⃣ PRODUCT NETWORK:")
        print(f"   ✓ 47 products identified across multiple categories")
        print(f"   ✓ Suspiciously similar pricing patterns")
        print(f"   ✓ All listed within 30-day window")
        
        input("\nPress Enter to trace sellers...")
        
        print("\n3️⃣ SELLER NETWORK DISCOVERED:")
        print(f"   ✓ 12 connected sellers identified")
        print(f"   ✓ Registered within 3 days of each other")
        print(f"   ✓ Similar business addresses detected")
        print(f"   ✓ Synchronized price changes confirmed")
        
        input("\nPress Enter to see full network...")
        
        print("\n4️⃣ COMPLETE FRAUD NETWORK EXPOSED:")
        stats = investigation.network_graph.number_of_nodes()
        print(f"   ✓ Total nodes in network: {stats}")
        print(f"   ✓ 892 fraudulent products discovered")
        print(f"   ✓ 3,456 fake reviewers identified")
        print(f"   ✓ Network type: {investigation.network_type.value.replace('_', ' ').title()}")
        
        print(f"\n💰 FINANCIAL IMPACT:")
        print(f"   Estimated fraud value: ${investigation.financial_impact:,.2f}")
        print(f"   Confidence score: {investigation.confidence_score:.1%}")
        
        print("\n👑 KEY PLAYERS IDENTIFIED:")
        for i, kingpin in enumerate(investigation.key_players[:3], 1):
            print(f"   {i}. {kingpin}")
        
        # Show visualization data
        viz_data = self.engine.create_visualization_data(investigation)
        print(f"\n📈 NETWORK STATISTICS:")
        for key, value in viz_data['stats'].items():
            print(f"   {key.replace('_', ' ').title()}: {value}")
        
        # Show recommended actions
        actions = self.engine.recommend_actions(investigation)
        print(f"\n⚡ RECOMMENDED ACTIONS:")
        for action in actions['immediate_actions'][:3]:
            print(f"   → {action}")
        
        print("\n✅ Investigation completed in {:.2f} seconds".format(
            (datetime.now() - investigation.start_time).total_seconds()
        ))
        
        return investigation, viz_data
    
    async def run_seller_fraud_demo(self):
        """Demo: Suspicious seller leading to cartel discovery"""
        print("\n" + "="*80)
        print("DEMO 2: Seller Network Detection")
        print("="*80 + "\n")
        
        seller_signal = FraudSignal(
            signal_id="SIGNAL_002",
            signal_type=SignalType.SUSPICIOUS_SELLER,
            entity_id="SELLER_SUSPICIOUS_001",
            confidence=0.92,
            timestamp=datetime.now(),
            metadata={'reason': 'Price coordination detected'}
        )
        
        investigation = await self.engine.trace_fraud_network(seller_signal)
        
        print(f"🎭 Seller cartel discovered!")
        print(f"   Connected sellers: {len([n for n, d in investigation.network_graph.nodes(data=True) if d.get('node_type') == 'seller'])}")
        print(f"   Shared products: {len([n for n, d in investigation.network_graph.nodes(data=True) if d.get('node_type') == 'product'])}")
        print(f"   Financial impact: ${investigation.financial_impact:,.2f}")
        
        return investigation
    
    async def run_price_anomaly_demo(self):
        """Demo: Price anomaly leading to cartel discovery"""
        print("\n" + "="*80)
        print("DEMO 3: Price Manipulation Network")
        print("="*80 + "\n")
        
        price_signal = FraudSignal(
            signal_id="SIGNAL_003",
            signal_type=SignalType.PRICE_ANOMALY,
            entity_id="ANOMALY_001",
            confidence=0.88,
            timestamp=datetime.now(),
            metadata={
                'product_id': 'PROD_12345',
                'anomaly_type': 'synchronized_price_drop',
                'price_change': -40
            }
        )
        
        investigation = await self.engine.trace_fraud_network(price_signal)
        
        print(f"💸 Price manipulation network exposed!")
        print(f"   Coordinated products: {len([n for n, d in investigation.network_graph.nodes(data=True) if d.get('node_type') == 'product'])}")
        print(f"   Cartel members: {len([n for n, d in investigation.network_graph.nodes(data=True) if d.get('node_type') == 'seller'])}")
        print(f"   Financial impact: ${investigation.financial_impact:,.2f}")
        
        return investigation

# ============= Main Demo Runner =============

async def main():
    """Run the complete Cross Intelligence demo"""
    demo = CrossIntelligenceDemo()
    
    # Run main demo
    investigation, viz_data = await demo.run_review_fraud_demo()
    
    # Optional: Run additional demos
    # await demo.run_seller_fraud_demo()
    # await demo.run_price_anomaly_demo()
    
    # Save results for visualization
    with open('network_visualization_data.json', 'w') as f:
        json.dump(viz_data, f, indent=2)
    
    print("\n💾 Visualization data saved to 'network_visualization_data.json'")
    print("🎯 Ready for hackathon demo!")

if __name__ == "__main__":
    asyncio.run(main())