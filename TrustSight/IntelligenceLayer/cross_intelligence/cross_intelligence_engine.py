import asyncio
import logging
import networkx as nx
import numpy as np
import pandas as pd
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from uuid import uuid4

from .enums import SignalType, NetworkType
from .models import FraudSignal, NetworkNode, Investigation
from .network_pattern_classifier import NetworkPatternClassifier

logger = logging.getLogger(__name__)

class CrossIntelligenceEngine:
    """Main engine that traces fraud networks from any single signal, exposing entire fraud rings"""
    
    def __init__(self, model_configs: Dict[str, str] = None):
        logger.info("Initializing Cross Intelligence Engine...")
        
        self._load_detection_models(model_configs)
        
        self.master_graph = nx.DiGraph()
        
        self.pattern_classifier = NetworkPatternClassifier()
        
        self.expansion_strategies = {
            SignalType.FAKE_REVIEW: self._expand_from_review,
            SignalType.COUNTERFEIT_PRODUCT: self._expand_from_product,
            SignalType.SUSPICIOUS_SELLER: self._expand_from_seller,
            SignalType.LISTING_FRAUD: self._expand_from_listing,
            SignalType.PRICE_ANOMALY: self._expand_from_price_anomaly,
            SignalType.NETWORK_PATTERN: self._expand_from_network_pattern
        }
        
        self.active_investigations = {}
        self.completed_investigations = {}
        
        self.max_expansion_depth = 5
        self.max_nodes_per_investigation = 10000
        self.confidence_threshold = 0.3
        
        self.avg_product_value = 50.0
        self.avg_review_impact = 0.02
        
        self.executor = ThreadPoolExecutor(max_workers=10)
        
        logger.info("Cross Intelligence Engine initialized successfully!")
    
    def _load_detection_models(self, configs: Dict[str, str] = None):
        try:
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
        class MockModel:
            def predict(self, data):
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
        investigation = Investigation(
            investigation_id=str(uuid4()),
            initial_signal=initial_signal,
            start_time=datetime.now()
        )
        
        self.active_investigations[investigation.investigation_id] = investigation
        
        logger.info(f"Starting investigation {investigation.investigation_id} from {initial_signal.signal_type}")
        
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
        
        expansion_strategy = self.expansion_strategies.get(initial_signal.signal_type)
        if expansion_strategy:
            await expansion_strategy(investigation, initial_node)
        
        investigation.network_type = self.pattern_classifier.classify(investigation.network_graph)
        
        investigation.financial_impact = self._calculate_financial_impact(investigation)
        
        investigation.key_players = self._identify_kingpins(investigation)
        
        investigation.confidence_score = self._calculate_network_confidence(investigation)
        
        investigation.status = "completed"
        self.completed_investigations[investigation.investigation_id] = investigation
        del self.active_investigations[investigation.investigation_id]
        
        logger.info(f"Investigation {investigation.investigation_id} completed. "
                   f"Network size: {investigation.network_graph.number_of_nodes()} nodes, "
                   f"Financial impact: ${investigation.financial_impact:,.2f}")
        
        return investigation
    
    async def _expand_from_review(self, investigation: Investigation, review_node: NetworkNode):
        logger.info(f"Expanding from review: {review_node.node_id}")
        
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
        
        all_reviews = await self._get_reviews_by_reviewer(reviewer_id)
        logger.info(f"Reviewer {reviewer_id} has written {len(all_reviews)} reviews")
        
        products_to_investigate = []
        
        for review in all_reviews[:100]:
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
            
            product_id = review['product_id']
            if product_id not in investigation.nodes_investigated:
                products_to_investigate.append(product_id)
        
        product_tasks = [
            self._investigate_product(investigation, product_id, depth=3)
            for product_id in products_to_investigate[:50]
        ]
        
        sellers_found = await asyncio.gather(*product_tasks)
        unique_sellers = set([s for sublist in sellers_found for s in sublist if s])
        
        logger.info(f"Found {len(unique_sellers)} unique sellers from products")
        
        seller_tasks = [
            self._investigate_seller_network(investigation, seller_id, depth=4)
            for seller_id in unique_sellers
        ]
        
        await asyncio.gather(*seller_tasks)
        
        await self._find_connected_reviewers(investigation, products_to_investigate)
        
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
        logger.info(f"Expanding from seller: {seller_node.node_id}")
        
        products = await self._get_products_by_seller(seller_node.node_id)
        logger.info(f"Seller {seller_node.node_id} has {len(products)} products")
        
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
        
        for product in products[:100]:
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
            
            reviews = await self._get_reviews_for_product(product['product_id'])
            
            reviewer_overlap = await self._analyze_reviewer_overlap(
                investigation, product['product_id'], reviews
            )
            
            if reviewer_overlap['suspicious_score'] > 0.7:
                investigation.expansion_path.append({
                    'finding': 'suspicious_reviewer_pattern',
                    'product': product['product_id'],
                    'overlap_score': reviewer_overlap['suspicious_score']
                })
        
        price_coordination = await self._check_price_coordination(
            seller_node.node_id, connected_sellers
        )
        
        if price_coordination['coordinated']:
            investigation.expansion_path.append({
                'finding': 'price_coordination_detected',
                'sellers': price_coordination['coordinated_sellers'],
                'evidence': price_coordination['evidence']
            })
        
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
        logger.info(f"Expanding from product: {product_node.node_id}")
        
        seller_id = await self._get_seller_from_product(product_node.node_id)
        
        if seller_id:
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
        logger.info(f"Expanding from listing: {listing_node.node_id}")
        
        history = await self._get_listing_history(listing_node.node_id)
        
        historical_sellers = set()
        
        for snapshot in history:
            if snapshot['seller_id'] not in historical_sellers:
                historical_sellers.add(snapshot['seller_id'])
                
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
        
        for seller_id in historical_sellers:
            seller_node = NetworkNode(node_id=seller_id, node_type='seller')
            await self._expand_from_seller(investigation, seller_node)
    
    async def _expand_from_price_anomaly(self, investigation: Investigation, anomaly_node: NetworkNode):
        logger.info(f"Expanding from price anomaly: {anomaly_node.node_id}")
        
        product_id = anomaly_node.attributes.get('product_id')
        if not product_id:
            return
        
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
        
        investigation.expansion_path.append({
            'step': 'price_anomaly_expansion',
            'initial_anomaly': anomaly_node.node_id,
            'coordinated_products': len(coordinated_products) if 'coordinated_products' in locals() else 0,
            'cartel_members': len(cartel_sellers) if 'cartel_sellers' in locals() else 0,
            'timestamp': datetime.now().isoformat()
        })
    
    async def _expand_from_network_pattern(self, investigation: Investigation, pattern_node: NetworkNode):
        logger.info(f"Expanding from network pattern: {pattern_node.node_id}")
        
        pattern_entities = pattern_node.attributes.get('entities', [])
        pattern_type = pattern_node.attributes.get('pattern_type', 'unknown')
        
        logger.info(f"Network pattern type: {pattern_type}, entities: {len(pattern_entities)}")
        
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
        
        if pattern_type == 'review_burst':
            await self._expand_review_burst_pattern(investigation, pattern_entities)
        elif pattern_type == 'seller_network':
            await self._expand_seller_network_pattern(investigation, pattern_entities)
        elif pattern_type == 'product_clone':
            await self._expand_product_clone_pattern(investigation, pattern_entities)
        elif pattern_type == 'coordinated_attack':
            await self._expand_coordinated_attack_pattern(investigation, pattern_entities)
        
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
        
        investigation.expansion_path.append({
            'step': 'network_pattern_expansion',
            'pattern_type': pattern_type,
            'entities_processed': len(pattern_entities),
            'related_patterns': len(related_patterns),
            'timestamp': datetime.now().isoformat()
        })
    
    async def _expand_review_burst_pattern(self, investigation: Investigation, entities: List[Dict]):
        reviewers = [e for e in entities if e['type'] == 'reviewer']
        
        for reviewer in reviewers:
            other_reviews = await self._get_reviews_by_reviewer(reviewer['id'])
            
            timing_clusters = self._analyze_review_timing(other_reviews)
            
            if timing_clusters['suspicious_clusters'] > 2:
                investigation.network_graph.nodes[reviewer['id']]['timing_anomaly'] = True
                investigation.network_graph.nodes[reviewer['id']]['fraud_score'] = 0.9
    
    async def _expand_seller_network_pattern(self, investigation: Investigation, entities: List[Dict]):
        sellers = [e for e in entities if e['type'] == 'seller']
        
        for seller in sellers:
            reg_details = await self._get_seller_registration_details(seller['id'])
            
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
        products = [e for e in entities if e['type'] == 'product']
        
        for product in products:
            brand_check = await self._verify_brand_authenticity(product['id'])
            
            if brand_check['is_authentic']:
                investigation.network_graph.nodes[product['id']]['authentic_product'] = True
                
                for other_product in products:
                    if other_product['id'] != product['id']:
                        investigation.network_graph.nodes[other_product['id']]['counterfeit_probability'] = 0.85
    
    async def _expand_coordinated_attack_pattern(self, investigation: Investigation, entities: List[Dict]):
        negative_reviews = [e for e in entities if e['type'] == 'review' and e.get('rating', 5) <= 2]
        
        if negative_reviews:
            target_product = negative_reviews[0].get('product_id')
            
            competitors = await self._find_product_competitors(target_product)
            
            for competitor in competitors:
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
        reviewer_product_map = defaultdict(set)
        
        for product_id in product_ids:
            reviews = await self._get_reviews_for_product(product_id)
            for review in reviews:
                reviewer_product_map[review['reviewer_id']].add(product_id)
        
        suspicious_reviewers = {
            reviewer: products 
            for reviewer, products in reviewer_product_map.items()
            if len(products) >= 3
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
        total_impact = 0.0
        
        for node_id, attrs in investigation.network_graph.nodes(data=True):
            node_type = attrs.get('node_type', '')
            
            if node_type == 'product':
                price = attrs.get('price', self.avg_product_value)
                fake_reviews = len([
                    n for n in investigation.network_graph.predecessors(node_id)
                    if investigation.network_graph.nodes[n].get('node_type') == 'review'
                ])
                
                estimated_impact = price * fake_reviews * 100 * self.avg_review_impact
                total_impact += estimated_impact
                
            elif node_type == 'seller':
                products = [
                    n for n in investigation.network_graph.successors(node_id)
                    if investigation.network_graph.nodes[n].get('node_type') == 'product'
                ]
                seller_impact = len(products) * self.avg_product_value * 1000
                total_impact += seller_impact
        
        network_size = investigation.network_graph.number_of_nodes()
        if network_size > 100:
            total_impact *= 1.5
        
        return total_impact
    
    def _identify_kingpins(self, investigation: Investigation) -> List[str]:
        kingpins = []
        
        try:
            degree_centrality = nx.degree_centrality(investigation.network_graph)
            betweenness_centrality = nx.betweenness_centrality(investigation.network_graph)
            pagerank = nx.pagerank(investigation.network_graph)
            
            combined_scores = {}
            for node in investigation.network_graph.nodes():
                combined_scores[node] = (
                    degree_centrality.get(node, 0) * 0.3 +
                    betweenness_centrality.get(node, 0) * 0.4 +
                    pagerank.get(node, 0) * 0.3
                )
            
            sorted_nodes = sorted(combined_scores.items(), key=lambda x: x[1], reverse=True)
            
            for node_id, score in sorted_nodes[:10]:
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
        factors = []
        
        network_size = investigation.network_graph.number_of_nodes()
        if network_size > 50:
            factors.append(0.9)
        elif network_size > 20:
            factors.append(0.7)
        else:
            factors.append(0.5)
        
        if network_size > 1:
            density = nx.density(investigation.network_graph)
            factors.append(min(density * 2, 1.0))
        
        factors.append(investigation.initial_signal.confidence)
        
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
        
        return sum(factors) / len(factors) if factors else 0.5
    
    def create_visualization_data(self, investigation: Investigation) -> Dict:
        nodes = []
        edges = []
        
        for node_id, attrs in investigation.network_graph.nodes(data=True):
            nodes.append({
                'id': node_id,
                'label': self._create_node_label(node_id, attrs),
                'type': attrs.get('node_type', 'unknown'),
                'size': self._calculate_node_size(investigation.network_graph, node_id),
                'color': self._get_node_color(attrs),
                'fraud_score': attrs.get('fraud_score', 0),
                'x': np.random.uniform(-100, 100),
                'y': np.random.uniform(-100, 100)
            })
        
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
        degree = graph.degree(node_id)
        return min(10 + (degree * 2), 50)
    
    def _get_node_color(self, attrs: Dict) -> str:
        colors = {
            'seller': '#FF6B6B',
            'product': '#4ECDC4',
            'reviewer': '#FFE66D',
            'review': '#95E1D3'
        }
        base_color = colors.get(attrs.get('node_type', ''), '#C7C7C7')
        
        fraud_score = attrs.get('fraud_score', 0)
        if fraud_score > 0.7:
            return '#8B0000'
        
        return base_color
    
    def _get_node_type(self, signal_type: SignalType) -> str:
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
        recommendations = {
            'immediate_actions': [],
            'investigation_priorities': [],
            'bulk_actions': [],
            'monitoring_targets': []
        }
        
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
            
        edge_nodes = [n for n, d in investigation.network_graph.nodes(data=True)
                     if investigation.network_graph.degree(n) == 1]
        recommendations['monitoring_targets'] = edge_nodes[:20]
        
        return recommendations
    
    async def _get_reviewer_from_review(self, review_id: str) -> Optional[str]:
        return f"REVIEWER_{review_id[-6:]}"
    
    async def _get_reviews_by_reviewer(self, reviewer_id: str) -> List[Dict]:
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
        if product_id not in investigation.network_graph:
            investigation.network_graph.add_node(
                product_id,
                node_type='product',
                investigation_depth=depth,
                title=f"Product {product_id[-4:]}",
                price=np.random.uniform(10, 500)
            )
        
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
        if np.random.random() > 0.5:
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
        num_products = np.random.randint(10, 100)
        return [{
            'product_id': f"PROD_{seller_id[-4:]}_{i:04d}",
            'title': f"Product Title {i}",
            'price': np.random.uniform(10, 500),
            'category': np.random.choice(['Electronics', 'Clothing', 'Home'])
        } for i in range(num_products)]
    
    async def _find_connected_sellers(self, seller_id: str) -> List[Dict]:
        connections = []
        
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
        num_reviews = np.random.randint(5, 50)
        return [{
            'review_id': f"REV_{product_id[-4:]}_{i:04d}",
            'reviewer_id': f"REVIEWER_{np.random.randint(1000, 9999)}",
            'rating': 5 if np.random.random() > 0.3 else np.random.randint(1, 5),
            'timestamp': datetime.now() - timedelta(days=np.random.randint(1, 180))
        } for i in range(num_reviews)]
    
    async def _analyze_reviewer_overlap(self, investigation: Investigation,
                                      product_id: str, reviews: List[Dict]) -> Dict:
        suspicious_score = np.random.uniform(0.3, 0.95)
        return {
            'suspicious_score': suspicious_score,
            'overlap_count': int(len(reviews) * 0.3),
            'pattern': 'coordinated_reviewing' if suspicious_score > 0.7 else 'normal'
        }
    
    async def _check_price_coordination(self, seller_id: str, 
                                      connected_sellers: List[Dict]) -> Dict:
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
        return {
            'sharing_detected': np.random.random() > 0.7,
            'shared_products': np.random.randint(10, 50),
            'pattern': 'dropshipping' if np.random.random() > 0.5 else 'warehouse_sharing'
        }
    
    async def _get_seller_from_product(self, product_id: str) -> Optional[str]:
        return f"SELLER_{product_id[-6:]}"
    
    async def _find_similar_products(self, product_id: str) -> List[Dict]:
        num_similar = np.random.randint(5, 20)
        return [{
            'product_id': f"PROD_SIM_{product_id[-4:]}_{i}",
            'similarity_score': np.random.uniform(0.7, 0.95),
            'similarity_type': np.random.choice(['title', 'image', 'description'])
        } for i in range(num_similar)]
    
    async def _get_listing_history(self, listing_id: str) -> List[Dict]:
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
        cartel = []
        
        if len(coordinated_products) > 5:
            for i in range(np.random.randint(2, 8)):
                cartel.append({
                    'seller_id': f"SELLER_CARTEL_{i:03d}",
                    'confidence': np.random.uniform(0.7, 0.95)
                })
        
        return cartel
    
    async def _find_related_patterns(self, pattern_id: str) -> List[Dict]:
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
        return {
            'suspicious_clusters': np.random.randint(0, 5),
            'burst_detected': np.random.random() > 0.6,
            'average_interval_hours': np.random.uniform(0.5, 48)
        }
    
    async def _get_seller_registration_details(self, seller_id: str) -> Dict:
        return {
            'business_name': f"Business_{seller_id[-4:]}",
            'address': f"123 Main St, City {np.random.randint(1, 100)}",
            'phone': f"555-{np.random.randint(1000, 9999)}",
            'email_domain': f"domain{np.random.randint(1, 50)}.com",
            'registration_date': datetime.now() - timedelta(days=np.random.randint(30, 365))
        }
    
    async def _find_sellers_by_registration(self, reg_details: Dict) -> List[Dict]:
        matches = []
        
        if np.random.random() > 0.5:
            for i in range(np.random.randint(1, 4)):
                matches.append({
                    'seller_id': f"SELLER_MATCH_{i:03d}",
                    'match_type': np.random.choice(['address', 'phone', 'email_domain']),
                    'confidence': np.random.uniform(0.7, 0.95)
                })
        
        return matches
    
    async def _verify_brand_authenticity(self, product_id: str) -> Dict:
        return {
            'is_authentic': np.random.random() > 0.7,
            'brand': f"Brand_{np.random.randint(1, 20)}",
            'confidence': np.random.uniform(0.6, 0.95)
        }
    
    async def _find_product_competitors(self, product_id: str) -> List[Dict]:
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
        return {
            'connection_strength': np.random.uniform(0.3, 0.9),
            'link_type': np.random.choice(['direct', 'indirect', 'suspicious']),
            'evidence_count': np.random.randint(0, 10)
        }