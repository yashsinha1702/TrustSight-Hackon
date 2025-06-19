import networkx as nx
import numpy as np
from .enums import NetworkType

class NetworkPatternClassifier:
    """Classifies fraud networks into types using graph features and patterns"""
    
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
        scores = {}
        
        for network_type, classifier_func in self.patterns.items():
            scores[network_type] = classifier_func(graph)
        
        return max(scores, key=scores.get)
    
    def _is_review_farm(self, graph: nx.DiGraph) -> float:
        score = 0.0
        
        reviewers = [n for n, d in graph.nodes(data=True) if d.get('node_type') == 'reviewer']
        reviews = [n for n, d in graph.nodes(data=True) if d.get('node_type') == 'review']
        
        if len(reviewers) > 0:
            ratio = len(reviews) / len(reviewers)
            if ratio > 20:
                score += 0.4
            
            products_per_reviewer = []
            for reviewer in reviewers[:10]:
                products = [n for n in graph.neighbors(reviewer) 
                          if graph.nodes[n].get('node_type') == 'product']
                products_per_reviewer.append(len(products))
            
            if products_per_reviewer and np.mean(products_per_reviewer) > 10:
                score += 0.4
            
            score += 0.2
        
        return score
    
    def _is_seller_cartel(self, graph: nx.DiGraph) -> float:
        score = 0.0
        
        sellers = [n for n, d in graph.nodes(data=True) if d.get('node_type') == 'seller']
        
        if len(sellers) > 3:
            seller_subgraph = graph.subgraph(sellers)
            if seller_subgraph.number_of_edges() > 0:
                density = nx.density(seller_subgraph)
                score += density * 0.5
            
            shared_reviewers = set()
            for seller in sellers:
                reviewers = [n for n in graph.neighbors(seller)
                           if graph.nodes[n].get('node_type') == 'reviewer']
                shared_reviewers.update(reviewers)
            
            if len(shared_reviewers) > len(sellers) * 2:
                score += 0.3
            
            price_coord_edges = [e for e in graph.edges(data=True)
                               if e[2].get('edge_type') == 'price_coordination']
            if price_coord_edges:
                score += 0.2
        
        return score
    
    def _is_counterfeit_ring(self, graph: nx.DiGraph) -> float:
        score = 0.0
        
        if nx.is_directed_acyclic_graph(graph):
            score += 0.3
        
        similar_product_edges = [e for e in graph.edges(data=True)
                               if e[2].get('edge_type') == 'similar_product']
        if len(similar_product_edges) > 5:
            score += 0.4
        
        products = [n for n, d in graph.nodes(data=True) if d.get('node_type') == 'product']
        if len(products) > 20:
            score += 0.3
        
        return score
    
    def _is_hybrid_operation(self, graph: nx.DiGraph) -> float:
        review_score = self._is_review_farm(graph)
        seller_score = self._is_seller_cartel(graph)
        counterfeit_score = self._is_counterfeit_ring(graph)
        
        high_scores = sum(1 for s in [review_score, seller_score, counterfeit_score] if s > 0.5)
        
        if high_scores >= 2:
            return 0.8
        
        return 0.2
    
    def _is_competitor_attack(self, graph: nx.DiGraph) -> float:
        score = 0.0
        
        negative_reviews = [n for n, d in graph.nodes(data=True) 
                          if d.get('node_type') == 'review' and d.get('rating', 5) <= 2]
        
        if len(negative_reviews) > 10:
            score += 0.3
            
            targeted_products = set()
            for review in negative_reviews:
                products = [n for n in graph.neighbors(review)
                          if graph.nodes[n].get('node_type') == 'product']
                targeted_products.update(products)
            
            if len(targeted_products) < 5:
                score += 0.4
            
            if any(d.get('attack_beneficiary') for n, d in graph.nodes(data=True)):
                score += 0.3
        
        return score
    
    def _is_exit_scam(self, graph: nx.DiGraph) -> float:
        score = 0.0
        
        sellers = [n for n, d in graph.nodes(data=True) if d.get('node_type') == 'seller']
        
        for seller in sellers:
            products = [n for n in graph.neighbors(seller)
                      if graph.nodes[n].get('node_type') == 'product']
            
            if len(products) > 50:
                score += 0.3
                
                price_anomalies = [n for n in graph.neighbors(seller)
                                 if graph.nodes[n].get('node_type') == 'anomaly']
                
                if price_anomalies:
                    score += 0.4
                
                if graph.nodes[seller].get('new_seller'):
                    score += 0.3
        
        return score