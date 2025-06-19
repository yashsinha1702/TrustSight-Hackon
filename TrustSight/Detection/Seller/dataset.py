import torch
import pandas as pd
import json
from datetime import datetime, timezone
from typing import List
from torch.utils.data import Dataset
from torch_geometric.data import Data

class SellerNetworkDataset(Dataset):
    """Dataset class for loading and preprocessing seller network data for fraud detection"""
    
    def __init__(self, data_path: str, config):
        self.config = config
        self.data = self.load_data(data_path)
        self.seller_graphs = self.build_seller_graphs()
        
    def load_data(self, data_path: str) -> pd.DataFrame:
        with open(data_path, 'r') as f:
            data = json.load(f)
        
        df = pd.DataFrame(data)
        df['registration_date'] = pd.to_datetime(df['registration_date'])
        df['days_active'] = (datetime.now(timezone.utc) - df['registration_date']).dt.days
        return df
    
    def build_seller_graphs(self) -> List[Data]:
        graphs = []
        
        for idx, seller in self.data.iterrows():
            node_features = self.extract_seller_features(seller)
            connected_sellers = seller.get('network_features', {}).get('shared_products_with_sellers', [])
            
            if connected_sellers:
                edge_index = self.build_edge_index(seller['seller_id'], connected_sellers)
                graph = Data(
                    x=node_features,
                    edge_index=edge_index,
                    seller_id=seller['seller_id']
                )
            else:
                graph = Data(
                    x=node_features,
                    edge_index=torch.tensor([[], []], dtype=torch.long),
                    seller_id=seller['seller_id']
                )
            
            graphs.append(graph)
        
        return graphs
    
    def extract_seller_features(self, seller) -> torch.Tensor:
        features = []
        
        metrics = seller.get('seller_metrics', {})
        features.extend([
            metrics.get('total_products', 0) / 1000,
            metrics.get('avg_product_price', 0) / 100,
            metrics.get('avg_rating', 0) / 5,
            metrics.get('response_time_hours', 0) / 24,
            metrics.get('fulfillment_rate', 0),
            metrics.get('return_rate', 0),
            metrics.get('customer_complaints', 0) / 100
        ])
        
        pricing = seller.get('pricing_behavior', {})
        features.extend([
            pricing.get('avg_price_change_frequency_days', 0) / 30,
            pricing.get('max_price_drop_percent', 0) / 100,
            pricing.get('synchronized_changes_count', 0) / 50,
            float(pricing.get('competitor_price_matching', False)),
            float(pricing.get('dynamic_pricing_detected', False))
        ])
        
        inventory = seller.get('inventory_patterns', {})
        features.extend([
            inventory.get('avg_stock_level', 0) / 1000,
            inventory.get('max_stock_spike', 0) / 10000,
            inventory.get('inventory_turnover_days', 0) / 30
        ])
        
        network = seller.get('network_features', {})
        features.extend([
            network.get('shared_product_count', 0) / 100,
            network.get('address_similarity_score', 0),
            network.get('customer_overlap_score', 0),
            network.get('reviewer_overlap_score', 0)
        ])
        
        features.extend([
            seller.get('account_age_days', 0) / 365,
            seller.get('temporal_features', {}).get('registration_hour', 0) / 24
        ])
        
        return torch.tensor(features, dtype=torch.float32)
    
    def build_edge_index(self, seller_id: str, connected_sellers: List[str]) -> torch.Tensor:
        edges = []
        seller_idx = 0
        
        for i, connected_id in enumerate(connected_sellers, 1):
            edges.append([seller_idx, i])
            edges.append([i, seller_idx])
        
        if edges:
            return torch.tensor(edges, dtype=torch.long).t()
        else:
            return torch.tensor([[], []], dtype=torch.long)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        seller = self.data.iloc[idx]
        
        features = {
            'seller_id': seller['seller_id'],
            'graph': self.seller_graphs[idx],
            'seller_features': self.extract_seller_features(seller),
            'registration_hour': seller['temporal_features']['registration_hour'],
            'days_active': seller['account_age_days'],
            'total_products': seller['seller_metrics']['total_products'],
            'labels': {
                'is_fraud': float(seller['labels']['is_fraud']),
                'is_network_member': float(seller['labels']['specific_patterns']['is_network_member']),
                'has_price_coordination': float(seller['labels']['specific_patterns']['has_price_coordination']),
                'has_inventory_sharing': float(seller['labels']['specific_patterns']['has_inventory_sharing']),
                'has_registration_cluster': float(seller['labels']['specific_patterns']['has_registration_cluster']),
                'exit_scam_risk': float(seller['labels']['specific_patterns']['exit_scam_risk'])
            }
        }
        
        return features