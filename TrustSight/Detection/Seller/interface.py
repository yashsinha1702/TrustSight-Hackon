import torch
import numpy as np
from typing import Dict, List

class SellerNetworkInterface:
    """Interface class for integrating seller network detection model with external systems"""
    
    def __init__(self, model_path: str, config, device='cuda'):
        self.device = device
        self.config = config
        
        from main_model import SellerNetworkDetectionModel
        
        self.model = SellerNetworkDetectionModel(config)
        checkpoint = torch.load(model_path, map_location=device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(device)
        self.model.eval()
        
    def analyze_seller(self, seller_data: Dict) -> Dict:
        features = self.prepare_seller_features(seller_data)
        
        with torch.no_grad():
            outputs = self.model(features)
        
        return {
            'seller_id': seller_data['seller_id'],
            'fraud_probability': outputs['fraud_probability'].item(),
            'network_member': outputs['is_network_member_pred'].item(),
            'price_coordination': outputs['has_price_coordination_pred'].item(),
            'inventory_sharing': outputs['has_inventory_sharing_pred'].item(),
            'exit_scam_risk': outputs['exit_scam_risk_pred'].item(),
            'network_signals': {
                'connected_sellers': seller_data.get('network_features', {}).get('shared_products_with_sellers', []),
                'network_strength': seller_data.get('network_features', {}).get('reviewer_overlap_score', 0)
            }
        }
    
    def analyze_network(self, sellers: List[Dict]) -> Dict:
        results = []
        
        for seller in sellers:
            result = self.analyze_seller(seller)
            results.append(result)
        
        network_stats = {
            'total_sellers': len(sellers),
            'fraud_sellers': sum(1 for r in results if r['fraud_probability'] > 0.5),
            'avg_fraud_score': np.mean([r['fraud_probability'] for r in results]),
            'price_coordination_detected': any(r['price_coordination'] > 0.5 for r in results),
            'exit_scam_risk': max(r['exit_scam_risk'] for r in results)
        }
        
        return {
            'individual_results': results,
            'network_statistics': network_stats,
            'action_required': network_stats['avg_fraud_score'] > 0.7
        }
    
    def prepare_seller_features(self, seller_data: Dict):
        pass