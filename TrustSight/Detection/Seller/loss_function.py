import torch
import torch.nn as nn

class SellerNetworkLoss(nn.Module):
    """Multi-task loss function for seller network detection with weighted task-specific losses"""
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.pos_weight = 1.5
        
    def forward(self, predictions, targets):
        losses = {}
        
        device = predictions['fraud_probability'].device
        pos_weight = torch.tensor([self.pos_weight]).to(device)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        
        losses['network_detection'] = criterion(
            predictions['fraud_probability'].squeeze(),
            targets['is_fraud']
        )
        
        task_mappings = {
            'price_coordination': ('has_price_coordination_pred', 'has_price_coordination'),
            'inventory_sharing': ('has_inventory_sharing_pred', 'has_inventory_sharing'),
            'registration_cluster': ('has_registration_cluster_pred', 'has_registration_cluster'),
            'exit_scam': ('exit_scam_risk_pred', 'exit_scam_risk')
        }
        
        for task, (pred_key, target_key) in task_mappings.items():
            if pred_key in predictions and target_key in targets:
                losses[task] = criterion(
                    predictions[pred_key].squeeze(),
                    targets[target_key]
                )
        
        total_loss = sum(
            self.config.task_weights.get(k, 1.0) * v 
            for k, v in losses.items()
        )
        
        return total_loss, losses