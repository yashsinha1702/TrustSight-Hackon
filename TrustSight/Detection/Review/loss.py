import torch.nn as nn
from config import ModelConfig

class MultiTaskLoss(nn.Module):
    """Multi-task loss function for weighted combination of fraud detection subtasks"""
    
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.bce = nn.BCELoss()
        
    def forward(self, predictions, targets):
        losses = {}
        
        losses['overall_fraud'] = self.bce(
            predictions['fraud_probability'].squeeze(),
            targets['is_fraud']
        )
        
        task_mappings = {
            'generic_text': 'is_generic_text',
            'timing_anomaly': 'has_timing_anomaly', 
            'bot_reviewer': 'is_bot_reviewer',
            'incentivized': 'is_incentivized',
            'network_fraud': 'part_of_network'
        }
        
        for task, target_key in task_mappings.items():
            if f'{task}_pred' in predictions and target_key in targets:
                losses[task] = self.bce(
                    predictions[f'{task}_pred'].squeeze(),
                    targets[target_key]
                )
        
        total_loss = sum(
            self.config.task_weights.get(k, 1.0) * v 
            for k, v in losses.items()
        )
        
        return total_loss, losses