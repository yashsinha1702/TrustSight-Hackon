from dataclasses import dataclass
from typing import Dict

@dataclass
class SellerNetworkConfig:
    """Configuration class for Seller Network Detection Model parameters and hyperparameters"""
    
    embedding_dim: int = 128
    hidden_dim: int = 256
    num_heads: int = 8
    num_gcn_layers: int = 3
    dropout_rate: float = 0.1
    learning_rate: float = 1e-5
    batch_size: int = 32
    num_epochs: int = 50
    warmup_steps: int = 500
    task_weights: Dict[str, float] = None
    
    def __post_init__(self):
        if self.task_weights is None:
            self.task_weights = {
                'network_detection': 2.0,
                'price_coordination': 1.5,
                'inventory_sharing': 1.2,
                'registration_cluster': 1.0,
                'exit_scam': 1.8
            }