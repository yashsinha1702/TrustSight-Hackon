from dataclasses import dataclass
from typing import Dict

@dataclass
class ModelConfig:
    """Configuration parameters for TrustSight Review Fraud Detection model"""
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
    task_weights: Dict[str, float] = None
    
    def __post_init__(self):
        if self.task_weights is None:
            self.task_weights = {
                'overall_fraud': 2.0,
                'generic_text': 0.8,
                'timing_anomaly': 1.0,
                'bot_reviewer': 1.2,
                'incentivized': 0.7,
                'network_fraud': 1.5
            }