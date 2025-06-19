import torch
import torch.nn as nn
from transformers import AutoModel
from config import ModelConfig
from model_components import (
    TemporalEncoder, ReviewAuthenticityHead, TimingAnomalyHead,
    ReviewerBehaviorHead, NetworkPatternHead, SignalFusionLayer
)

class TrustSightReviewFraudDetector(nn.Module):
    """Main unified model for multi-task review fraud detection using transformer and specialized detection heads"""
    
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        
        self.text_encoder = AutoModel.from_pretrained(config.model_name)
        
        self.temporal_encoder = TemporalEncoder(
            input_dim=3,
            hidden_dim=128
        )
        
        self.authenticity_head = ReviewAuthenticityHead(config.hidden_size)
        self.timing_head = TimingAnomalyHead(128, config.hidden_size)
        self.behavior_head = ReviewerBehaviorHead(config.hidden_size)
        self.network_head = NetworkPatternHead(config.hidden_size)
        
        self.signal_fusion = SignalFusionLayer(num_signals=4, signal_dim=1)
        
        self.fraud_classifier = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(config.dropout_rate),
            nn.Linear(64, 1)
        )
        
        self.task_classifiers = nn.ModuleDict({
            'generic_text': nn.Linear(128, 1),
            'timing_anomaly': nn.Linear(128, 1),
            'bot_reviewer': nn.Linear(128, 1),
            'incentivized': nn.Linear(128, 1),
            'network_fraud': nn.Linear(128, 1)
        })
        
        self.behavior_type_classifier = nn.Linear(128, 4)
        
    def forward(self, batch):
        text_outputs = self.text_encoder(
            input_ids=batch['input_ids'],
            attention_mask=batch['attention_mask']
        )
        text_features = text_outputs.last_hidden_state.mean(dim=1)
        
        temporal_input = torch.stack([
            batch['review_hour'].float(),
            batch['hours_since_delivery'].float(),
            batch['review_velocity'].float()
        ], dim=1).unsqueeze(1)
        
        temporal_features = self.temporal_encoder(temporal_input)
        
        signals = {}
        signals['authenticity'] = self.authenticity_head(text_features)
        signals['timing'] = self.timing_head(temporal_features, text_features)
        signals['behavior'] = self.behavior_head(text_features, None)
        signals['network'] = self.network_head(text_features)
        
        fused_features, attention_weights = self.signal_fusion(signals)
        
        outputs = {
            'fraud_probability': torch.sigmoid(self.fraud_classifier(fused_features)),
            'signals': signals,
            'attention_weights': attention_weights
        }
        
        for task, classifier in self.task_classifiers.items():
            outputs[f'{task}_pred'] = torch.sigmoid(classifier(fused_features))
        
        outputs['behavior_types'] = torch.softmax(
            self.behavior_type_classifier(fused_features), 
            dim=1
        )
        
        return outputs