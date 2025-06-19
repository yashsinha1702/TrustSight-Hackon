import torch
import torch.nn as nn
from typing import Dict

class TemporalEncoder(nn.Module):
    """Neural network module for encoding temporal patterns in review timing"""
    
    def __init__(self, input_dim: int = 6, hidden_dim: int = 128):
        super().__init__()
        self.lstm = nn.LSTM(
            input_dim, 
            hidden_dim, 
            num_layers=2, 
            batch_first=True,
            bidirectional=True
        )
        self.projection = nn.Linear(hidden_dim * 2, hidden_dim)
        
    def forward(self, temporal_features):
        lstm_out, _ = self.lstm(temporal_features)
        last_output = lstm_out[:, -1, :]
        return self.projection(last_output)

class ReviewAuthenticityHead(nn.Module):
    """Detection head for identifying generic or template-based review text patterns"""
    
    def __init__(self, hidden_dim: int = 768):
        super().__init__()
        
        self.generic_detector = nn.Sequential(
            nn.Linear(hidden_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 128)
        )
        
        self.specificity_scorer = nn.Sequential(
            nn.Linear(hidden_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 128)
        )
        
        self.emotion_consistency = nn.Sequential(
            nn.Linear(hidden_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 128)
        )
        
        self.attention = nn.MultiheadAttention(
            embed_dim=128,
            num_heads=4,
            dropout=0.1
        )
        
        self.output = nn.Sequential(
            nn.Linear(128 * 3, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 1)
        )
        
    def forward(self, text_features):
        generic = self.generic_detector(text_features)
        specificity = self.specificity_scorer(text_features)
        emotion = self.emotion_consistency(text_features)
        
        signals = torch.stack([generic, specificity, emotion], dim=1)
        attended, _ = self.attention(signals, signals, signals)
        
        combined = torch.cat([generic, specificity, emotion], dim=1)
        return self.output(combined)

class TimingAnomalyHead(nn.Module):
    """Detection head for identifying timing-based fraud patterns and bot behavior"""
    
    def __init__(self, temporal_dim: int = 128, text_dim: int = 768):
        super().__init__()
        
        self.temporal_processor = nn.Sequential(
            nn.Linear(temporal_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 128)
        )
        
        self.bot_pattern_detector = nn.Sequential(
            nn.Linear(temporal_dim + text_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 128)
        )
        
        self.velocity_analyzer = nn.Sequential(
            nn.Linear(temporal_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64)
        )
        
        self.output = nn.Sequential(
            nn.Linear(128 + 128 + 64, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 1)
        )
        
    def forward(self, temporal_features, text_features):
        temporal = self.temporal_processor(temporal_features)
        combined = torch.cat([temporal_features, text_features], dim=1)
        bot_patterns = self.bot_pattern_detector(combined)
        velocity = self.velocity_analyzer(temporal_features)
        all_features = torch.cat([temporal, bot_patterns, velocity], dim=1)
        return self.output(all_features)

class ReviewerBehaviorHead(nn.Module):
    """Detection head for classifying reviewer behavior patterns and identifying fake reviewers"""
    
    def __init__(self, hidden_dim: int = 768):
        super().__init__()
        
        self.behavior_encoder = nn.LSTM(
            hidden_dim,
            256,
            num_layers=2,
            batch_first=True,
            bidirectional=True
        )
        
        self.pattern_classifier = nn.Sequential(
            nn.Linear(512,  256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 4)
        )
        
        self.fraud_score = nn.Linear(4, 1)
        
    def forward(self, text_features, reviewer_features):
        if len(text_features.shape) == 2:
            text_features = text_features.unsqueeze(1)
        
        encoded, _ = self.behavior_encoder(text_features)
        last_hidden = encoded[:, -1, :]
        behavior_logits = self.pattern_classifier(last_hidden)
        fraud_score = self.fraud_score(behavior_logits)
        
        return fraud_score

class NetworkPatternHead(nn.Module):
    """Detection head for identifying coordinated fraud networks and group behavior"""
    
    def __init__(self, hidden_dim: int = 768):
        super().__init__()
        
        self.network_encoder = nn.Sequential(
            nn.Linear(hidden_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, 256)
        )
        
        self.graph_attention = nn.MultiheadAttention(
            embed_dim=256,
            num_heads=8,
            dropout=0.1
        )
        
        self.output = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )
        
    def forward(self, text_features, network_features=None):
        encoded = self.network_encoder(text_features)
        attended, _ = self.graph_attention(
            encoded.unsqueeze(1),
            encoded.unsqueeze(1),
            encoded.unsqueeze(1)
        )
        return self.output(attended.squeeze(1))

class SignalFusionLayer(nn.Module):
    """Neural fusion layer for combining signals from multiple fraud detection heads"""
    
    def __init__(self, num_signals: int = 4, signal_dim: int = 1):
        super().__init__()
        
        self.fusion_network = nn.Sequential(
            nn.Linear(num_signals, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 128)
        )
        
        self.signal_attention = nn.Linear(num_signals, num_signals)
        
    def forward(self, signals: Dict[str, torch.Tensor]):
        signal_list = [s.squeeze(-1) if s.shape[-1] == 1 else s for s in signals.values()]
        concatenated = torch.cat([s.unsqueeze(-1) for s in signal_list], dim=1)
        attention_weights = torch.softmax(self.signal_attention(concatenated), dim=1)
        weighted_signals = concatenated * attention_weights
        fused = self.fusion_network(weighted_signals)
        
        return fused, attention_weights