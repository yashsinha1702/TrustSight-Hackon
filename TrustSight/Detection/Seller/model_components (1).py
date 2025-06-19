import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool

class SellerEmbedding(nn.Module):
    """Neural network module for embedding seller features into dense representation"""
    
    def __init__(self, input_dim: int, embedding_dim: int):
        super().__init__()
        
        self.embedding = nn.Sequential(
            nn.Linear(input_dim, embedding_dim * 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(embedding_dim * 2, embedding_dim),
            nn.LayerNorm(embedding_dim)
        )
        
    def forward(self, x):
        return self.embedding(x)

class GraphNetworkEncoder(nn.Module):
    """Graph Neural Network encoder for processing seller network relationships"""
    
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, num_layers: int = 3):
        super().__init__()
        
        self.convs = nn.ModuleList()
        self.convs.append(GCNConv(input_dim, hidden_dim))
        
        for _ in range(num_layers - 2):
            self.convs.append(GCNConv(hidden_dim, hidden_dim))
        
        self.convs.append(GCNConv(hidden_dim, output_dim))
        self.dropout = nn.Dropout(0.1)
        
    def forward(self, x, edge_index, batch=None):
        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x, edge_index)
            x = F.relu(x)
            x = self.dropout(x)
        
        x = self.convs[-1](x, edge_index)
        
        if batch is not None:
            x = global_mean_pool(x, batch)
        
        return x

class PriceCoordinationDetector(nn.Module):
    """Neural network module for detecting price coordination patterns using temporal analysis"""
    
    def __init__(self, input_dim: int):
        super().__init__()
        
        self.temporal_encoder = nn.LSTM(
            input_dim + 21,
            hidden_size=128,
            num_layers=2,
            batch_first=True,
            bidirectional=True
        )
        
        self.pattern_detector = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
        
    def forward(self, price_features, temporal_features):
        combined = torch.cat([price_features, temporal_features], dim=-1)
        
        if len(combined.shape) == 2:
            combined = combined.unsqueeze(1)
        
        lstm_out, _ = self.temporal_encoder(combined)
        last_hidden = lstm_out[:, -1, :]
        
        return self.pattern_detector(last_hidden)

class InventoryPatternAnalyzer(nn.Module):
    """Neural network module for analyzing inventory patterns to detect sharing behaviors"""
    
    def __init__(self, input_dim: int):
        super().__init__()
        
        self.pattern_encoder = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 64),
            nn.ReLU()
        )
        
        self.sharing_detector = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )
        
    def forward(self, inventory_features):
        encoded = self.pattern_encoder(inventory_features)
        return self.sharing_detector(encoded)

class RegistrationClusterDetector(nn.Module):
    """Neural network module for detecting registration clusters in temporal data"""
    
    def __init__(self, input_dim: int):
        super().__init__()
        
        self.temporal_analyzer = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )
        
    def forward(self, registration_features):
        return self.temporal_analyzer(registration_features)

class ExitScamPredictor(nn.Module):
    """Neural network module for predicting exit scam risk based on seller behavior patterns"""
    
    def __init__(self, input_dim: int):
        super().__init__()
        
        self.risk_analyzer = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )
        
    def forward(self, seller_features):
        return self.risk_analyzer(seller_features)