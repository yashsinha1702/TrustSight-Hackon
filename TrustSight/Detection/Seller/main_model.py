import torch
import torch.nn as nn

class SellerNetworkDetectionModel(nn.Module):
    """Main model class that combines all detection components for comprehensive seller network fraud detection"""
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.seller_feature_dim = 21
        
        from model_components import (
            SellerEmbedding, GraphNetworkEncoder, PriceCoordinationDetector,
            InventoryPatternAnalyzer, RegistrationClusterDetector, ExitScamPredictor
        )
        
        self.seller_embedding = SellerEmbedding(
            self.seller_feature_dim,
            config.embedding_dim
        )
        
        self.graph_encoder = GraphNetworkEncoder(
            self.seller_feature_dim,
            config.hidden_dim,
            config.embedding_dim,
            config.num_gcn_layers
        )
        
        self.price_coordinator = PriceCoordinationDetector(config.embedding_dim)
        self.inventory_analyzer = InventoryPatternAnalyzer(config.embedding_dim)
        self.registration_detector = RegistrationClusterDetector(config.embedding_dim)
        self.exit_scam_predictor = ExitScamPredictor(config.embedding_dim)
        
        self.network_detector = nn.Sequential(
            nn.Linear(config.embedding_dim * 2, config.hidden_dim),
            nn.ReLU(),
            nn.Dropout(config.dropout_rate),
            nn.Linear(config.hidden_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )
        
        self.signal_fusion = nn.Sequential(
            nn.Linear(5, 32),
            nn.ReLU(),
            nn.Dropout(config.dropout_rate),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1)
        )
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LSTM):
            for name, param in module.named_parameters():
                if 'weight' in name:
                    torch.nn.init.xavier_uniform_(param)
                elif 'bias' in name:
                    torch.nn.init.zeros_(param)
        
    def forward(self, batch):
        seller_embeddings = self.seller_embedding(batch['seller_features'])
        
        if 'graph' in batch and batch['graph'].edge_index.numel() > 0:
            graph_embeddings = self.graph_encoder(
                batch['graph'].x,
                batch['graph'].edge_index,
                batch.get('batch')
            )
        else:
            graph_embeddings = seller_embeddings
        
        combined_features = torch.cat([seller_embeddings, graph_embeddings], dim=-1)
        
        signals = {
            'network': self.network_detector(combined_features),
            'price_coordination': self.price_coordinator(
                seller_embeddings,
                batch['seller_features']
            ),
            'inventory_sharing': self.inventory_analyzer(seller_embeddings),
            'registration_cluster': self.registration_detector(seller_embeddings),
            'exit_scam': self.exit_scam_predictor(seller_embeddings)
        }
        
        all_signals = torch.cat(list(signals.values()), dim=-1)
        fraud_probability = self.signal_fusion(all_signals)
        
        outputs = {
            'fraud_probability': fraud_probability,
            'is_network_member_pred': signals['network'],
            'has_price_coordination_pred': signals['price_coordination'],
            'has_inventory_sharing_pred': signals['inventory_sharing'],
            'has_registration_cluster_pred': signals['registration_cluster'],
            'exit_scam_risk_pred': signals['exit_scam'],
            'signals': signals
        }
        
        return outputs