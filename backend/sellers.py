import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from transformers import AutoModel, AutoTokenizer
from torch_geometric.nn import GCNConv, global_mean_pool, GAT
from torch_geometric.data import Data, Batch
import json
from datetime import datetime
from typing import Dict, List, Tuple, Optional
import logging
from dataclasses import dataclass
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score
from tqdm import tqdm
import networkx as nx
import warnings
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

# Or suppress all warnings from transformers
import transformers
transformers.logging.set_verbosity_error()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ============= Configuration =============
@dataclass
class SellerNetworkConfig:
    # Model architecture
    embedding_dim: int = 256
    hidden_dim: int = 512
    num_heads: int = 8
    num_gcn_layers: int = 3
    dropout_rate: float = 0.3  # Increased
    
    # Training - UPDATED
    learning_rate: float = 1e-5
    batch_size: int = 64  # Larger batches
    num_epochs: int = 50  # More epochs
    warmup_steps: int = 200
    
    # Scheduler
    use_scheduler: bool = True
    scheduler_patience: int = 5
    
    # Task weights - UPDATED
    task_weights: Dict[str, float] = None
    
    def __post_init__(self):
        if self.task_weights is None:
            self.task_weights = {
                'network_detection': 3.0,  # Increased
                'price_coordination': 1.5,
                'inventory_sharing': 1.2,
                'registration_cluster': 1.0,
                'exit_scam': 2.0  # Increased
            }

# ============= Dataset =============
class SellerNetworkDataset(Dataset):
    """Dataset for Seller Network Detection"""
    
    def __init__(self, data_path: str, config: SellerNetworkConfig):
        self.config = config
        self.data = self.load_data(data_path)
        self.seller_graphs = self.build_seller_graphs()
        
    def load_data(self, data_path: str) -> pd.DataFrame:
        """Load and preprocess seller data"""
        with open(data_path, 'r') as f:
            data = json.load(f)
        
        df = pd.DataFrame(data)
        
        # Convert timestamps
        df['registration_date'] = pd.to_datetime(df['registration_date'])
        
        # Extract additional features
        from datetime import datetime, timezone

        df['days_active'] = (datetime.now(timezone.utc) - df['registration_date']).dt.days

        
        return df
    
    def build_seller_graphs(self) -> List[Data]:
        """Build graph representation for each seller and their network"""
        graphs = []
        
        for idx, seller in self.data.iterrows():
            # Create node features
            node_features = self.extract_seller_features(seller)
            
            # Get connected sellers
            connected_sellers = seller.get('network_features', {}).get('shared_products_with_sellers', [])
            
            if connected_sellers:
                # Build edge index for this seller's network
                edge_index = self.build_edge_index(seller['seller_id'], connected_sellers)
                
                # Create PyG Data object
                graph = Data(
                    x=node_features,
                    edge_index=edge_index,
                    seller_id=seller['seller_id']
                )
            else:
                # Single node graph
                graph = Data(
                    x=node_features,
                    edge_index=torch.tensor([[], []], dtype=torch.long),
                    seller_id=seller['seller_id']
                )
            
            graphs.append(graph)
        
        return graphs
    
    def extract_seller_features(self, seller) -> torch.Tensor:
        """Extract and NORMALIZE features"""
        features = []
        
        # Seller metrics (normalized differently)
        metrics = seller.get('seller_metrics', {})
        features.extend([
            np.log1p(metrics.get('total_products', 0)) / 10,  # Log scale
            np.log1p(metrics.get('avg_product_price', 0)) / 10,
            metrics.get('avg_rating', 0) / 5,
            np.log1p(metrics.get('response_time_hours', 0)) / 5,
            metrics.get('fulfillment_rate', 0),
            metrics.get('return_rate', 0),
            np.log1p(metrics.get('customer_complaints', 0)) / 10
        ])
    
    
        
        # Pricing behavior
        pricing = seller.get('pricing_behavior', {})
        features.extend([
            pricing.get('avg_price_change_frequency_days', 0) / 30,
            pricing.get('max_price_drop_percent', 0) / 100,
            pricing.get('synchronized_changes_count', 0) / 50,
            float(pricing.get('competitor_price_matching', False)),
            float(pricing.get('dynamic_pricing_detected', False))
        ])
        
        # Inventory patterns
        inventory = seller.get('inventory_patterns', {})
        features.extend([
            inventory.get('avg_stock_level', 0) / 1000,
            inventory.get('max_stock_spike', 0) / 10000,
            inventory.get('inventory_turnover_days', 0) / 30
        ])
        
        # Network features
        network = seller.get('network_features', {})
        features.extend([
            network.get('shared_product_count', 0) / 100,
            network.get('address_similarity_score', 0),
            network.get('customer_overlap_score', 0),
            network.get('reviewer_overlap_score', 0)
        ])
        
        # Temporal features
        features.extend([
            seller.get('account_age_days', 0) / 365,
            seller.get('temporal_features', {}).get('registration_hour', 0) / 24
        ])
        
        return torch.tensor(features, dtype=torch.float32)
    
    def build_edge_index(self, seller_id: str, connected_sellers: List[str]) -> torch.Tensor:
        """Build edge index for graph"""
        edges = []
        
        # Create bidirectional edges
        seller_idx = 0  # Current seller is always node 0
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
            
            # Seller features for non-graph models
            'seller_features': self.extract_seller_features(seller),
            
            # Additional context
            'registration_hour': seller['temporal_features']['registration_hour'],
            'days_active': seller['account_age_days'],
            'total_products': seller['seller_metrics']['total_products'],
            
            # Labels
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

# ============= Model Components =============

class SellerEmbedding(nn.Module):
    """Embed seller features into dense representation"""
    
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
    """Encode seller network using Graph Neural Networks"""
    
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
        
        # Global pooling if batch is provided
        if batch is not None:
            x = global_mean_pool(x, batch)
        
        return x

class PriceCoordinationDetector(nn.Module):
    """Detect price coordination patterns"""
    
    def __init__(self, input_dim: int):
        super().__init__()
        
        # FIX: The LSTM should expect concatenated features
        # input_dim (128) + seller features (21) = 149
        self.temporal_encoder = nn.LSTM(
            input_dim + 21,  # Changed from input_dim to input_dim + 21
            hidden_size=128,
            num_layers=2,
            batch_first=True,
            bidirectional=True
        )
        
        self.pattern_detector = nn.Sequential(
            nn.Linear(256, 128),  # bidirectional
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
        
    def forward(self, price_features, temporal_features):
        # Combine features
        combined = torch.cat([price_features, temporal_features], dim=-1)
        
        if len(combined.shape) == 2:
            combined = combined.unsqueeze(1)
        
        # Encode temporal patterns
        lstm_out, _ = self.temporal_encoder(combined)
        last_hidden = lstm_out[:, -1, :]
        
        # Detect coordination
        return self.pattern_detector(last_hidden)

class InventoryPatternAnalyzer(nn.Module):
    """Analyze inventory patterns for sharing detection"""
    
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
    """Detect registration clusters"""
    
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
    """Predict exit scam risk"""
    
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

# ============= Main Model =============
import json
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
from datetime import datetime
import torch.nn.functional as F
from torch.nn import BatchNorm1d

class SellerNetworkDetectionModel(nn.Module):
    """Enhanced Seller Network Detection Model with more parameters and better architecture"""
    
    def __init__(self, config: SellerNetworkConfig):
        super().__init__()
        self.config = config
        self.seller_feature_dim = 21
        
        # Enhanced dimensions for more parameters
        self.base_hidden = 512
        self.mid_hidden = 256
        self.final_hidden = 128
        
        # ENHANCED Seller Embedding (21 -> 512 -> 256 -> 128)
        self.seller_embedding = nn.Sequential(
            nn.Linear(self.seller_feature_dim, self.base_hidden),
            nn.BatchNorm1d(self.base_hidden),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(self.base_hidden, self.mid_hidden),
            nn.BatchNorm1d(self.mid_hidden),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(self.mid_hidden, self.final_hidden),
            nn.BatchNorm1d(self.final_hidden),
            nn.ReLU()
        )
        
        # ENHANCED Graph Network Encoder
        self.graph_encoder = EnhancedGraphEncoder(
            self.seller_feature_dim,
            self.base_hidden,
            self.final_hidden,
            num_layers=4  # More layers
        )
        
        # DEEPER Detection Heads
        self.price_coordinator = self._build_detection_head(
            self.final_hidden, 
            [256, 128, 64, 32], 
            "price"
        )
        
        self.inventory_analyzer = self._build_detection_head(
            self.final_hidden,
            [256, 128, 64, 32],
            "inventory"
        )
        
        self.registration_detector = self._build_detection_head(
            self.final_hidden,
            [256, 128, 64, 32],
            "registration"
        )
        
        self.exit_scam_predictor = self._build_detection_head(
            self.final_hidden,
            [256, 128, 64, 32],
            "exit"
        )
        
        # BIGGER Network Detection Head (takes concatenated features)
        self.network_detector = nn.Sequential(
            nn.Linear(self.final_hidden * 2, self.base_hidden),  # 256 input (128*2)
            nn.BatchNorm1d(self.base_hidden),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(self.base_hidden, self.mid_hidden),
            nn.BatchNorm1d(self.mid_hidden),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(self.mid_hidden, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
        
        # Multi-Head Attention for Signal Importance
        self.signal_attention = nn.MultiheadAttention(
            embed_dim=64,
            num_heads=8,
            dropout=0.1,
            batch_first=True
        )
        
        # Transform signals to attention dimension
        self.signal_projector = nn.Linear(5, 64)
        
        # ENHANCED Signal Fusion Network
        self.signal_fusion = nn.Sequential(
            nn.Linear(64, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )
        
        # Residual connection for final prediction
        self.residual_weight = nn.Parameter(torch.tensor(0.1))
        
        # Initialize weights
        self.apply(self._init_weights)
        # In SellerNetworkDetectionModel.__init__, at the end:
        # Initialize final layers to break symmetry
        with torch.no_grad():
            # Initialize fraud detection head to slightly negative (more negatives in data)
            if hasattr(self.signal_fusion[-1], 'bias'):
                self.signal_fusion[-1].bias.data.fill_(-1.0)  # Start predicting more negatives
            
            # Initialize other heads similarly
            for head in [self.price_coordinator, self.inventory_analyzer, 
                        self.registration_detector, self.exit_scam_predictor]:
                if hasattr(head[-1], 'bias'):
                    head[-1].bias.data.fill_(-0.5)
        
    def _build_detection_head(self, input_dim, hidden_dims, head_type):
        """Build a detection head with specified architecture"""
        layers = []
        prev_dim = input_dim
        
        for i, hidden_dim in enumerate(hidden_dims):
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.ReLU())
            
            # Different dropout rates for different heads
            if head_type == "price":
                dropout_rate = 0.3 if i < 2 else 0.2
            elif head_type == "exit":
                dropout_rate = 0.4 if i < 2 else 0.3  # Higher dropout for exit scam
            else:
                dropout_rate = 0.2
                
            layers.append(nn.Dropout(dropout_rate))
            prev_dim = hidden_dim
            
        # Final layer
        layers.append(nn.Linear(prev_dim, 1))
        
        return nn.Sequential(*layers)
    
    def _init_weights(self, module):
        """Initialize weights with better strategies"""
        if isinstance(module, nn.Linear):
            # Xavier initialization for linear layers
            torch.nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                # Initialize bias for output layers differently
                if module.out_features == 1:
                    # Initialize to predict slightly positive (helps with all-negative predictions)
                    torch.nn.init.constant_(module.bias, 0.0)
                else:
                    torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.BatchNorm1d):
            torch.nn.init.ones_(module.weight)
            torch.nn.init.zeros_(module.bias)
    
    def forward(self, batch):
        # Extract seller embeddings with residual
        raw_features = batch['seller_features']
        seller_embeddings = self.seller_embedding(raw_features)
        
        # Graph encoding if available
        if 'graph' in batch and batch['graph'].edge_index.numel() > 0:
            graph_embeddings = self.graph_encoder(
                batch['graph'].x,
                batch['graph'].edge_index,
                batch.get('batch')
            )
        else:
            # If no graph, use seller embeddings
            graph_embeddings = seller_embeddings
        
        # Combine embeddings for network detection
        combined_features = torch.cat([seller_embeddings, graph_embeddings], dim=-1)
        
        # All detection heads
        signals = {
            'network': self.network_detector(combined_features),
            'price_coordination': self.price_coordinator(seller_embeddings),
            'inventory_sharing': self.inventory_analyzer(seller_embeddings),
            'registration_cluster': self.registration_detector(seller_embeddings),
            'exit_scam': self.exit_scam_predictor(seller_embeddings)
        }
        
        # Stack signals for attention
        all_signals = torch.stack(list(signals.values()), dim=1)  # [batch, 5, 1]
        all_signals = all_signals.squeeze(-1)  # [batch, 5]
        
        # Project signals to attention dimension
        signal_embeddings = self.signal_projector(all_signals)  # [batch, 64]
        signal_embeddings = signal_embeddings.unsqueeze(1)  # [batch, 1, 64]
        
        # Apply self-attention
        attended_signals, attention_weights = self.signal_attention(
            signal_embeddings, signal_embeddings, signal_embeddings
        )
        attended_signals = attended_signals.squeeze(1)  # [batch, 64]
        
        # Final fraud prediction
        fraud_logit = self.signal_fusion(attended_signals)
        
        # Add residual connection from raw signals
        signal_mean = all_signals.mean(dim=1, keepdim=True)
        fraud_logit = fraud_logit + self.residual_weight * signal_mean
        
        # Prepare outputs
        outputs = {
            'fraud_probability': fraud_logit,
            'is_network_member_pred': signals['network'],
            'has_price_coordination_pred': signals['price_coordination'],
            'has_inventory_sharing_pred': signals['inventory_sharing'],
            'has_registration_cluster_pred': signals['registration_cluster'],
            'exit_scam_risk_pred': signals['exit_scam'],
            'signals': signals,
            'attention_weights': attention_weights.squeeze(1) if attention_weights.dim() > 2 else attention_weights
        }
        
        return outputs


class EnhancedGraphEncoder(nn.Module):
    """Enhanced Graph Encoder with more layers and skip connections"""
    
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=4):
        super().__init__()
        
        self.convs = nn.ModuleList()
        self.batch_norms = nn.ModuleList()
        
        # First layer
        self.convs.append(GCNConv(input_dim, hidden_dim))
        self.batch_norms.append(nn.BatchNorm1d(hidden_dim))
        
        # Hidden layers with residual connections
        for _ in range(num_layers - 2):
            self.convs.append(GCNConv(hidden_dim, hidden_dim))
            self.batch_norms.append(nn.BatchNorm1d(hidden_dim))
        
        # Output layer
        self.convs.append(GCNConv(hidden_dim, output_dim))
        self.batch_norms.append(nn.BatchNorm1d(output_dim))
        
        self.dropout = nn.Dropout(0.2)
        
        # Skip connection weights
        self.skip_weight = nn.Parameter(torch.tensor(0.1))
        
    def forward(self, x, edge_index, batch=None):
        # First layer
        h = self.convs[0](x, edge_index)
        h = self.batch_norms[0](h)
        h = F.relu(h)
        h = self.dropout(h)
        
        # Hidden layers with skip connections
        for i in range(1, len(self.convs) - 1):
            h_prev = h
            h = self.convs[i](h, edge_index)
            h = self.batch_norms[i](h)
            h = F.relu(h)
            h = self.dropout(h)
            # Skip connection
            h = h + self.skip_weight * h_prev
        
        # Output layer
        h = self.convs[-1](h, edge_index)
        h = self.batch_norms[-1](h)
        
        # Global pooling if batch is provided
        if batch is not None:
            h = global_mean_pool(h, batch)
        
        return h


# Quick parameter count function
def count_model_parameters(model):
    """Count and display model parameters"""
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"\nTotal trainable parameters: {total_params:,}")
    print(f"Model size: {total_params * 4 / 1024 / 1024:.2f} MB\n")
    
    print("Parameter breakdown:")
    for name, module in model.named_children():
        params = sum(p.numel() for p in module.parameters() if p.requires_grad)
        print(f"  {name}: {params:,} parameters")
    
    return total_params

# ============= Custom Collate Function =============

def seller_collate_fn(batch):
    """Custom collate function for seller data with graphs"""
    
    # Regular features
    seller_ids = [item['seller_id'] for item in batch]
    seller_features = torch.stack([item['seller_features'] for item in batch])
    
    # Graph data - batch graphs together
    graphs = [item['graph'] for item in batch]
    batched_graph = Batch.from_data_list(graphs)
    
    # Other features
    registration_hours = torch.tensor([item['registration_hour'] for item in batch], dtype=torch.float32)
    days_active = torch.tensor([item['days_active'] for item in batch], dtype=torch.float32)
    total_products = torch.tensor([item['total_products'] for item in batch], dtype=torch.float32)
    
    # Labels
    labels = {}
    label_keys = batch[0]['labels'].keys()
    for key in label_keys:
        labels[key] = torch.tensor([item['labels'][key] for item in batch], dtype=torch.float32)
    
    return {
        'seller_ids': seller_ids,
        'seller_features': seller_features,
        'graph': batched_graph,
        'registration_hour': registration_hours,
        'days_active': days_active,
        'total_products': total_products,
        'labels': labels,
        'batch': batched_graph.batch  # For graph pooling
    }

# ============= Training Components =============

class FocalLoss(nn.Module):
    """Focal Loss for extreme class imbalance"""
    def __init__(self, alpha=1, gamma=2, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        
    def forward(self, inputs, targets):
        ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        p_t = torch.exp(-ce_loss)
        loss = self.alpha * (1-p_t)**self.gamma * ce_loss
        
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss

class SellerNetworkLoss(nn.Module):
    def __init__(self, config, pos_weight=10.0):  # Increased pos_weight
        super().__init__()
        self.config = config
        self.focal = FocalLoss(alpha=0.25, gamma=2.0)  # Adjusted alpha for class imbalance
        self.pos_weight = pos_weight
        
    def forward(self, predictions, targets):
        losses = {}
        device = predictions['fraud_probability'].device
        
        # Use Focal Loss for better handling of imbalance
        fraud_pred = predictions['fraud_probability'].squeeze()
        fraud_target = targets['is_fraud'].float()
        
        # Add a small epsilon to avoid log(0)
        fraud_pred = torch.clamp(fraud_pred, min=1e-7, max=1-1e-7)
        
        losses['network_detection'] = self.focal(fraud_pred, fraud_target)
        
        # For subtasks, use BCE with pos_weight
        pos_weight_tensor = torch.tensor([self.pos_weight]).to(device)
        
        task_mappings = {
            'price_coordination': ('has_price_coordination_pred', 'has_price_coordination'),
            'inventory_sharing': ('has_inventory_sharing_pred', 'has_inventory_sharing'),
            'registration_cluster': ('has_registration_cluster_pred', 'has_registration_cluster'),
            'exit_scam': ('exit_scam_risk_pred', 'exit_scam_risk')
        }
        
        for task, (pred_key, target_key) in task_mappings.items():
            if pred_key in predictions and target_key in targets:
                pred = predictions[pred_key].squeeze()
                target = targets[target_key].float()
                
                # Use BCEWithLogitsLoss for numerical stability
                criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight_tensor)
                losses[task] = criterion(pred, target)
        
        # Weighted sum with adjusted weights
        total_loss = sum(
            self.config.task_weights.get(k, 1.0) * v 
            for k, v in losses.items()
        )
        
        return total_loss, losses

class SellerNetworkTrainer:
    def __init__(self, model, config, device='cuda'):
        self.model = model.to(device)
        self.config = config
        self.device = device
        
        # Use Adam with very low learning rate
        self.optimizer = torch.optim.Adam(
            model.parameters(),
            lr=config.learning_rate,  # 1e-5
            weight_decay=1e-5
        )
        
        # NO SCHEDULER! Or use a gentle one
        self.scheduler = None
        
        self.criterion = SellerNetworkLoss(config, pos_weight=10.0)
        self.best_f1 = 0
        self.current_step = 0
        self.warmup_steps = config.warmup_steps
    
    def train(self, train_loader, val_loader, num_epochs):
        """Full training loop"""
        logger.info("Starting Seller Network Detection Training")
        
        # DON'T CREATE ONECYCLELR HERE!
        
        print(f"\n=== TRAINING CONFIG ===")
        print(f"Learning rate: {self.optimizer.param_groups[0]['lr']}")
        print(f"Scheduler: {type(self.scheduler).__name__ if self.scheduler else 'None'}")
        print(f"Batch size: {self.config.batch_size}")
        print(f"Epochs: {num_epochs}")
        
        for epoch in range(num_epochs):
            # Train
            train_loss = self.train_epoch(train_loader, epoch + 1, num_epochs)
            
            # Evaluate
            val_metrics = self.evaluate(val_loader)
            
            # If using ReduceLROnPlateau
            if self.scheduler and isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                self.scheduler.step(val_metrics['loss'])
            
            logger.info(f"\nEpoch {epoch + 1} Results:")
            logger.info(f"Train Loss: {train_loss:.4f}")
            logger.info(f"Val Precision: {val_metrics['precision']:.4f}")
            logger.info(f"Val Recall: {val_metrics['recall']:.4f}")
            logger.info(f"Val F1: {val_metrics['f1']:.4f}")
            logger.info(f"Val AUC: {val_metrics['auc']:.4f}")
            
            # Save best model
            if val_metrics['f1'] > self.best_f1:
                self.best_f1 = val_metrics['f1']
                self.save_model(f'seller_network_best_f1_{self.best_f1:.4f}.pt')



    
    def debug_predictions(self, outputs, labels, epoch, batch_idx=None):
        """Debug helper to understand predictions"""
        with torch.no_grad():
            fraud_probs = torch.sigmoid(outputs['fraud_probability']).cpu().numpy()
            true_labels = labels['is_fraud'].cpu().numpy()
            
            # Only print every 100 batches to avoid spam
            if batch_idx is not None and batch_idx % 100 != 0:
                return
            
            print(f"\n=== Epoch {epoch}, Batch {batch_idx} Debug Info ===")
            print(f"Fraud probability range: [{fraud_probs.min():.4f}, {fraud_probs.max():.4f}]")
            print(f"Mean fraud probability: {fraud_probs.mean():.4f}")
            print(f"Std fraud probability: {fraud_probs.std():.4f}")
            print(f"Predictions > 0.5: {(fraud_probs > 0.5).sum()} / {len(fraud_probs)}")
            print(f"Predictions > 0.3: {(fraud_probs > 0.3).sum()} / {len(fraud_probs)}")
            print(f"True positives in batch: {true_labels.sum()} / {len(true_labels)}")
            
            # Check if model is learning anything
            for task_name, pred_key in [
                ('network', 'is_network_member_pred'),
                ('price_coordination', 'has_price_coordination_pred'),
                ('inventory_sharing', 'has_inventory_sharing_pred'),
                ('registration_cluster', 'has_registration_cluster_pred'),
                ('exit_scam', 'exit_scam_risk_pred')
            ]:
                if pred_key in outputs:
                    task_probs = torch.sigmoid(outputs[pred_key]).cpu().numpy()
                    print(f"{task_name}: mean={task_probs.mean():.4f}, std={task_probs.std():.4f}")
    
    def train_epoch(self, dataloader, epoch_num, total_epochs):
        self.model.train()
        epoch_losses = []
        
        pbar = tqdm(dataloader, desc=f'Epoch {epoch_num}/{total_epochs}')
        
        for batch_idx, batch in enumerate(pbar):
            # Move to device
            batch = self.move_to_device(batch)
            
            # Forward pass
            outputs = self.model(batch)
            
            # Calculate loss
            loss, losses = self.criterion(outputs, batch['labels'])
            
            # Check for NaN
            if torch.isnan(loss):
                logger.warning("NaN loss detected, skipping batch")
                continue
            
            # ADD THIS: Debug predictions every 100 batches
            if batch_idx % 100 == 0:
                self.debug_predictions(outputs, batch['labels'], epoch_num, batch_idx)
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            
            epoch_losses.append(loss.item())
            
            # Update progress bar with prediction info
            with torch.no_grad():
                fraud_probs = torch.sigmoid(outputs['fraud_probability'])
                pbar.set_postfix({
                    'loss': f'{loss.item():.4f}',
                    'network': f'{losses.get("network_detection", 0):.4f}',
                    'lr': f'{self.optimizer.param_groups[0]["lr"]:.2e}',
                    'pred_mean': f'{fraud_probs.mean().item():.4f}',
                    'pred_max': f'{fraud_probs.max().item():.4f}'
                })
        
        return np.mean(epoch_losses)
    
    def evaluate(self, dataloader, desc="Validation"):
        """Complete evaluation function with diagnostics and error handling"""
        self.model.eval()
        all_preds = []
        all_labels = []
        total_loss = 0
        num_batches = 0
        
        # ADD THIS: Debug first batch during evaluation
        first_batch_debug = True
        
        pbar = tqdm(dataloader, desc=desc, unit='batch', leave=False)
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(pbar):
                batch = self.move_to_device(batch)
                outputs = self.model(batch)
                
                # ADD THIS: Debug first batch
                if first_batch_debug:
                    self.debug_predictions(outputs, batch['labels'], 'EVAL', 0)
                    first_batch_debug = False
                
                # Calculate loss for monitoring
                loss, _ = self.criterion(outputs, batch['labels'])
                total_loss += loss.item()
                num_batches += 1
                
                # Apply sigmoid since we're using BCEWithLogitsLoss
                preds = torch.sigmoid(outputs['fraud_probability']).cpu().numpy()
                labels = batch['labels']['is_fraud'].cpu().numpy()
                
                all_preds.extend(preds)
                all_labels.extend(labels)
                
                pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        # Convert to numpy arrays
        all_preds = np.array(all_preds).flatten()  # Ensure 1D
        all_labels = np.array(all_labels).flatten()  # Ensure 1D
        
        # DIAGNOSTIC: Check predictions
        print(f"\nDIAGNOSTIC INFO:")
        print(f"Pred min: {all_preds.min():.4f}, max: {all_preds.max():.4f}, mean: {all_preds.mean():.4f}")
        print(f"Pred std: {all_preds.std():.4f}")
        print(f"Label distribution: {np.unique(all_labels, return_counts=True)}")
        print(f"Predictions > 0.5: {(all_preds > 0.5).sum()}")
        print(f"Predictions > 0.3: {(all_preds > 0.3).sum()}")
        print(f"Predictions > 0.1: {(all_preds > 0.1).sum()}")
        
        # Try different thresholds
        thresholds = [0.5, 0.4, 0.3, 0.2, 0.1]
        best_f1 = 0
        best_threshold = 0.5
        
        for thresh in thresholds:
            binary_preds = (all_preds > thresh).astype(int)
            
            # Only calculate if we have both positive and negative predictions
            if binary_preds.sum() > 0 and binary_preds.sum() < len(binary_preds):
                try:
                    precision, recall, f1, _ = precision_recall_fscore_support(
                        all_labels, binary_preds, average='binary', zero_division=0
                    )
                    print(f"Threshold {thresh}: Precision={precision:.4f}, Recall={recall:.4f}, F1={f1:.4f}")
                    
                    if f1 > best_f1:
                        best_f1 = f1
                        best_threshold = thresh
                except:
                    pass
        
        print(f"Best threshold: {best_threshold} with F1={best_f1:.4f}")
        
        # Calculate final metrics with standard threshold (0.5)
        binary_preds = (all_preds > 0.5).astype(int)
        
        # Handle edge cases
        if len(np.unique(binary_preds)) == 1:
            # Model predicts all same class
            if binary_preds[0] == 0:
                # All negative predictions
                precision = 0.0
                recall = 0.0
                f1 = 0.0
                print("WARNING: Model predicting all negative (0)")
            else:
                # All positive predictions
                true_positives = (all_labels == 1).sum()
                false_positives = (all_labels == 0).sum()
                precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
                recall = 1.0 if true_positives > 0 else 0
                f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
                print("WARNING: Model predicting all positive (1)")
        else:
            # Normal case - mixed predictions
            try:
                precision, recall, f1, _ = precision_recall_fscore_support(
                    all_labels, binary_preds, average='binary', zero_division=0
                )
            except Exception as e:
                print(f"Error calculating metrics: {e}")
                precision = recall = f1 = 0.0
        
        # Calculate AUC
        try:
            # AUC requires at least one positive and one negative sample
            if len(np.unique(all_labels)) > 1:
                auc = roc_auc_score(all_labels, all_preds)
            else:
                auc = 0.5
                print("WARNING: Only one class in labels, AUC set to 0.5")
        except Exception as e:
            print(f"Error calculating AUC: {e}")
            auc = 0.5
        
        # Calculate average loss
        avg_loss = total_loss / num_batches if num_batches > 0 else 0
        
        # Additional metrics for debugging
        true_positives = ((binary_preds == 1) & (all_labels == 1)).sum()
        false_positives = ((binary_preds == 1) & (all_labels == 0)).sum()
        true_negatives = ((binary_preds == 0) & (all_labels == 0)).sum()
        false_negatives = ((binary_preds == 0) & (all_labels == 1)).sum()
        
        print(f"\nConfusion Matrix:")
        print(f"TP: {true_positives}, FP: {false_positives}")
        print(f"FN: {false_negatives}, TN: {true_negatives}")
        
        # Calculate per-task metrics if needed
        task_metrics = {}
        for task in ['is_network_member_pred', 'has_price_coordination_pred', 
                    'has_inventory_sharing_pred', 'has_registration_cluster_pred', 
                    'exit_scam_risk_pred']:
            if task in outputs:
                task_preds = torch.sigmoid(outputs[task]).cpu().numpy()
                task_metrics[task] = {
                    'mean': np.mean(task_preds),
                    'std': np.std(task_preds)
                }
        
        if task_metrics:
            print(f"\nTask-specific predictions:")
            for task, stats in task_metrics.items():
                print(f"  {task}: mean={stats['mean']:.4f}, std={stats['std']:.4f}")
        
        # Return metrics
        metrics = {
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'auc': auc,
            'loss': avg_loss,
            'best_threshold': best_threshold,
            'best_f1': best_f1,
            'confusion_matrix': {
                'tp': int(true_positives),
                'fp': int(false_positives),
                'tn': int(true_negatives),
                'fn': int(false_negatives)
            }
        }
        
        return metrics
    
    def move_to_device(self, batch):
        """Move batch to device"""
        moved = {}
        
        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                moved[k] = v.to(self.device)
            elif isinstance(v, dict):
                moved[k] = {
                    k2: v2.to(self.device) if isinstance(v2, torch.Tensor) else v2
                    for k2, v2 in v.items()
                }
            elif k == 'graph':
                # Move graph data
                moved[k] = v.to(self.device)
            else:
                moved[k] = v
                
        return moved
    
    
    def save_model(self, path):
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'config': self.config,
            'best_f1': self.best_f1
        }, path)

# ============= Integration Interface =============

class SellerNetworkInterface:
    """Interface for integration with other TrustSight components"""
    
    def __init__(self, model_path: str, config: SellerNetworkConfig, device='cuda'):
        self.device = device
        self.config = config
        
        # Load model
        self.model = SellerNetworkDetectionModel(config)
        checkpoint = torch.load(model_path, map_location=device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(device)
        self.model.eval()
        
    def analyze_seller(self, seller_data: Dict) -> Dict:
        """Analyze a single seller"""
        # Prepare features
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
        """Analyze a network of sellers"""
        results = []
        
        for seller in sellers:
            result = self.analyze_seller(seller)
            results.append(result)
        
        # Aggregate network statistics
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

# ============= Main Execution =============

def validate_and_fix_seller_data(data_path: str, output_path: str = None):
    """Validate and fix seller data before training"""
    
    print("Loading seller data for validation...")
    with open(data_path, 'r') as f:
        if isinstance(data_path, str) and data_path.endswith('.json'):
            sellers = json.load(f)
        else:
            sellers = data_path  # If data is passed directly
    
    fixed_count = 0
    
    for seller in sellers:
        # Fix 1: active_products should be <= total_products
        metrics = seller['seller_metrics']
        if metrics['active_products'] > metrics['total_products']:
            metrics['active_products'] = int(metrics['total_products'] * 0.8)
            fixed_count += 1
        
        # Fix 2: Ensure network connections are lists
        network = seller['network_features']
        if network['shared_products_with_sellers'] is None:
            network['shared_products_with_sellers'] = []
        if network['same_day_registrations'] is None:
            network['same_day_registrations'] = []
        
        # Fix 3: Ensure all numeric fields are not None
        for key in ['shared_product_count', 'address_similarity_score', 
                    'customer_overlap_score', 'reviewer_overlap_score']:
            if network[key] is None:
                network[key] = 0
        
        # Fix 4: Ensure temporal features
        if seller['temporal_features']['activity_gaps'] is None:
            seller['temporal_features']['activity_gaps'] = []
        if seller['temporal_features']['peak_activity_hours'] is None:
            seller['temporal_features']['peak_activity_hours'] = []
    
    print(f"Fixed {fixed_count} data inconsistencies")
    
    # Save fixed data
    if output_path:
        with open(output_path, 'w') as f:
            json.dump(sellers, f, indent=2)
        print(f"Saved fixed data to {output_path}")
    
    return sellers

def analyze_dataset_distribution(data_path: str):
    """Analyze the dataset distribution"""
    
    with open(data_path, 'r') as f:
        sellers = json.load(f)
    
    df = pd.DataFrame(sellers)
    
    print("\n" + "="*50)
    print("DATASET ANALYSIS")
    print("="*50)
    
    print(f"\nTotal sellers: {len(df)}")
    
    # Fraud distribution
    fraud_counts = df['labels'].apply(lambda x: x['is_fraud']).value_counts()
    print(f"\nFraud distribution:")
    print(f"  Legitimate: {fraud_counts.get(0, 0)} ({fraud_counts.get(0, 0)/len(df)*100:.1f}%)")
    print(f"  Fraudulent: {fraud_counts.get(1, 0)} ({fraud_counts.get(1, 0)/len(df)*100:.1f}%)")
    
    # Specific patterns
    print(f"\nSpecific fraud patterns:")
    patterns = ['is_network_member', 'has_price_coordination', 'has_inventory_sharing', 
                'has_registration_cluster', 'exit_scam_risk']
    
    for pattern in patterns:
        count = df['labels'].apply(lambda x: x['specific_patterns'][pattern]).sum()
        print(f"  {pattern}: {count} ({count/len(df)*100:.1f}%)")
    
    # Network statistics
    network_sellers = df[df['network_id'].notna()]
    print(f"\nNetwork statistics:")
    print(f"  Sellers in networks: {len(network_sellers)}")
    print(f"  Unique networks: {df['network_id'].nunique() - 1}")  # -1 for None
    print(f"  Avg sellers per network: {network_sellers['connected_seller_count'].mean():.1f}")
    
    return df

def split_seller_dataset(data_path: str, train_ratio: float = 0.7, val_ratio: float = 0.15):
    """Split seller dataset maintaining network integrity"""
    
    print("\nSplitting dataset...")
    
    with open(data_path, 'r') as f:
        sellers = json.load(f)
    
    df = pd.DataFrame(sellers)
    
    # Separate network and non-network sellers
    network_sellers = df[df['network_id'].notna()].copy()
    non_network_sellers = df[df['network_id'].isna()].copy()
    
    print(f"Network sellers: {len(network_sellers)}")
    print(f"Non-network sellers: {len(non_network_sellers)}")
    
    # Split network sellers by network_id to keep networks intact
    if len(network_sellers) > 0:
        unique_networks = network_sellers['network_id'].unique()
        
        # Split networks
        train_networks, temp_networks = train_test_split(
            unique_networks, test_size=(1-train_ratio), random_state=42
        )
        val_networks, test_networks = train_test_split(
            temp_networks, test_size=0.5, random_state=42
        )
        
        # Get sellers for each split
        train_network_sellers = network_sellers[network_sellers['network_id'].isin(train_networks)]
        val_network_sellers = network_sellers[network_sellers['network_id'].isin(val_networks)]
        test_network_sellers = network_sellers[network_sellers['network_id'].isin(test_networks)]
    else:
        train_network_sellers = pd.DataFrame()
        val_network_sellers = pd.DataFrame()
        test_network_sellers = pd.DataFrame()
    
    # Split non-network sellers normally
    if len(non_network_sellers) > 0:
        # Stratify by fraud label
        fraud_labels = non_network_sellers['labels'].apply(lambda x: x['is_fraud'])
        
        train_non_network, temp_non_network = train_test_split(
            non_network_sellers, 
            test_size=(1-train_ratio), 
            stratify=fraud_labels,
            random_state=42
        )
        
        temp_fraud_labels = temp_non_network['labels'].apply(lambda x: x['is_fraud'])
        val_non_network, test_non_network = train_test_split(
            temp_non_network, 
            test_size=0.5, 
            stratify=temp_fraud_labels,
            random_state=42
        )
    else:
        train_non_network = pd.DataFrame()
        val_non_network = pd.DataFrame()
        test_non_network = pd.DataFrame()
    
    # Combine
    train_df = pd.concat([train_network_sellers, train_non_network], ignore_index=True)
    val_df = pd.concat([val_network_sellers, val_non_network], ignore_index=True)
    test_df = pd.concat([test_network_sellers, test_non_network], ignore_index=True)
    
    # Shuffle
    train_df = train_df.sample(frac=1, random_state=42).reset_index(drop=True)
    val_df = val_df.sample(frac=1, random_state=42).reset_index(drop=True)
    test_df = test_df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    print(f"\nDataset split:")
    print(f"Train: {len(train_df)} sellers ({len(train_df)/len(df)*100:.1f}%)")
    print(f"Val: {len(val_df)} sellers ({len(val_df)/len(df)*100:.1f}%)")
    print(f"Test: {len(test_df)} sellers ({len(test_df)/len(df)*100:.1f}%)")
    
    # Save splits
    train_df.to_json('train_sellers.json', orient='records', indent=2)
    val_df.to_json('val_sellers.json', orient='records', indent=2)
    test_df.to_json('test_sellers.json', orient='records', indent=2)
    
    print("\nSplits saved!")
    
    return train_df, val_df, test_df

# ============= UPDATE YOUR MAIN FUNCTION =============
def check_class_balance(dataset_path):
    with open(dataset_path, 'r') as f:
        data = json.load(f)
    
    fraud_count = sum(1 for item in data if item['labels']['is_fraud'] == 1)
    total = len(data)
    
    print(f"Dataset: {dataset_path}")
    print(f"Total: {total}, Fraud: {fraud_count} ({fraud_count/total*100:.1f}%)")
    print(f"Class ratio: {(total-fraud_count)/fraud_count:.2f}:1")

def main():
    """Main training script with data validation"""
    
    # Step 1: Validate and fix data
    print("Step 1: Validating data...")
    validate_and_fix_seller_data('sellers_dataset_100k.json', 'sellers_10000_fixed.json')
    
    # Step 2: Analyze distribution
    print("\nStep 2: Analyzing dataset...")
    analyze_dataset_distribution('sellers_10000_fixed.json')
    
    # Step 3: Split dataset
    print("\nStep 3: Splitting dataset...")
    train_df, val_df, test_df = split_seller_dataset('sellers_10000_fixed.json')
    
    # Step 4: Initialize config
    config = SellerNetworkConfig()
    
    # Step 5: Load datasets
    print("\nStep 5: Loading datasets for training...")
    train_dataset = SellerNetworkDataset('train_sellers.json', config)
    val_dataset = SellerNetworkDataset('val_sellers.json', config)
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        collate_fn=seller_collate_fn,
        num_workers=4
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        collate_fn=seller_collate_fn,
        num_workers=4
    )
    
    # Initialize model
    model = SellerNetworkDetectionModel(config)
    print(f"\nModel initialized with {sum(p.numel() for p in model.parameters()):,} parameters")
    
    # Initialize trainer
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    trainer = SellerNetworkTrainer(model, config, device)
    
    # Train
    print("\nStarting training...")
    trainer.train(train_loader, val_loader, config.num_epochs)
    
    print("\nTraining completed!")

# ============= QUICK TEST SCRIPT =============

def quick_test_data_loading():
    """Quick test to ensure data loads correctly"""
    
    # Test loading one seller
    with open('sellers_dataset_100k.json', 'r') as f:
        sellers = json.load(f)
    
    # Test feature extraction
    config = SellerNetworkConfig()
    dataset = SellerNetworkDataset('sellers_dataset_100k.json', config)
    
    # Get one sample
    sample = dataset[0]
    
    print(f"Seller ID: {sample['seller_id']}")
    print(f"Feature shape: {sample['seller_features'].shape}")
    print(f"Expected shape: torch.Size([21])")
    print(f"Labels: {sample['labels']}")
    
    return sample
def validate_and_fix_seller_data(data_path: str, output_path: str = None):
    """Validate and fix seller data before training"""
    
    print("Loading seller data for validation...")
    with open(data_path, 'r') as f:
        if isinstance(data_path, str) and data_path.endswith('.json'):
            sellers = json.load(f)
        else:
            sellers = data_path  # If data is passed directly
    
    fixed_count = 0
    
    for seller in sellers:
        # Fix 1: active_products should be <= total_products
        metrics = seller['seller_metrics']
        if metrics['active_products'] > metrics['total_products']:
            metrics['active_products'] = int(metrics['total_products'] * 0.8)
            fixed_count += 1
        
        # Fix 2: Ensure network connections are lists
        network = seller['network_features']
        if network['shared_products_with_sellers'] is None:
            network['shared_products_with_sellers'] = []
        if network['same_day_registrations'] is None:
            network['same_day_registrations'] = []
        
        # Fix 3: Ensure all numeric fields are not None
        for key in ['shared_product_count', 'address_similarity_score', 
                    'customer_overlap_score', 'reviewer_overlap_score']:
            if network[key] is None:
                network[key] = 0
        
        # Fix 4: Ensure temporal features
        if seller['temporal_features']['activity_gaps'] is None:
            seller['temporal_features']['activity_gaps'] = []
        if seller['temporal_features']['peak_activity_hours'] is None:
            seller['temporal_features']['peak_activity_hours'] = []
    
    print(f"Fixed {fixed_count} data inconsistencies")
    
    # Save fixed data
    if output_path:
        with open(output_path, 'w') as f:
            json.dump(sellers, f, indent=2)
        print(f"Saved fixed data to {output_path}")
    
    return sellers

# Add this before training
def check_label_distribution(json_file):
    with open(json_file, 'r') as f:
        data = json.load(f)
    
    fraud_count = sum(1 for seller in data if seller['labels']['is_fraud'] == 1)
    total = len(data)
    
    print(f"\nDataset: {json_file}")
    print(f"Total sellers: {total}")
    print(f"Fraud sellers: {fraud_count} ({fraud_count/total*100:.1f}%)")
    print(f"Legitimate sellers: {total-fraud_count} ({(total-fraud_count)/total*100:.1f}%)")
    
    # Check specific patterns
    patterns = ['is_network_member', 'has_price_coordination', 'has_inventory_sharing', 
                'has_registration_cluster', 'exit_scam_risk']
    
    for pattern in patterns:
        count = sum(1 for seller in data if seller['labels']['specific_patterns'][pattern] == 1)
        print(f"{pattern}: {count} ({count/total*100:.1f}%)")

# Check all your datasets
def check_data_quality(json_file):
    with open(json_file, 'r') as f:
        data = json.load(f)
    
    print(f"\nChecking {json_file} for data quality issues...")
    
    issues = []
    for i, seller in enumerate(data):
        # Check for negative metrics
        metrics = seller['seller_metrics']
        if any(metrics[k] < 0 for k in ['total_products', 'avg_rating', 'response_time_hours']):
            issues.append(f"Seller {i}: Negative metrics")
        
        # Check for impossible values
        if metrics['return_rate'] > 1.0 or metrics['fulfillment_rate'] > 1.0:
            issues.append(f"Seller {i}: Rate > 1.0")
        
        # Check network features
        network = seller['network_features']
        if any(network[k] is None for k in ['shared_product_count', 'address_similarity_score']):
            issues.append(f"Seller {i}: None values in network features")
    
    print(f"Found {len(issues)} issues")
    for issue in issues[:10]:  # Show first 10
        print(f"  - {issue}")
def analyze_feature_distributions(json_file, num_samples=1000):
    """Check if features are in reasonable ranges"""
    with open(json_file, 'r') as f:
        data = json.load(f)
    
    # Sample data if too large
    import random
    if len(data) > num_samples:
        data = random.sample(data, num_samples)
    
    print(f"\nAnalyzing feature distributions in {json_file}...")
    
    # Check seller metrics ranges
    metrics_ranges = {
        'total_products': [],
        'avg_product_price': [],
        'avg_rating': [],
        'response_time_hours': [],
        'customer_complaints': []
    }
    
    for seller in data:
        metrics = seller['seller_metrics']
        for key in metrics_ranges:
            if key in metrics:
                metrics_ranges[key].append(metrics[key])
    
    # Print statistics
    import numpy as np
    for key, values in metrics_ranges.items():
        if values:
            print(f"\n{key}:")
            print(f"  Min: {np.min(values):.2f}, Max: {np.max(values):.2f}")
            print(f"  Mean: {np.mean(values):.2f}, Std: {np.std(values):.2f}")


if __name__ == "__main__":
    # First run this to test
    #quick_test_data_loading()
    #check_class_balance('train_sellers.json')
    #check_class_balance('val_sellers.json')
    validate_and_fix_seller_data('sellers_dataset_100k.json')
    # Then run main training
    # check_data_quality("train_sellers.json")
    # check_data_quality("val_sellers.json")
    # check_data_quality("test_sellers.json")
    # check_label_distribution('train_sellers.json')
    # check_label_distribution('val_sellers.json')
    # check_label_distribution('test_sellers.json')
    analyze_feature_distributions('train_sellers.json')
    main()