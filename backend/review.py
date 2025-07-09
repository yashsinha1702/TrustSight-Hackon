import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from transformers import AutoModel, AutoTokenizer, AutoConfig
from datetime import datetime
import json
from typing import Dict, List, Tuple, Optional
import logging
from dataclasses import dataclass
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score
from typing import Dict, Any
# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ============= Configuration =============
from tqdm import tqdm
import time
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from IPython.display import clear_output
import wandb  # Optional: for advanced tracking

# Enhanced Configuration
@dataclass
class ModelConfig:
    """Configuration for TrustSight Review Fraud Detector"""
    model_name: str = "roberta-base"
    hidden_size: int = 768
    num_fraud_types: int = 6
    max_length: int = 256
    dropout_rate: float = 0.1
    learning_rate: float = 2e-5
    batch_size: int = 32
    num_epochs: int = 10
    warmup_steps: int = 500
    
    # Add monitoring config
    log_every_n_steps: int = 10
    validate_every_n_steps: int = 100
    save_every_n_epochs: int = 1
    use_wandb: bool = False  # Set to True if you want W&B tracking
    
    # Task weights for multi-task learning
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

# ============= Dataset Class =============
class ReviewFraudDataset(Dataset):
    """Dataset for review fraud detection"""
    
    def __init__(self, data_path: str, tokenizer, config: ModelConfig):
        self.tokenizer = tokenizer
        self.config = config
        self.data = self.load_data(data_path)
        
    def load_data(self, data_path: str) -> pd.DataFrame:
        """Load and preprocess data"""
        if data_path.endswith('.json'):
            data = pd.read_json(data_path)
        else:
            data = pd.read_csv(data_path)
        
        # Convert timestamps to features
        data['review_timestamp'] = pd.to_datetime(data['review_timestamp'])
        data['delivery_timestamp'] = pd.to_datetime(data['delivery_timestamp'])
        data['hours_since_delivery'] = (
            data['review_timestamp'] - data['delivery_timestamp']
        ).dt.total_seconds() / 3600
        
        return data
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        
        # Tokenize text
        text_inputs = self.tokenizer(
            row['review_text'],
            row.get('title', ''),  # Handle missing title
            padding='max_length',
            truncation=True,
            max_length=self.config.max_length,
            return_tensors='pt'
        )
        
        # Extract features and ensure float32
        features = {
            'input_ids': text_inputs['input_ids'].squeeze(),
            'attention_mask': text_inputs['attention_mask'].squeeze(),
            
            # Timing features - convert to float32
            'review_hour': torch.tensor(row.get('review_hour', 0), dtype=torch.float32),
            'hours_since_delivery': torch.tensor(row.get('hours_since_delivery', 0), dtype=torch.float32),
            'review_velocity': torch.tensor(row.get('reviews_per_day_avg', 0), dtype=torch.float32),
            
            # Reviewer features - convert to float32
            'reviewer_account_age': torch.tensor(row.get('reviewer_account_age_days', 0), dtype=torch.float32),
            'reviewer_total_reviews': torch.tensor(row.get('reviewer_total_reviews', 0), dtype=torch.float32),
            'verified_purchase_rate': torch.tensor(row.get('reviewer_verified_purchases_pct', 0), dtype=torch.float32),
            
            # Labels - IMPORTANT: Convert to float32
            'labels': {
                'is_fraud': torch.tensor(float(row['is_fraud']), dtype=torch.float32),
                'is_generic_text': torch.tensor(float(row.get('is_generic_text', 0)), dtype=torch.float32),
                'has_timing_anomaly': torch.tensor(float(row.get('has_timing_anomaly', 0)), dtype=torch.float32),
                'is_bot_reviewer': torch.tensor(float(row.get('is_bot_reviewer', 0)), dtype=torch.float32),
                'is_incentivized': torch.tensor(float(row.get('is_incentivized', 0)), dtype=torch.float32),
                'part_of_network': torch.tensor(float(row.get('part_of_network', 0)), dtype=torch.float32)
            }
        }
        
        return features

# ============= Model Components =============

class TemporalEncoder(nn.Module):
    """Encode temporal patterns in review timing"""
    
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
        # temporal_features shape: (batch_size, seq_len, features)
        lstm_out, _ = self.lstm(temporal_features)
        # Take the last output
        last_output = lstm_out[:, -1, :]
        return self.projection(last_output)

class ReviewAuthenticityHead(nn.Module):
    """Detect generic/template text patterns"""
    
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
        
        # Attention to combine sub-components
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
        # Detect different authenticity signals
        generic = self.generic_detector(text_features)
        specificity = self.specificity_scorer(text_features)
        emotion = self.emotion_consistency(text_features)
        
        # Stack for attention
        signals = torch.stack([generic, specificity, emotion], dim=1)
        attended, _ = self.attention(signals, signals, signals)
        
        # Combine all signals
        combined = torch.cat([generic, specificity, emotion], dim=1)
        return self.output(combined)

class TimingAnomalyHead(nn.Module):
    """Detect timing-based fraud patterns"""
    
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
        # Process temporal patterns
        temporal = self.temporal_processor(temporal_features)
        
        # Detect bot patterns (timing + text)
        combined = torch.cat([temporal_features, text_features], dim=1)
        bot_patterns = self.bot_pattern_detector(combined)
        
        # Analyze velocity
        velocity = self.velocity_analyzer(temporal_features)
        
        # Combine all timing signals
        all_features = torch.cat([temporal, bot_patterns, velocity], dim=1)
        return self.output(all_features)

class ReviewerBehaviorHead(nn.Module):
    """Classify reviewer behavior patterns"""
    
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
            nn.Linear(512, 256),  # bidirectional
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 4)  # genuine, bot, incentivized, farm
        )
        
        # Add a final layer to produce single fraud score
        self.fraud_score = nn.Linear(4, 1)
        
    def forward(self, text_features, reviewer_features):
        # Process sequence of reviews (if available)
        if len(text_features.shape) == 2:
            text_features = text_features.unsqueeze(1)
        
        encoded, _ = self.behavior_encoder(text_features)
        last_hidden = encoded[:, -1, :]
        
        # Get behavior classifications
        behavior_logits = self.pattern_classifier(last_hidden)
        
        # Convert to single fraud score for fusion
        fraud_score = self.fraud_score(behavior_logits)
        
        return fraud_score  # Now returns [batch_size, 1]

class NetworkPatternHead(nn.Module):
    """Detect network-based fraud patterns"""
    
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
        # Encode text features
        encoded = self.network_encoder(text_features)
        
        # Self-attention to find patterns
        attended, _ = self.graph_attention(
            encoded.unsqueeze(1),
            encoded.unsqueeze(1),
            encoded.unsqueeze(1)
        )
        
        return self.output(attended.squeeze(1))

class SignalFusionLayer(nn.Module):
    """Fuse signals from all detection heads"""
    
    def __init__(self, num_signals: int = 4, signal_dim: int = 1):
        super().__init__()
        
        # Since all signals now have dimension 1, we concatenate them
        self.fusion_network = nn.Sequential(
            nn.Linear(num_signals, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 128)
        )
        
        # Optional: attention mechanism
        self.signal_attention = nn.Linear(num_signals, num_signals)
        
    def forward(self, signals: Dict[str, torch.Tensor]):
        # Concatenate all signals
        signal_list = [s.squeeze(-1) if s.shape[-1] == 1 else s for s in signals.values()]
        concatenated = torch.cat([s.unsqueeze(-1) for s in signal_list], dim=1)
        
        # Apply attention weights
        attention_weights = torch.softmax(self.signal_attention(concatenated), dim=1)
        weighted_signals = concatenated * attention_weights
        
        # Fuse signals
        fused = self.fusion_network(weighted_signals)
        
        return fused, attention_weights

# ============= Main Model =============

class TrustSightReviewFraudDetector(nn.Module):
    """Unified review fraud detection model"""
    
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        
        # Text encoder (pre-trained transformer)
        self.text_encoder = AutoModel.from_pretrained(config.model_name)
        
        # Temporal encoder for timing features
        self.temporal_encoder = TemporalEncoder(
            input_dim=3,  # hour, hours_since_delivery, velocity
            hidden_dim=128
        )
        
        # Detection heads
        self.authenticity_head = ReviewAuthenticityHead(config.hidden_size)
        self.timing_head = TimingAnomalyHead(128, config.hidden_size)
        self.behavior_head = ReviewerBehaviorHead(config.hidden_size)
        self.network_head = NetworkPatternHead(config.hidden_size)
        
        # Signal fusion - now expects 4 signals of dimension 1
        self.signal_fusion = SignalFusionLayer(num_signals=4, signal_dim=1)
        
        # Final classifiers
        self.fraud_classifier = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(config.dropout_rate),
            nn.Linear(64, 1)
        )
        
        # Multi-task outputs
        self.task_classifiers = nn.ModuleDict({
            'generic_text': nn.Linear(128, 1),
            'timing_anomaly': nn.Linear(128, 1),
            'bot_reviewer': nn.Linear(128, 1),
            'incentivized': nn.Linear(128, 1),
            'network_fraud': nn.Linear(128, 1)
        })
        
        # Separate classifier for detailed behavior types
        self.behavior_type_classifier = nn.Linear(128, 4)  # For detailed behavior analysis
        
    def forward(self, batch):
        # Encode text
        text_outputs = self.text_encoder(
            input_ids=batch['input_ids'],
            attention_mask=batch['attention_mask']
        )
        text_features = text_outputs.last_hidden_state.mean(dim=1)
        
        # Prepare temporal features
        temporal_input = torch.stack([
            batch['review_hour'].float(),
            batch['hours_since_delivery'].float(),
            batch['review_velocity'].float()
        ], dim=1).unsqueeze(1)
        
        temporal_features = self.temporal_encoder(temporal_input)
        
        # Run detection heads - all now output [batch_size, 1]
        signals = {}
        signals['authenticity'] = self.authenticity_head(text_features)
        signals['timing'] = self.timing_head(temporal_features, text_features)
        signals['behavior'] = self.behavior_head(text_features, None)
        signals['network'] = self.network_head(text_features)
        
        # Fuse signals
        fused_features, attention_weights = self.signal_fusion(signals)
        
        # Generate outputs
        outputs = {
            'fraud_probability': torch.sigmoid(self.fraud_classifier(fused_features)),
            'signals': signals,
            'attention_weights': attention_weights
        }
        
        # Multi-task predictions
        for task, classifier in self.task_classifiers.items():
            outputs[f'{task}_pred'] = torch.sigmoid(classifier(fused_features))
        
        # Add detailed behavior classification
        outputs['behavior_types'] = torch.softmax(
            self.behavior_type_classifier(fused_features), 
            dim=1
        )
        
        return outputs

# ============= Training Logic =============

class MultiTaskLoss(nn.Module):
    """Multi-task loss for fraud detection"""
    
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.bce = nn.BCELoss()
        
    def forward(self, predictions, targets):
        losses = {}
        
        # Main fraud detection loss
        losses['overall_fraud'] = self.bce(
            predictions['fraud_probability'].squeeze(),
            targets['is_fraud']
        )
        
        # Task-specific losses - FIXED KEY MAPPING
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
        
        # Weighted combination
        total_loss = sum(
            self.config.task_weights.get(k, 1.0) * v 
            for k, v in losses.items()
        )
        
        return total_loss, losses

class FraudDetectionTrainer:
    """Trainer for the fraud detection model with comprehensive monitoring"""
    
    def __init__(self, model, config: ModelConfig, device='cuda'):
        self.model = model.to(device)
        self.config = config
        self.device = device
        
        # Optimizer
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=0.01
        )
        
        # Learning rate scheduler
        self.scheduler = None  # Will be set after knowing training steps
        
        # Loss function
        self.criterion = MultiTaskLoss(config)
        
        # Metrics tracking
        self.train_losses = []
        self.val_losses = []
        self.val_metrics = []
        self.best_f1 = 0
        self.start_time = None
        
        # Initialize wandb if requested
        if config.use_wandb:
            wandb.init(
                project="trustsight-fraud-detection",
                config=config.__dict__
            )
            wandb.watch(model)
    
    def train_epoch(self, dataloader, epoch_num, total_epochs):
        self.model.train()
        epoch_losses = []
        task_losses = {task: [] for task in self.config.task_weights.keys()}
        
        # Progress bar for batches
        pbar = tqdm(dataloader, desc=f'Epoch {epoch_num}/{total_epochs}', 
                    unit='batch', leave=True)
        
        for batch_idx, batch in enumerate(pbar):
            # Move to device
            batch = self.move_to_device(batch)
            
            # Forward pass
            outputs = self.model(batch)
            
            # Calculate loss
            loss, losses = self.criterion(outputs, batch['labels'])
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            
            if self.scheduler:
                self.scheduler.step()
            
            # Track losses
            epoch_losses.append(loss.item())
            for task, task_loss in losses.items():
                task_losses[task].append(task_loss.item())
            
            # Update progress bar
            current_lr = self.optimizer.param_groups[0]['lr']
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'lr': f'{current_lr:.2e}',
                'fraud_loss': f'{losses.get("overall_fraud", 0):.4f}'
            })
            
            # Log every N steps
            if batch_idx % self.config.log_every_n_steps == 0:
                self.log_step_metrics(
                    epoch_num, batch_idx, len(dataloader), 
                    loss.item(), losses, current_lr
                )
            
            # Optional: Validate during training
            if batch_idx % self.config.validate_every_n_steps == 0 and batch_idx > 0:
                self.model.eval()
                val_metrics = self.quick_validation(dataloader)
                self.model.train()
                logger.info(f"Mid-epoch validation - F1: {val_metrics['f1']:.4f}")
        
        # Calculate average losses
        avg_losses = {
            'total': np.mean(epoch_losses),
            **{task: np.mean(task_losses[task]) for task in task_losses}
        }
        
        return avg_losses
    
    def evaluate(self, dataloader, desc="Validation"):
        self.model.eval()
        all_preds = []
        all_labels = []
        total_loss = 0
        num_batches = 0
        
        # Progress bar for validation
        pbar = tqdm(dataloader, desc=desc, unit='batch', leave=False)
        
        with torch.no_grad():
            for batch in pbar:
                batch = self.move_to_device(batch)
                outputs = self.model(batch)
                
                # Calculate loss
                loss, _ = self.criterion(outputs, batch['labels'])
                total_loss += loss.item()
                num_batches += 1
                
                preds = outputs['fraud_probability'].cpu().numpy()
                labels = batch['labels']['is_fraud'].cpu().numpy()
                
                all_preds.extend(preds)
                all_labels.extend(labels)
                
                pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        # Calculate metrics
        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)
        
        # Binary predictions
        binary_preds = (all_preds > 0.5).astype(int)
        
        # Metrics
        precision, recall, f1, _ = precision_recall_fscore_support(
            all_labels, binary_preds, average='binary'
        )
        auc = roc_auc_score(all_labels, all_preds)
        
        metrics = {
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'auc': auc,
            'loss': total_loss / num_batches
        }
        
        return metrics
    
    def train(self, train_loader, val_loader, num_epochs):
        """Full training loop with comprehensive monitoring"""
        
        # Setup learning rate scheduler
        total_steps = len(train_loader) * num_epochs
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=total_steps
        )
        
        self.start_time = time.time()
        
        # Training header
        logger.info("=" * 80)
        logger.info("Starting TrustSight Fraud Detection Training")
        logger.info(f"Model: {self.config.model_name}")
        logger.info(f"Device: {self.device}")
        logger.info(f"Epochs: {num_epochs}")
        logger.info(f"Batch size: {self.config.batch_size}")
        logger.info(f"Learning rate: {self.config.learning_rate}")
        logger.info(f"Total training samples: {len(train_loader.dataset)}")
        logger.info(f"Total validation samples: {len(val_loader.dataset)}")
        logger.info("=" * 80)
        
        # Main training loop
        for epoch in range(num_epochs):
            epoch_start_time = time.time()
            
            # Train
            logger.info(f"\n{'='*50}")
            logger.info(f"EPOCH {epoch + 1}/{num_epochs}")
            logger.info(f"{'='*50}")
            
            train_losses = self.train_epoch(train_loader, epoch + 1, num_epochs)
            self.train_losses.append(train_losses['total'])
            
            # Evaluate
            logger.info("\nRunning validation...")
            val_metrics = self.evaluate(val_loader, desc=f"Validation Epoch {epoch + 1}")
            self.val_losses.append(val_metrics['loss'])
            self.val_metrics.append(val_metrics)
            
            # Log epoch summary
            epoch_time = time.time() - epoch_start_time
            total_time = time.time() - self.start_time
            eta = (total_time / (epoch + 1)) * (num_epochs - epoch - 1)
            
            logger.info(f"\nEpoch {epoch + 1} Summary:")
            logger.info(f"  Train Loss: {train_losses['total']:.4f}")
            logger.info(f"  Val Loss: {val_metrics['loss']:.4f}")
            logger.info(f"  Val Precision: {val_metrics['precision']:.4f}")
            logger.info(f"  Val Recall: {val_metrics['recall']:.4f}")
            logger.info(f"  Val F1: {val_metrics['f1']:.4f}")
            logger.info(f"  Val AUC: {val_metrics['auc']:.4f}")
            logger.info(f"  Epoch Time: {self.format_time(epoch_time)}")
            logger.info(f"  Total Time: {self.format_time(total_time)}")
            logger.info(f"  ETA: {self.format_time(eta)}")
            
            # Log task-specific losses
            logger.info("\nTask-specific losses:")
            for task, loss in train_losses.items():
                if task != 'total':
                    logger.info(f"  {task}: {loss:.4f}")
            
            # Save best model
            if val_metrics['f1'] >= self.best_f1:
                self.best_f1 = val_metrics['f1']
                self.save_model(f'best_model_f1_{self.best_f1:.4f}.pt')
                logger.info(f"  ✓ New best F1 score: {self.best_f1:.4f}")
            
            # Save checkpoint every N epochs
            if (epoch + 1) % self.config.save_every_n_epochs == 0:
                self.save_checkpoint(epoch + 1)
            
            # Log to wandb
            if self.config.use_wandb:
                wandb.log({
                    'epoch': epoch + 1,
                    'train_loss': train_losses['total'],
                    'val_loss': val_metrics['loss'],
                    'val_f1': val_metrics['f1'],
                    'val_precision': val_metrics['precision'],
                    'val_recall': val_metrics['recall'],
                    'val_auc': val_metrics['auc'],
                    'learning_rate': self.optimizer.param_groups[0]['lr'],
                    **{f'train_{k}': v for k, v in train_losses.items() if k != 'total'}
                })
            
            # Plot progress (optional - for Jupyter notebooks)
            # self.plot_training_progress()
        
        # Training complete
        logger.info("\n" + "=" * 80)
        logger.info("TRAINING COMPLETE!")
        logger.info(f"Total training time: {self.format_time(time.time() - self.start_time)}")
        logger.info(f"Best validation F1: {self.best_f1:.4f}")
        logger.info("=" * 80)
    
    def log_step_metrics(self, epoch, batch_idx, total_batches, loss, losses, lr):
        """Log metrics during training steps"""
        progress = (batch_idx / total_batches) * 100
        
        # Create log message
        log_msg = (
            f"Epoch: {epoch} [{batch_idx}/{total_batches} ({progress:.0f}%)] | "
            f"Loss: {loss:.4f} | "
            f"Fraud: {losses.get('overall_fraud', 0):.4f} | "
            f"LR: {lr:.2e}"
        )
        
        # Only log every N steps to avoid cluttering
        if batch_idx % (self.config.log_every_n_steps * 5) == 0:
            logger.info(log_msg)
    
    def quick_validation(self, train_loader):
        """Quick validation on subset of data"""
        # Use only 10% of validation data for quick check
        subset_size = max(1, len(train_loader) // 10)
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for i, batch in enumerate(train_loader):
                if i >= subset_size:
                    break
                    
                batch = self.move_to_device(batch)
                outputs = self.model(batch)
                
                preds = outputs['fraud_probability'].cpu().numpy()
                labels = batch['labels']['is_fraud'].cpu().numpy()
                
                all_preds.extend(preds)
                all_labels.extend(labels)
        
        # Quick metrics
        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)
        binary_preds = (all_preds > 0.5).astype(int)
        
        _, _, f1, _ = precision_recall_fscore_support(
            all_labels, binary_preds, average='binary'
        )
        
        return {'f1': f1}
    
    def save_checkpoint(self, epoch):
        """Save training checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'val_metrics': self.val_metrics,
            'best_f1': self.best_f1,
            'config': self.config
        }
        
        path = f'checkpoint_epoch_{epoch}.pt'
        torch.save(checkpoint, path)
        logger.info(f"Checkpoint saved: {path}")
    
    def plot_training_progress(self):
        """Plot training progress (for Jupyter notebooks)"""
        if len(self.train_losses) > 1:
            clear_output(wait=True)
            
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
            
            # Loss plot
            epochs = range(1, len(self.train_losses) + 1)
            ax1.plot(epochs, self.train_losses, 'b-', label='Train Loss')
            ax1.plot(epochs, self.val_losses, 'r-', label='Val Loss')
            ax1.set_xlabel('Epoch')
            ax1.set_ylabel('Loss')
            ax1.set_title('Training and Validation Loss')
            ax1.legend()
            ax1.grid(True)
            
            # Metrics plot
            f1_scores = [m['f1'] for m in self.val_metrics]
            ax2.plot(epochs, f1_scores, 'g-', label='F1 Score')
            ax2.set_xlabel('Epoch')
            ax2.set_ylabel('F1 Score')
            ax2.set_title('Validation F1 Score')
            ax2.legend()
            ax2.grid(True)
            
            plt.tight_layout()
            plt.show()
    
    @staticmethod
    def format_time(seconds):
        """Format time in human-readable format"""
        if seconds < 60:
            return f"{seconds:.0f}s"
        elif seconds < 3600:
            return f"{seconds/60:.0f}m {seconds%60:.0f}s"
        else:
            return f"{seconds/3600:.0f}h {(seconds%3600)/60:.0f}m"
    
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
            else:
                moved[k] = v
        return moved
    
    def save_model(self, path):
        """Save model checkpoint"""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'config': self.config,
            'best_f1': self.best_f1
        }, path)

# ============= Inference Pipeline =============

class FraudDetectionInference:
    """Inference pipeline for production"""
    
    def __init__(self, model_path: str, config: ModelConfig, device='cuda'):
        self.device = device
        self.config = config
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(config.model_name)
        
        # Load model
        self.model = TrustSightReviewFraudDetector(config)
        checkpoint = torch.load(model_path, map_location=device, weights_only=False)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(device)
        self.model.eval()

    def extract_single_result(self, outputs, index):
        if isinstance(outputs, torch.Tensor):
            prob = torch.sigmoid(outputs[index]).item()
            label = 1 if prob >= 0.5 else 0
            return {
                "label": label,
                "score": round(prob, 4)
            }
        elif isinstance(outputs, (list, tuple)):
            # Handle list of scores
            prob = outputs[index]
            label = 1 if prob >= 0.5 else 0
            return {
                "label": label,
                "score": round(prob, 4)
            }
        else:
            return {
                "label": -1,
                "score": 0.0,
                "error": "Invalid output format"
            }



    def prepare_batch(self, reviews: List[Dict]) -> Dict:
        """Prepare batch of reviews for inference"""
        input_ids = []
        attention_mask = []
        review_hour = []
        hours_since_delivery = []
        review_velocity = []

        for review_data in reviews:
            text_inputs = self.tokenizer(
                review_data['review_text'],
                review_data.get('title', ''),
                padding='max_length',
                truncation=True,
                max_length=self.config.max_length,
                return_tensors='pt'
            )

            review_time = pd.to_datetime(review_data.get('review_timestamp', review_data.get('review_date')))
            delivery_time = pd.to_datetime(review_data.get('delivery_timestamp', review_time - pd.Timedelta(hours=24)))

            hours_since = (review_time - delivery_time).total_seconds() / 3600

            input_ids.append(text_inputs['input_ids'].squeeze(0))
            attention_mask.append(text_inputs['attention_mask'].squeeze(0))
            review_hour.append(review_time.hour)
            hours_since_delivery.append(hours_since)
            review_velocity.append(review_data.get('reviews_per_day_avg', 0))

        return {
            'input_ids': torch.stack(input_ids).to(self.device),
            'attention_mask': torch.stack(attention_mask).to(self.device),
            'review_hour': torch.tensor(review_hour).float().to(self.device),
            'hours_since_delivery': torch.tensor(hours_since_delivery).float().to(self.device),
            'review_velocity': torch.tensor(review_velocity).float().to(self.device)
        }

        
    def predict_single(self, review_data: Dict) -> Dict:
        """Predict fraud for a single review"""
        # Prepare input
        inputs = self.prepare_input(review_data)
        
        # Run inference
        with torch.no_grad():
            outputs = self.model(inputs)
        
        # Process outputs
        result = {
            'fraud_probability': outputs['fraud_probability'].item(),
            'fraud_signals': {
                'authenticity_issue': outputs['generic_text_pred'].item(),
                'timing_anomaly': outputs['timing_anomaly_pred'].item(),
                'bot_reviewer': outputs['bot_reviewer_pred'].item(),
                'incentivized': outputs['incentivized_pred'].item(),
                'network_fraud': outputs['network_fraud_pred'].item()
            },
            'confidence': self.calculate_confidence(outputs),
            'explanation': self.generate_explanation(outputs)
        }
        
        return result
    
    def predict_batch(self, batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        batch_inputs = self.prepare_batch(batch)

        with torch.no_grad():
            raw_output = self.model(batch_inputs)                 # ✅ custom dict
            outputs = raw_output["fraud_probability"].squeeze(1)  # ✅ final tensor: shape [B]

        results = []
        for j in range(len(batch)):
            result = self.extract_single_result(outputs, j)
            results.append(result)

        avg_score = sum(r["score"] for r in results) / len(results) if results else 0.0
        return {
            "avg_fraud_score": avg_score,
            "individual": results
        }




    
    def prepare_input(self, review_data: Dict) -> Dict:
        """Prepare single review for inference"""
        # Tokenize text
        text_inputs = self.tokenizer(
            review_data['review_text'],
            review_data.get('title', ''),
            padding='max_length',
            truncation=True,
            max_length=self.config.max_length,
            return_tensors='pt'
        )
        
        # Calculate temporal features
        review_time = pd.to_datetime(review_data['review_timestamp'])
        delivery_time = pd.to_datetime(review_data['delivery_timestamp'])
        hours_since = (review_time - delivery_time).total_seconds() / 3600
        
        inputs = {
            'input_ids': text_inputs['input_ids'].to(self.device),
            'attention_mask': text_inputs['attention_mask'].to(self.device),
            'review_hour': torch.tensor([review_time.hour]).float().to(self.device),
            'hours_since_delivery': torch.tensor([hours_since]).float().to(self.device),
            'review_velocity': torch.tensor([review_data.get('reviews_per_day_avg', 0)]).float().to(self.device)
        }
        
        return inputs
    
    def calculate_confidence(self, outputs: Dict) -> float:
        """Calculate confidence score based on signal agreement"""
        signals = [
            outputs['generic_text_pred'].item(),
            outputs['timing_anomaly_pred'].item(),
            outputs['bot_reviewer_pred'].item(),
            outputs['network_fraud_pred'].item()
        ]
        
        # High confidence if multiple signals agree
        agreement = sum(1 for s in signals if s > 0.5)
        fraud_prob = outputs['fraud_probability'].item()
        
        if agreement >= 3 and fraud_prob > 0.8:
            return 0.95
        elif agreement >= 2 and fraud_prob > 0.6:
            return 0.80
        elif agreement >= 1 and fraud_prob > 0.5:
            return 0.65
        else:
            return 0.50
    
    def generate_explanation(self, outputs: Dict) -> str:
        """Generate human-readable explanation"""
        explanations = []
        
        if outputs['generic_text_pred'].item() > 0.5:
            explanations.append("Generic text pattern detected")
        
        if outputs['timing_anomaly_pred'].item() > 0.5:
            explanations.append("Suspicious timing pattern")
        
        if outputs['bot_reviewer_pred'].item() > 0.5:
            explanations.append("Bot-like reviewer behavior")
        
        if outputs['network_fraud_pred'].item() > 0.5:
            explanations.append("Part of fraud network")
        
        if outputs['incentivized_pred'].item() > 0.5:
            explanations.append("Potentially incentivized review")
        
        return "; ".join(explanations) if explanations else "No specific fraud patterns detected"

# ============= Kafka Integration =============

class KafkaFraudDetector:
    """Real-time fraud detection with Kafka"""
    
    def __init__(self, model_inference: FraudDetectionInference):
        self.detector = model_inference
        
    async def process_review_stream(self, kafka_consumer, kafka_producer):
        """Process reviews from Kafka stream"""
        batch = []
        
        async for message in kafka_consumer:
            review = json.loads(message.value)
            batch.append(review)
            
            # Process when batch is full
            if len(batch) >= self.detector.config.batch_size:
                results = self.detector.predict_batch(batch)
                
                # Send results to fraud detection topic
                for review, result in zip(batch, results):
                    if result['fraud_probability'] > 0.5:
                        await self.send_fraud_alert(
                            kafka_producer,
                            review,
                            result
                        )
                
                batch = []
    
    async def send_fraud_alert(self, producer, review, result):
        """Send fraud detection to Kafka"""
        alert = {
            'detection_id': f"FRAUD_{review['review_id']}",
            'timestamp': datetime.now().isoformat(),
            'review_id': review['review_id'],
            'fraud_probability': result['fraud_probability'],
            'confidence': result['confidence'],
            'fraud_signals': result['fraud_signals'],
            'explanation': result['explanation'],
            'action_required': 'INVESTIGATE' if result['confidence'] > 0.8 else 'MONITOR'
        }
        
        await producer.send(
            'fraud-detection-events',
            value=json.dumps(alert).encode()
        )

# ============= Main Execution =============

# ============= Modified Main Execution =============

from sklearn.model_selection import train_test_split
import json
import pandas as pd

import pandas as pd
import json
from sklearn.model_selection import train_test_split

def split_dataset(data_path: str, train_ratio: float = 0.7, val_ratio: float = 0.15):
    """
    Loads nested dataset (with train/validation/test already) and re-splits it into new train/val/test splits.
    Ensures correct formatting and stratification.
    """
    print(f"Loading dataset from {data_path}...")

    with open(data_path, 'r') as f:
        data = json.load(f)

    # Combine all splits into a single flat list
    all_reviews = data["train"] + data["validation"] + data["test"]
    df = pd.DataFrame(all_reviews)
    print(f"Total reviews loaded: {len(df)}")

    # Check fraud distribution
    print(f"Fraud distribution: {df['is_fraud'].value_counts().to_dict()}")

    # Stratified split - first extract test set (15%)
    X = df.drop(columns=['is_fraud'])
    y = df['is_fraud']

    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y,
        test_size=val_ratio,
        stratify=y,
        random_state=42
    )

    # Then split remaining into train and val
    val_size = val_ratio / (train_ratio + val_ratio)

    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp,
        test_size=val_size,
        stratify=y_temp,
        random_state=42
    )

    # Reattach target column
    train_df = X_train.copy()
    train_df['is_fraud'] = y_train

    val_df = X_val.copy()
    val_df['is_fraud'] = y_val

    test_df = X_test.copy()
    test_df['is_fraud'] = y_test

    print(f"\nDataset split:")
    print(f"Train: {len(train_df)} reviews ({len(train_df)/len(df)*100:.1f}%)")
    print(f"Val: {len(val_df)} reviews ({len(val_df)/len(df)*100:.1f}%)")
    print(f"Test: {len(test_df)} reviews ({len(test_df)/len(df)*100:.1f}%)")

    print(f"\nFraud distribution in splits:")
    print(f"Train - Fraud: {train_df['is_fraud'].sum()}, Legitimate: {len(train_df) - train_df['is_fraud'].sum()}")
    print(f"Val - Fraud: {val_df['is_fraud'].sum()}, Legitimate: {len(val_df) - val_df['is_fraud'].sum()}")
    print(f"Test - Fraud: {test_df['is_fraud'].sum()}, Legitimate: {len(test_df) - test_df['is_fraud'].sum()}")

    return train_df, val_df, test_df


def save_splits(train_df, val_df, test_df, output_dir: str = './'):
    """Save the splits to separate JSON files"""
    train_df.to_json(f'{output_dir}/train.json', orient='records', indent=2)
    val_df.to_json(f'{output_dir}/val.json', orient='records', indent=2)
    test_df.to_json(f'{output_dir}/test.json', orient='records', indent=2)
    print(f"\nSplits saved to {output_dir}/")

# ============= Fixed Main Execution =============

def main():
    """Main training script with automatic dataset splitting"""
    
    # Handle command line arguments
    import sys
    if len(sys.argv) > 1 and sys.argv[1].startswith('--dataset'):
        if '=' in sys.argv[1]:
            # Handle --dataset=filename.json format
            COMPLETE_DATASET_PATH = sys.argv[1].split('=')[1]
        elif len(sys.argv) > 2:
            # Handle --dataset filename.json format
            COMPLETE_DATASET_PATH = sys.argv[2]
        else:
            print("Error: Please provide dataset path after --dataset")
            sys.exit(1)
    else:
        # Default path
        COMPLETE_DATASET_PATH = 'reviews_50000.json'
    
    print(f"Using dataset: {COMPLETE_DATASET_PATH}")
    
    # Configuration
    config = ModelConfig()
    
    # Check if we need to split the dataset
    import os
    if not os.path.exists('train.json') or not os.path.exists('val.json') or not os.path.exists('test.json'):
        if not os.path.exists(COMPLETE_DATASET_PATH):
            print(f"Error: Dataset file '{COMPLETE_DATASET_PATH}' not found!")
            print(f"Current directory: {os.getcwd()}")
            print(f"Files in directory: {os.listdir('.')}")
            sys.exit(1)
            
        print("Splitting dataset...")
        train_df, val_df, test_df = split_dataset(COMPLETE_DATASET_PATH)
        save_splits(train_df, val_df, test_df)
    else:
        print("Using existing train/val/test splits...")
    
    # Initialize tokenizer
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    
    # Load datasets
    print("\nLoading datasets for training...")
    train_dataset = ReviewFraudDataset('train.json', tokenizer, config)
    val_dataset = ReviewFraudDataset('val.json', tokenizer, config)
    test_dataset = ReviewFraudDataset('test.json', tokenizer, config)
    
    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Val dataset size: {len(val_dataset)}")
    print(f"Test dataset size: {len(test_dataset)}")
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=4
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=4
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=4
    )
    
    # Initialize model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nUsing device: {device}")
    
    model = TrustSightReviewFraudDetector(config)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Initialize trainer
    trainer = FraudDetectionTrainer(model, config, device=device)
    
    # Train model
    print("\nStarting training...")
    trainer.train(train_loader, val_loader, config.num_epochs)
    
    # Final evaluation on test set
    print("\nEvaluating on test set...")
    test_metrics = trainer.evaluate(test_loader)
    print(f"Test Set Performance:")
    print(f"Precision: {test_metrics['precision']:.4f}")
    print(f"Recall: {test_metrics['recall']:.4f}")
    print(f"F1 Score: {test_metrics['f1']:.4f}")
    print(f"AUC: {test_metrics['auc']:.4f}")
    
    # Save final model
    trainer.save_model('trustsight_review_fraud_final.pt')
    print("\nTraining completed! Model saved as 'trustsight_review_fraud_final.pt'")
    
    # Optional: Generate classification report for different fraud types
    print("\nGenerating detailed performance report...")
    generate_detailed_report(model, test_loader, device)

if __name__ == "__main__":
    main()

def generate_detailed_report(model, test_loader, device):
    """Generate detailed performance metrics for each fraud type"""
    model.eval()
    
    fraud_type_predictions = {
        'generic_text': {'preds': [], 'labels': []},
        'timing_anomaly': {'preds': [], 'labels': []},
        'bot_reviewer': {'preds': [], 'labels': []},
        'incentivized': {'preds': [], 'labels': []},
        'network_fraud': {'preds': [], 'labels': []}
    }
    
    with torch.no_grad():
        for batch in test_loader:
            # Move batch to device
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            if 'labels' in batch:
                batch['labels'] = {k: v.to(device) if isinstance(v, torch.Tensor) else v 
                                  for k, v in batch['labels'].items()}
            
            outputs = model(batch)
            
            # Collect predictions for each fraud type
            for fraud_type in fraud_type_predictions.keys():
                pred_key = f'{fraud_type}_pred'
                if pred_key in outputs:
                    preds = outputs[pred_key].cpu().numpy()
                    labels = batch['labels'][fraud_type].cpu().numpy()
                    
                    fraud_type_predictions[fraud_type]['preds'].extend(preds)
                    fraud_type_predictions[fraud_type]['labels'].extend(labels)
    
    # Print performance for each fraud type
    print("\nPerformance by Fraud Type:")
    print("-" * 60)
    
    for fraud_type, data in fraud_type_predictions.items():
        if len(data['preds']) > 0:
            preds = np.array(data['preds']) > 0.5
            labels = np.array(data['labels'])
            
            precision, recall, f1, _ = precision_recall_fscore_support(
                labels, preds, average='binary', zero_division=0
            )
            
            print(f"\n{fraud_type.upper()}:")
            print(f"  Precision: {precision:.4f}")
            print(f"  Recall: {recall:.4f}")
            print(f"  F1 Score: {f1:.4f}")
            print(f"  Positive samples: {labels.sum()}/{len(labels)}")

# if __name__ == "__main__":
#     # You can also call this directly with your dataset path
#     # Example: python train_model.py --dataset reviews_50000.json
#     import sys
    
#     if len(sys.argv) > 1:
#         dataset_path = sys.argv[1]
#         print(f"Using dataset: {dataset_path}")
#         # Update the path in main()
#         COMPLETE_DATASET_PATH = dataset_path
    
#     main()
