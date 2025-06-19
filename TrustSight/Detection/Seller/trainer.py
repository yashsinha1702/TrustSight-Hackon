import torch
import torch.nn as nn
import numpy as np
import logging
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score
from tqdm import tqdm

logger = logging.getLogger(__name__)

class SellerNetworkTrainer:
    """Training class for seller network detection model with comprehensive evaluation and monitoring"""
    
    def __init__(self, model, config, device='cuda'):
        self.model = model.to(device)
        self.config = config
        self.device = device
        
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=1e-4,
            weight_decay=0.01
        )
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, 
            mode='max', 
            factor=0.5, 
            patience=3,
            verbose=True
        )
        
        self.warmup_steps = 50
        self.current_step = 0
        
        from loss_function import SellerNetworkLoss
        self.criterion = SellerNetworkLoss(config)
        self.best_f1 = 0
        
    def train_epoch(self, dataloader, epoch_num, total_epochs):
        self.model.train()
        epoch_losses = []
        predictions_tracker = []
        labels_tracker = []
        
        pbar = tqdm(dataloader, desc=f'Epoch {epoch_num}/{total_epochs}')
        
        for batch_idx, batch in enumerate(pbar):
            if self.current_step < self.warmup_steps:
                lr = self.config.learning_rate * (self.current_step / self.warmup_steps)
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] = lr
            self.current_step += 1
            
            batch = self.move_to_device(batch)
            outputs = self.model(batch)
            
            with torch.no_grad():
                pred_probs = torch.sigmoid(outputs['fraud_probability']).cpu().numpy()
                predictions_tracker.extend(pred_probs)
                labels_tracker.extend(batch['labels']['is_fraud'].cpu().numpy())
            
            loss, losses = self.criterion(outputs, batch['labels'])
            
            if torch.isnan(loss):
                logger.warning("NaN loss detected, skipping batch")
                continue
            
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            
            epoch_losses.append(loss.item())
            
            current_lr = self.optimizer.param_groups[0]['lr']
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'network': f'{losses.get("network_detection", 0):.4f}',
                'lr': f'{current_lr:.2e}',
                'pred_mean': f'{np.mean(predictions_tracker[-100:]):.3f}'
            })
        
        logger.info(f"Epoch {epoch_num} prediction stats - Min: {np.min(predictions_tracker):.4f}, "
                    f"Max: {np.max(predictions_tracker):.4f}, Mean: {np.mean(predictions_tracker):.4f}")
        
        return np.mean(epoch_losses)
    
    def evaluate(self, dataloader, desc="Validation"):
        self.model.eval()
        all_preds = []
        all_labels = []
        total_loss = 0
        num_batches = 0
        
        pbar = tqdm(dataloader, desc=desc, unit='batch', leave=False)
        
        with torch.no_grad():
            for batch in pbar:
                batch = self.move_to_device(batch)
                outputs = self.model(batch)
                
                loss, _ = self.criterion(outputs, batch['labels'])
                total_loss += loss.item()
                num_batches += 1
                
                preds = torch.sigmoid(outputs['fraud_probability']).cpu().numpy()
                labels = batch['labels']['is_fraud'].cpu().numpy()
                
                all_preds.extend(preds)
                all_labels.extend(labels)
                
                pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        all_preds = np.array(all_preds).flatten()
        all_labels = np.array(all_labels).flatten()
        
        thresholds = [0.5, 0.4, 0.3, 0.2, 0.1]
        best_f1 = 0
        best_threshold = 0.5
        
        for thresh in thresholds:
            binary_preds = (all_preds > thresh).astype(int)
            
            if binary_preds.sum() > 0 and binary_preds.sum() < len(binary_preds):
                try:
                    precision, recall, f1, _ = precision_recall_fscore_support(
                        all_labels, binary_preds, average='binary', zero_division=0
                    )
                    
                    if f1 > best_f1:
                        best_f1 = f1
                        best_threshold = thresh
                except:
                    pass
        
        binary_preds = (all_preds > 0.5).astype(int)
        
        if len(np.unique(binary_preds)) == 1:
            if binary_preds[0] == 0:
                precision = 0.0
                recall = 0.0
                f1 = 0.0
            else:
                true_positives = (all_labels == 1).sum()
                false_positives = (all_labels == 0).sum()
                precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
                recall = 1.0 if true_positives > 0 else 0
                f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        else:
            try:
                precision, recall, f1, _ = precision_recall_fscore_support(
                    all_labels, binary_preds, average='binary', zero_division=0
                )
            except Exception as e:
                precision = recall = f1 = 0.0
        
        try:
            if len(np.unique(all_labels)) > 1:
                auc = roc_auc_score(all_labels, all_preds)
            else:
                auc = 0.5
        except Exception as e:
            auc = 0.5
        
        avg_loss = total_loss / num_batches if num_batches > 0 else 0
        
        true_positives = ((binary_preds == 1) & (all_labels == 1)).sum()
        false_positives = ((binary_preds == 1) & (all_labels == 0)).sum()
        true_negatives = ((binary_preds == 0) & (all_labels == 0)).sum()
        false_negatives = ((binary_preds == 0) & (all_labels == 1)).sum()
        
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
                moved[k] = v.to(self.device)
            else:
                moved[k] = v
                
        return moved
    
    def train(self, train_loader, val_loader, num_epochs):
        logger.info("Starting Seller Network Detection Training")
        
        for epoch in range(num_epochs):
            train_loss = self.train_epoch(train_loader, epoch + 1, num_epochs)
            val_metrics = self.evaluate(val_loader)
            self.scheduler.step(val_metrics['f1'])
            
            logger.info(f"\nEpoch {epoch + 1} Results:")
            logger.info(f"Train Loss: {train_loss:.4f}")
            logger.info(f"Val Precision: {val_metrics['precision']:.4f}")
            logger.info(f"Val Recall: {val_metrics['recall']:.4f}")
            logger.info(f"Val F1: {val_metrics['f1']:.4f}")
            logger.info(f"Val AUC: {val_metrics['auc']:.4f}")
            
            if val_metrics['f1'] > self.best_f1:
                self.best_f1 = val_metrics['f1']
                self.save_model(f'seller_network_best_f1_{self.best_f1:.4f}.pt')
    
    def save_model(self, path):
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'config': self.config,
            'best_f1': self.best_f1
        }, path)