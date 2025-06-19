import torch
import torch.nn as nn
import numpy as np
import time
import logging
from tqdm import tqdm
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score
from config import ModelConfig
from loss import MultiTaskLoss

logger = logging.getLogger(__name__)

class FraudDetectionTrainer:
    """Training manager for the fraud detection model with comprehensive monitoring and evaluation"""
    
    def __init__(self, model, config: ModelConfig, device='cuda'):
        self.model = model.to(device)
        self.config = config
        self.device = device
        
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=0.01
        )
        
        self.scheduler = None
        self.criterion = MultiTaskLoss(config)
        
        self.train_losses = []
        self.val_losses = []
        self.val_metrics = []
        self.best_f1 = 0
        self.start_time = None
        
        if config.use_wandb:
            import wandb
            wandb.init(
                project="trustsight-fraud-detection",
                config=config.__dict__
            )
            wandb.watch(model)
    
    def train_epoch(self, dataloader, epoch_num, total_epochs):
        self.model.train()
        epoch_losses = []
        task_losses = {task: [] for task in self.config.task_weights.keys()}
        
        pbar = tqdm(dataloader, desc=f'Epoch {epoch_num}/{total_epochs}', 
                    unit='batch', leave=True)
        
        for batch_idx, batch in enumerate(pbar):
            batch = self.move_to_device(batch)
            
            outputs = self.model(batch)
            loss, losses = self.criterion(outputs, batch['labels'])
            
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            
            if self.scheduler:
                self.scheduler.step()
            
            epoch_losses.append(loss.item())
            for task, task_loss in losses.items():
                task_losses[task].append(task_loss.item())
            
            current_lr = self.optimizer.param_groups[0]['lr']
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'lr': f'{current_lr:.2e}',
                'fraud_loss': f'{losses.get("overall_fraud", 0):.4f}'
            })
            
            if batch_idx % self.config.log_every_n_steps == 0:
                self.log_step_metrics(
                    epoch_num, batch_idx, len(dataloader), 
                    loss.item(), losses, current_lr
                )
            
            if batch_idx % self.config.validate_every_n_steps == 0 and batch_idx > 0:
                self.model.eval()
                val_metrics = self.quick_validation(dataloader)
                self.model.train()
                logger.info(f"Mid-epoch validation - F1: {val_metrics['f1']:.4f}")
        
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
        
        pbar = tqdm(dataloader, desc=desc, unit='batch', leave=False)
        
        with torch.no_grad():
            for batch in pbar:
                batch = self.move_to_device(batch)
                outputs = self.model(batch)
                
                loss, _ = self.criterion(outputs, batch['labels'])
                total_loss += loss.item()
                num_batches += 1
                
                preds = outputs['fraud_probability'].cpu().numpy()
                labels = batch['labels']['is_fraud'].cpu().numpy()
                
                all_preds.extend(preds)
                all_labels.extend(labels)
                
                pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)
        binary_preds = (all_preds > 0.5).astype(int)
        
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
        total_steps = len(train_loader) * num_epochs
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=total_steps
        )
        
        self.start_time = time.time()
        
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
        
        for epoch in range(num_epochs):
            epoch_start_time = time.time()
            
            logger.info(f"\n{'='*50}")
            logger.info(f"EPOCH {epoch + 1}/{num_epochs}")
            logger.info(f"{'='*50}")
            
            train_losses = self.train_epoch(train_loader, epoch + 1, num_epochs)
            self.train_losses.append(train_losses['total'])
            
            logger.info("\nRunning validation...")
            val_metrics = self.evaluate(val_loader, desc=f"Validation Epoch {epoch + 1}")
            self.val_losses.append(val_metrics['loss'])
            self.val_metrics.append(val_metrics)
            
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
            
            logger.info("\nTask-specific losses:")
            for task, loss in train_losses.items():
                if task != 'total':
                    logger.info(f"  {task}: {loss:.4f}")
            
            if val_metrics['f1'] >= self.best_f1:
                self.best_f1 = val_metrics['f1']
                self.save_model(f'best_model_f1_{self.best_f1:.4f}.pt')
                logger.info(f"  ✓ New best F1 score: {self.best_f1:.4f}")
            
            if (epoch + 1) % self.config.save_every_n_epochs == 0:
                self.save_checkpoint(epoch + 1)
            
            if self.config.use_wandb:
                import wandb
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
        
        logger.info("\n" + "=" * 80)
        logger.info("TRAINING COMPLETE!")
        logger.info(f"Total training time: {self.format_time(time.time() - self.start_time)}")
        logger.info(f"Best validation F1: {self.best_f1:.4f}")
        logger.info("=" * 80)
    
    def log_step_metrics(self, epoch, batch_idx, total_batches, loss, losses, lr):
        progress = (batch_idx / total_batches) * 100
        log_msg = (
            f"Epoch: {epoch} [{batch_idx}/{total_batches} ({progress:.0f}%)] | "
            f"Loss: {loss:.4f} | "
            f"Fraud: {losses.get('overall_fraud', 0):.4f} | "
            f"LR: {lr:.2e}"
        )
        
        if batch_idx % (self.config.log_every_n_steps * 5) == 0:
            logger.info(log_msg)
    
    def quick_validation(self, train_loader):
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
        
        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)
        binary_preds = (all_preds > 0.5).astype(int)
        
        _, _, f1, _ = precision_recall_fscore_support(
            all_labels, binary_preds, average='binary'
        )
        
        return {'f1': f1}
    
    def save_checkpoint(self, epoch):
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
    
    @staticmethod
    def format_time(seconds):
        if seconds < 60:
            return f"{seconds:.0f}s"
        elif seconds < 3600:
            return f"{seconds/60:.0f}m {seconds%60:.0f}s"
        else:
            return f"{seconds/3600:.0f}h {(seconds%3600)/60:.0f}m"
    
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
            else:
                moved[k] = v
        return moved
    
    def save_model(self, path):
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'config': self.config,
            'best_f1': self.best_f1
        }, path)