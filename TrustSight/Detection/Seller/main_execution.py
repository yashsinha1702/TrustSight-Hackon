import torch
import logging
from torch.utils.data import DataLoader

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SellerNetworkPipeline:
    """Main pipeline class for training and deploying seller network detection models"""
    
    def __init__(self, data_path: str, config_params: dict = None):
        from config import SellerNetworkConfig
        
        self.data_path = data_path
        self.config = SellerNetworkConfig(**config_params) if config_params else SellerNetworkConfig()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
    def prepare_data(self):
        from utilities import validate_and_fix_seller_data, analyze_dataset_distribution, split_seller_dataset
        
        validate_and_fix_seller_data(self.data_path, 'sellers_fixed.json')
        analyze_dataset_distribution('sellers_fixed.json')
        train_df, val_df, test_df = split_seller_dataset('sellers_fixed.json')
        
        return train_df, val_df, test_df
    
    def create_dataloaders(self):
        from dataset import SellerNetworkDataset
        from utilities import seller_collate_fn
        
        train_dataset = SellerNetworkDataset('train_sellers.json', self.config)
        val_dataset = SellerNetworkDataset('val_sellers.json', self.config)
        
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            collate_fn=seller_collate_fn,
            num_workers=4
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            collate_fn=seller_collate_fn,
            num_workers=4
        )
        
        return train_loader, val_loader
    
    def train_model(self, train_loader, val_loader):
        from main_model import SellerNetworkDetectionModel
        from trainer import SellerNetworkTrainer
        
        model = SellerNetworkDetectionModel(self.config)
        trainer = SellerNetworkTrainer(model, self.config, self.device)
        
        trainer.train(train_loader, val_loader, self.config.num_epochs)
        
        return model, trainer
    
    def load_trained_model(self, model_path: str):
        from interface import SellerNetworkInterface
        
        interface = SellerNetworkInterface(model_path, self.config, self.device)
        return interface
    
    def run_full_pipeline(self):
        logger.info("Starting Seller Network Detection Pipeline")
        
        self.prepare_data()
        train_loader, val_loader = self.create_dataloaders()
        model, trainer = self.train_model(train_loader, val_loader)
        
        logger.info("Pipeline completed successfully")
        return model, trainer