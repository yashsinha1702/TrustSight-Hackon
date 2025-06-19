import torch
from torch.utils.data import Dataset
import pandas as pd
from config import ModelConfig

class ReviewFraudDataset(Dataset):
    """Dataset class for loading and preprocessing review fraud detection data"""
    
    def __init__(self, data_path: str, tokenizer, config: ModelConfig):
        self.tokenizer = tokenizer
        self.config = config
        self.data = self.load_data(data_path)
        
    def load_data(self, data_path: str) -> pd.DataFrame:
        if data_path.endswith('.json'):
            data = pd.read_json(data_path)
        else:
            data = pd.read_csv(data_path)
        
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
        
        text_inputs = self.tokenizer(
            row['review_text'],
            row.get('title', ''),
            padding='max_length',
            truncation=True,
            max_length=self.config.max_length,
            return_tensors='pt'
        )
        
        features = {
            'input_ids': text_inputs['input_ids'].squeeze(),
            'attention_mask': text_inputs['attention_mask'].squeeze(),
            'review_hour': torch.tensor(row.get('review_hour', 0), dtype=torch.float32),
            'hours_since_delivery': torch.tensor(row.get('hours_since_delivery', 0), dtype=torch.float32),
            'review_velocity': torch.tensor(row.get('reviews_per_day_avg', 0), dtype=torch.float32),
            'reviewer_account_age': torch.tensor(row.get('reviewer_account_age_days', 0), dtype=torch.float32),
            'reviewer_total_reviews': torch.tensor(row.get('reviewer_total_reviews', 0), dtype=torch.float32),
            'verified_purchase_rate': torch.tensor(row.get('reviewer_verified_purchases_pct', 0), dtype=torch.float32),
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