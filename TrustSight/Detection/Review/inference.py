import torch
import pandas as pd
from typing import Dict, List
from transformers import AutoTokenizer
from config import ModelConfig
from model import TrustSightReviewFraudDetector

class FraudDetectionInference:
    """Production inference pipeline for real-time fraud detection on review data"""
    
    def __init__(self, model_path: str, config: ModelConfig, device='cuda'):
        self.device = device
        self.config = config
        
        self.tokenizer = AutoTokenizer.from_pretrained(config.model_name)
        
        self.model = TrustSightReviewFraudDetector(config)
        checkpoint = torch.load(model_path, map_location=device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(device)
        self.model.eval()
        
    def predict_single(self, review_data: Dict) -> Dict:
        inputs = self.prepare_input(review_data)
        
        with torch.no_grad():
            outputs = self.model(inputs)
        
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
    
    def predict_batch(self, reviews: List[Dict]) -> List[Dict]:
        results = []
        
        for i in range(0, len(reviews), self.config.batch_size):
            batch = reviews[i:i + self.config.batch_size]
            batch_inputs = self.prepare_batch(batch)
            
            with torch.no_grad():
                outputs = self.model(batch_inputs)
            
            for j in range(len(batch)):
                result = self.extract_single_result(outputs, j)
                results.append(result)
        
        return results
    
    def prepare_input(self, review_data: Dict) -> Dict:
        text_inputs = self.tokenizer(
            review_data['review_text'],
            review_data.get('title', ''),
            padding='max_length',
            truncation=True,
            max_length=self.config.max_length,
            return_tensors='pt'
        )
        
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
        signals = [
            outputs['generic_text_pred'].item(),
            outputs['timing_anomaly_pred'].item(),
            outputs['bot_reviewer_pred'].item(),
            outputs['network_fraud_pred'].item()
        ]
        
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