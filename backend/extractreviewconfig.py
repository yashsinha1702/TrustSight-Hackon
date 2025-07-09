import torch
import json
import pickle
import joblib
import numpy as np
from transformers import AutoTokenizer
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
import os
from datetime import datetime

# ============= REVIEW FRAUD MODEL FILES =============

def generate_review_fraud_files(model_path='C:\\Users\\syash\\Desktop\\trustsightnotclean\\best_model_f1_1.0000.pt'):
    """Generate tokenizer and config files for review fraud model"""
    
    print("=== Generating Review Fraud Model Files ===\n")
    
    # 1. Save tokenizer
    print("1. Saving tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained("roberta-base")
    
    # Save tokenizer using pickle (for compatibility with your code)
    tokenizer_path = 'models/review/tokenizer.pkl'
    os.makedirs(os.path.dirname(tokenizer_path), exist_ok=True)
    
    with open(tokenizer_path, 'wb') as f:
        pickle.dump(tokenizer, f)
    print(f"✓ Tokenizer saved to {tokenizer_path}")
    
    # 2. Create and save model config
    print("\n2. Saving model config...")
    model_config = {
        "model_name": "roberta-base",
        "hidden_size": 768,
        "num_fraud_types": 6,
        "max_length": 256,
        "dropout_rate": 0.1,
        "learning_rate": 2e-5,
        "batch_size": 32,
        "num_epochs": 10,
        "task_weights": {
            "overall_fraud": 2.0,
            "generic_text": 0.8,
            "timing_anomaly": 1.0,
            "bot_reviewer": 1.2,
            "incentivized": 0.7,
            "network_fraud": 1.5
        },
        "created_at": datetime.now().isoformat(),
        "model_version": "1.0",
        "model_path": model_path
    }
    
    config_path = 'models/review/model_config.json'
    with open(config_path, 'w') as f:
        json.dump(model_config, f, indent=2)
    print(f"✓ Config saved to {config_path}")
    
    # 3. Update model paths in integration config
    print("\n3. Model paths for integration.py:")
    print(f"    'model': r'{model_path}',")
    print(f"    'tokenizer': '{tokenizer_path}',")
    print(f"    'config': '{config_path}'")
    
    return tokenizer_path, config_path

if __name__ == "__main__":
    try:
        generate_review_fraud_files()
    except Exception as e:
        print(f"❌ Error: {e}")
