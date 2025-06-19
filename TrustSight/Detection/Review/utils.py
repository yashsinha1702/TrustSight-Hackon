import pandas as pd
import json
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_fscore_support

class DatasetSplitter:
    """Utility class for splitting and managing fraud detection datasets"""
    
    @staticmethod
    def split_dataset(data_path: str, train_ratio: float = 0.7, val_ratio: float = 0.15):
        print(f"Loading dataset from {data_path}...")

        with open(data_path, 'r') as f:
            data = json.load(f)

        all_reviews = data["train"] + data["validation"] + data["test"]
        df = pd.DataFrame(all_reviews)
        print(f"Total reviews loaded: {len(df)}")

        print(f"Fraud distribution: {df['is_fraud'].value_counts().to_dict()}")

        X = df.drop(columns=['is_fraud'])
        y = df['is_fraud']

        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y,
            test_size=val_ratio,
            stratify=y,
            random_state=42
        )

        val_size = val_ratio / (train_ratio + val_ratio)

        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp,
            test_size=val_size,
            stratify=y_temp,
            random_state=42
        )

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

    @staticmethod
    def save_splits(train_df, val_df, test_df, output_dir: str = './'):
        train_df.to_json(f'{output_dir}/train.json', orient='records', indent=2)
        val_df.to_json(f'{output_dir}/val.json', orient='records', indent=2)
        test_df.to_json(f'{output_dir}/test.json', orient='records', indent=2)
        print(f"\nSplits saved to {output_dir}/")

class ModelEvaluator:
    """Utility class for generating detailed performance reports and metrics"""
    
    @staticmethod
    def generate_detailed_report(model, test_loader, device):
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
                batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
                if 'labels' in batch:
                    batch['labels'] = {k: v.to(device) if isinstance(v, torch.Tensor) else v 
                                      for k, v in batch['labels'].items()}
                
                outputs = model(batch)
                
                for fraud_type in fraud_type_predictions.keys():
                    pred_key = f'{fraud_type}_pred'
                    if pred_key in outputs:
                        preds = outputs[pred_key].cpu().numpy()
                        labels = batch['labels'][fraud_type].cpu().numpy()
                        
                        fraud_type_predictions[fraud_type]['preds'].extend(preds)
                        fraud_type_predictions[fraud_type]['labels'].extend(labels)
        
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