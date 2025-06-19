import torch
import json
import pandas as pd
import numpy as np
from torch_geometric.data import Batch
from sklearn.model_selection import train_test_split

def seller_collate_fn(batch):
    """Custom collate function for batching seller data with graph structures"""
    
    seller_ids = [item['seller_id'] for item in batch]
    seller_features = torch.stack([item['seller_features'] for item in batch])
    
    graphs = [item['graph'] for item in batch]
    batched_graph = Batch.from_data_list(graphs)
    
    registration_hours = torch.tensor([item['registration_hour'] for item in batch], dtype=torch.float32)
    days_active = torch.tensor([item['days_active'] for item in batch], dtype=torch.float32)
    total_products = torch.tensor([item['total_products'] for item in batch], dtype=torch.float32)
    
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
        'batch': batched_graph.batch
    }

def validate_and_fix_seller_data(data_path: str, output_path: str = None):
    """Validate and fix inconsistencies in seller dataset"""
    
    with open(data_path, 'r') as f:
        if isinstance(data_path, str) and data_path.endswith('.json'):
            sellers = json.load(f)
        else:
            sellers = data_path
    
    fixed_count = 0
    
    for seller in sellers:
        metrics = seller['seller_metrics']
        if metrics['active_products'] > metrics['total_products']:
            metrics['active_products'] = int(metrics['total_products'] * 0.8)
            fixed_count += 1
        
        network = seller['network_features']
        if network['shared_products_with_sellers'] is None:
            network['shared_products_with_sellers'] = []
        if network['same_day_registrations'] is None:
            network['same_day_registrations'] = []
        
        for key in ['shared_product_count', 'address_similarity_score', 
                    'customer_overlap_score', 'reviewer_overlap_score']:
            if network[key] is None:
                network[key] = 0
        
        if seller['temporal_features']['activity_gaps'] is None:
            seller['temporal_features']['activity_gaps'] = []
        if seller['temporal_features']['peak_activity_hours'] is None:
            seller['temporal_features']['peak_activity_hours'] = []
    
    if output_path:
        with open(output_path, 'w') as f:
            json.dump(sellers, f, indent=2)
    
    return sellers

def analyze_dataset_distribution(data_path: str):
    """Analyze and display dataset statistics and distributions"""
    
    with open(data_path, 'r') as f:
        sellers = json.load(f)
    
    df = pd.DataFrame(sellers)
    
    fraud_counts = df['labels'].apply(lambda x: x['is_fraud']).value_counts()
    
    patterns = ['is_network_member', 'has_price_coordination', 'has_inventory_sharing', 
                'has_registration_cluster', 'exit_scam_risk']
    
    network_sellers = df[df['network_id'].notna()]
    
    stats = {
        'total_sellers': len(df),
        'fraud_distribution': dict(fraud_counts),
        'pattern_counts': {pattern: df['labels'].apply(lambda x: x['specific_patterns'][pattern]).sum() 
                          for pattern in patterns},
        'network_stats': {
            'sellers_in_networks': len(network_sellers),
            'unique_networks': df['network_id'].nunique() - 1,
            'avg_sellers_per_network': network_sellers['connected_seller_count'].mean()
        }
    }
    
    return stats

def split_seller_dataset(data_path: str, train_ratio: float = 0.7, val_ratio: float = 0.15):
    """Split seller dataset while maintaining network integrity"""
    
    with open(data_path, 'r') as f:
        sellers = json.load(f)
    
    df = pd.DataFrame(sellers)
    
    network_sellers = df[df['network_id'].notna()].copy()
    non_network_sellers = df[df['network_id'].isna()].copy()
    
    if len(network_sellers) > 0:
        unique_networks = network_sellers['network_id'].unique()
        
        train_networks, temp_networks = train_test_split(
            unique_networks, test_size=(1-train_ratio), random_state=42
        )
        val_networks, test_networks = train_test_split(
            temp_networks, test_size=0.5, random_state=42
        )
        
        train_network_sellers = network_sellers[network_sellers['network_id'].isin(train_networks)]
        val_network_sellers = network_sellers[network_sellers['network_id'].isin(val_networks)]
        test_network_sellers = network_sellers[network_sellers['network_id'].isin(test_networks)]
    else:
        train_network_sellers = pd.DataFrame()
        val_network_sellers = pd.DataFrame()
        test_network_sellers = pd.DataFrame()
    
    if len(non_network_sellers) > 0:
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
    
    train_df = pd.concat([train_network_sellers, train_non_network], ignore_index=True)
    val_df = pd.concat([val_network_sellers, val_non_network], ignore_index=True)
    test_df = pd.concat([test_network_sellers, test_non_network], ignore_index=True)
    
    train_df = train_df.sample(frac=1, random_state=42).reset_index(drop=True)
    val_df = val_df.sample(frac=1, random_state=42).reset_index(drop=True)
    test_df = test_df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    train_df.to_json('train_sellers.json', orient='records', indent=2)
    val_df.to_json('val_sellers.json', orient='records', indent=2)
    test_df.to_json('test_sellers.json', orient='records', indent=2)
    
    return train_df, val_df, test_df