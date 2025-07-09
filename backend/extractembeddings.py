"""
Quick script to extract artifacts from your already trained seller network model
Run this to get feature_extractor.pkl and graph_embeddings.pkl without retraining
"""

import torch
import numpy as np
import json
import joblib
import pickle
from sklearn.preprocessing import StandardScaler
import os

def quick_extract_seller_artifacts():
    """Extract artifacts from your trained model and data"""
    
    print("=" * 60)
    print("EXTRACTING SELLER NETWORK ARTIFACTS")
    print("=" * 60)
    
    # Paths
    model_path = r'C:\Users\syash\Desktop\trustsightnotclean\seller_network_best_f1_0.5707.pt'
    data_path = 'sellers_10000_fixed.json'
    
    # Check files exist
    if not os.path.exists(data_path):
        print(f"ERROR: {data_path} not found!")
        return
    
    print(f"\n1. Loading data from {data_path}...")
    with open(data_path, 'r') as f:
        sellers_data = json.load(f)
    
    if isinstance(sellers_data, dict) and 'sellers' in sellers_data:
        sellers = sellers_data['sellers']
    else:
        sellers = sellers_data
    
    print(f"   Loaded {len(sellers)} sellers")
    
    # 2. Extract features and create scaler
    print("\n2. Extracting features and creating scaler...")
    
    feature_names = [
        "total_products_normalized", "avg_product_price_normalized", 
        "avg_rating_normalized", "response_time_normalized",
        "fulfillment_rate", "return_rate", "customer_complaints_normalized",
        "price_change_frequency_normalized", "max_price_drop_normalized",
        "synchronized_changes_normalized", "competitor_price_matching",
        "dynamic_pricing_detected", "avg_stock_level_normalized",
        "max_stock_spike_normalized", "inventory_turnover_normalized",
        "shared_product_count_normalized", "address_similarity_score",
        "customer_overlap_score", "reviewer_overlap_score",
        "account_age_normalized", "registration_hour_normalized"
    ]
    
    # Extract features using the same logic as training
    all_features = []
    fraud_sellers = []
    
    for seller in sellers:
        # Extract features (matching the training code exactly)
        features = []
        
        # Seller metrics
        metrics = seller.get('seller_metrics', {})
        features.extend([
            metrics.get('total_products', 0) / 1000,
            metrics.get('avg_product_price', 0) / 100,
            metrics.get('avg_rating', 0) / 5,
            metrics.get('response_time_hours', 0) / 24,
            metrics.get('fulfillment_rate', 0),
            metrics.get('return_rate', 0),
            metrics.get('customer_complaints', 0) / 100
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
        
        all_features.append(features)
        
        # Track fraud sellers
        if seller.get('labels', {}).get('is_fraud', False):
            fraud_sellers.append(seller)
    
    # Create feature matrix
    X = np.array(all_features)
    print(f"   Feature matrix shape: {X.shape}")
    
    # Fit scaler
    scaler = StandardScaler()
    scaler.fit(X)
    
    # 3. Create and save feature extractor
    print("\n3. Creating feature extractor...")
    
    feature_extractor = {
        'scaler': scaler,
        'feature_names': feature_names,
        'feature_count': 21,
        'normalization_factors': {
            'total_products': 1000,
            'avg_product_price': 100,
            'avg_rating': 5,
            'response_time_hours': 24,
            'customer_complaints': 100,
            'avg_price_change_frequency_days': 30,
            'max_price_drop_percent': 100,
            'synchronized_changes_count': 50,
            'avg_stock_level': 1000,
            'max_stock_spike': 10000,
            'inventory_turnover_days': 30,
            'shared_product_count': 100,
            'account_age_days': 365,
            'registration_hour': 24
        },
        'feature_stats': {
            'mean': scaler.mean_.tolist(),
            'std': scaler.scale_.tolist(),
            'min': X.min(axis=0).tolist(),
            'max': X.max(axis=0).tolist()
        },
        'n_samples_trained': len(X)
    }
    
    os.makedirs('models/seller', exist_ok=True)
    joblib.dump(feature_extractor, 'models/seller/feature_extractor.pkl')
    print("   ✓ Saved to models/seller/feature_extractor.pkl")
    
    # 4. Create graph embeddings
    print("\n4. Creating graph embeddings...")
    
    # Group fraud sellers by pattern
    pattern_features = {
        'price_coordination': [],
        'inventory_sharing': [],
        'registration_cluster': [],
        'exit_scam': []
    }
    
    for seller in fraud_sellers[:100]:  # Use first 100 fraud sellers
        seller_idx = sellers.index(seller)
        features = all_features[seller_idx]
        
        patterns = seller.get('labels', {}).get('specific_patterns', {})
        if patterns.get('has_price_coordination'):
            pattern_features['price_coordination'].append(features)
        if patterns.get('has_inventory_sharing'):
            pattern_features['inventory_sharing'].append(features)
        if patterns.get('has_registration_cluster'):
            pattern_features['registration_cluster'].append(features)
        if patterns.get('exit_scam_risk'):
            pattern_features['exit_scam'].append(features)
    
    # Create pattern signatures
    network_signatures = {}
    for pattern_name, features_list in pattern_features.items():
        if features_list:
            # Create embedding by expanding features to 128 dims
            pattern_features_avg = np.mean(features_list, axis=0)
            # Tile to create 128-dim embedding
            pattern_embedding = np.tile(pattern_features_avg, 128 // 21 + 1)[:128]
            network_signatures[f'{pattern_name}_pattern'] = pattern_embedding
            print(f"   Created {pattern_name} signature from {len(features_list)} examples")
    
    # Create embeddings for some fraud sellers
    embeddings = {}
    for i, seller in enumerate(fraud_sellers[:50]):
        seller_idx = sellers.index(seller)
        features = all_features[seller_idx]
        # Create 128-dim embedding
        embedding = np.tile(features, 128 // 21 + 1)[:128]
        embeddings[seller['seller_id']] = embedding
    
    graph_embeddings = {
        'embedding_dim': 128,
        'embeddings': embeddings,
        'network_signatures': network_signatures,
        'similarity_threshold': 0.8,
        'created_from': 'training_data',
        'n_fraud_examples': len(fraud_sellers)
    }
    
    with open('models/seller/graph_embeddings.pkl', 'wb') as f:
        pickle.dump(graph_embeddings, f)
    print("   ✓ Saved to models/seller/graph_embeddings.pkl")
    
    # 5. Print summary
    print("\n" + "=" * 60)
    print("EXTRACTION COMPLETE!")
    print("=" * 60)
    print("\nGenerated files:")
    print("  - models/seller/feature_extractor.pkl")
    print("  - models/seller/graph_embeddings.pkl")
    print("\nModel paths for integration.py:")
    print("'seller_network': {")
    print(f"    'gcn_model': r'{model_path}',")
    print("    'feature_extractor': 'models/seller/feature_extractor.pkl',")
    print("    'graph_embeddings': 'models/seller/graph_embeddings.pkl'")
    print("}")

if __name__ == "__main__":
    quick_extract_seller_artifacts()