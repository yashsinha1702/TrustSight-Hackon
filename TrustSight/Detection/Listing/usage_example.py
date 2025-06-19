from listing_fraud_detector import ListingFraudDetector
from listing_fraud_model_trainer import ListingFraudModelTrainer
import pandas as pd
from datetime import datetime

def basic_usage_example():
    """Basic example of how to use the listing fraud detection system."""
    
    detector = ListingFraudDetector()
    
    test_listing = {
        'id': 'PROD_001',
        'asin': 'B08EXAMPLE',
        'title': 'Wireless Bluetooth Headphones',
        'category': 'Electronics',
        'brand': 'SampleBrand',
        'price': 59.99,
        'seller_id': 'SELLER_123',
        'ship_from': 'USA',
        'description': 'High quality wireless headphones with noise cancellation',
        'bullet_points': [
            'Wireless Bluetooth 5.0',
            'Noise cancellation technology',
            'Long battery life'
        ]
    }
    
    test_reviews = [
        {
            'id': 'REV001',
            'text': 'Great headphones! Sound quality is excellent.',
            'rating': 5
        }
    ]
    
    historical_data = [
        {
            'timestamp': datetime(2024, 1, 1),
            'title': 'Wireless Bluetooth Headphones',
            'brand': 'SampleBrand',
            'price': 59.99,
            'seller_id': 'SELLER_123',
            'ship_from': 'USA'
        }
    ]
    
    results = detector.detect_listing_fraud(test_listing, test_reviews, historical_data)
    
    print(f"Overall Risk: {results['overall_risk']:.1%}")
    print(f"Fraud Types Detected: {results['fraud_types_detected']}")
    print(f"Recommended Action: {results['action_recommended']['severity']}")
    
    return results

def training_example():
    """Example of how to train the models."""
    
    sample_mismatch_data = pd.DataFrame({
        'listing_title': ['Product A', 'Product B'],
        'listing_category': ['Electronics', 'Clothing'],
        'listing_brand': ['Brand X', 'Brand Y'],
        'listing_description': ['Electronic device', 'Clothing item'],
        'review_text': ['Great product!', 'Nice clothes'],
        'is_mismatch': [0, 0]
    })
    
    sample_seo_data = pd.DataFrame({
        'title': ['Normal Title', 'Keyword Stuffed Title'],
        'description': ['Normal description', 'Stuffed keywords'],
        'bullet_points': ['Normal points', 'More keywords'],
        'backend_keywords': ['normal', 'excessive keywords'],
        'has_seo_manipulation': [0, 1]
    })
    
    sample_integrated_data = pd.DataFrame({
        'feature_mismatch_score': [0.1, 0.8],
        'feature_evolution_score': [0.2, 0.9],
        'feature_seo_score': [0.0, 0.7],
        'feature_variation_score': [0.1, 0.6],
        'feature_hijack_score': [0.0, 0.8],
        'feature_review_count': [100, 50],
        'feature_price_variance': [0.1, 0.8],
        'feature_seller_age': [365, 30],
        'feature_category_competitiveness': [0.5, 0.9],
        'overall_fraud_label': ['legitimate', 'fraudulent']
    })
    
    datasets = {
        'mismatch_data': sample_mismatch_data,
        'seo_data': sample_seo_data,
        'integrated_data': sample_integrated_data
    }
    
    trainer = ListingFraudModelTrainer()
    trainer.train_all_models(datasets)

if __name__ == "__main__":
    print("Running basic fraud detection example...")
    basic_usage_example()
    
    print("\nRunning training example...")
    training_example()
