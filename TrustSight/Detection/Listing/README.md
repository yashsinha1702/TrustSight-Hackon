# Listing Fraud Detection System

A comprehensive fraud detection system for Amazon listings that identifies multiple types of fraudulent activities.

## Components

### Core Classes

1. **ListingFraudDetector** (`listing_fraud_detector.py`)
   - Main orchestrating class that coordinates all fraud detection components
   - Provides comprehensive fraud analysis and risk scoring

2. **AmazonCategoryTaxonomy** (`amazon_category_taxonomy.py`)
   - Complete Amazon category taxonomy for accurate product categorization
   - Enables cross-category compatibility checking

3. **ReviewProductMismatchDetector** (`review_product_mismatch_detector.py`)
   - Detects mismatches between product listings and customer reviews
   - Uses semantic analysis and category verification

4. **ListingEvolutionTracker** (`listing_evolution_tracker.py`)
   - Tracks listing changes over time to detect suspicious patterns
   - Identifies brand injection, category hopping, and other evolution anomalies

5. **SEOManipulationDetector** (`seo_manipulation_detector.py`)
   - Detects SEO manipulation, keyword abuse, and deceptive optimization
   - Identifies hidden text, spam patterns, and manipulation tactics

6. **VariationAbuseDetector** (`variation_abuse_detector.py`)
   - Detects product variation abuse including unrelated variations
   - Identifies price manipulation and review pooling abuse

7. **ListingHijackingDetector** (`listing_hijacking_detector.py`)
   - Detects listing hijacking attempts through critical change analysis
   - Monitors seller changes, price crashes, and suspicious patterns

8. **ListingFraudModelTrainer** (`listing_fraud_model_trainer.py`)
   - Training system for all fraud detection models
   - Provides integrated training pipeline

## Installation

Install required dependencies:

```bash
pip install -r requirements.txt
```

## Usage

### Basic Detection

```python
from listing_fraud_detector import ListingFraudDetector

detector = ListingFraudDetector()

# Load trained models (if available)
detector.load_models()

# Analyze a listing
results = detector.detect_listing_fraud(listing_data, reviews_data, historical_data)

print(f"Risk Score: {results['overall_risk']:.1%}")
print(f"Fraud Types: {results['fraud_types_detected']}")
```

### Training Models

```python
from listing_fraud_model_trainer import ListingFraudModelTrainer

trainer = ListingFraudModelTrainer()

# Prepare datasets
datasets = {
    'mismatch_data': mismatch_df,
    'seo_data': seo_df,
    'integrated_data': integrated_df
}

# Train all models
trainer.train_all_models(datasets)
```

## Data Format

### Listing Data
```python
listing = {
    'id': 'product_id',
    'asin': 'B08EXAMPLE',
    'title': 'Product Title',
    'category': 'Electronics',
    'brand': 'Brand Name',
    'price': 29.99,
    'seller_id': 'seller_123',
    'ship_from': 'USA',
    'description': 'Product description',
    'bullet_points': ['Feature 1', 'Feature 2'],
    'variations': [{'name': 'Color: Red', 'price': 29.99}]
}
```

### Review Data
```python
reviews = [
    {
        'id': 'review_id',
        'text': 'Review text content',
        'rating': 5
    }
]
```

### Historical Data
```python
historical = [
    {
        'timestamp': datetime(2024, 1, 1),
        'title': 'Previous title',
        'brand': 'Previous brand',
        'price': 39.99,
        'seller_id': 'previous_seller'
    }
]
```

## Output Format

The detection system returns a comprehensive fraud report:

```python
{
    'listing_id': 'product_id',
    'overall_risk': 0.75,  # Risk score 0-1
    'confidence': 0.85,    # Confidence in assessment
    'fraud_types_detected': ['review_product_mismatch', 'seo_manipulation'],
    'fraud_scores': {
        'mismatch': 0.8,
        'seo_manipulation': 0.6
    },
    'evidence': [...],     # Detailed evidence
    'action_recommended': {
        'severity': 'HIGH',
        'actions': ['FLAG_FOR_URGENT_REVIEW'],
        'automated_actions': ['flag_listing']
    },
    'estimated_impact': 50000,  # Financial impact estimate
    'network_indicators': []    # Network fraud indicators
}
```

## File Structure

```
├── listing_fraud_detector.py
├── amazon_category_taxonomy.py
├── review_product_mismatch_detector.py
├── listing_evolution_tracker.py
├── seo_manipulation_detector.py
├── variation_abuse_detector.py
├── listing_hijacking_detector.py
├── listing_fraud_model_trainer.py
├── requirements.txt
├── usage_example.py
└── README.md
```

## Features

- **Multi-vector fraud detection** across 5+ fraud types
- **Real-time analysis** with pre-trained models
- **Comprehensive evidence collection** for manual review
- **Risk-based action recommendations** with automation support
- **Network fraud pattern detection** for organized attacks
- **Financial impact estimation** for prioritization
- **Extensible architecture** for adding new detection methods

## Integration

This system is designed to be integrated into larger e-commerce fraud prevention platforms. Each detector can be used independently or as part of the complete system.
