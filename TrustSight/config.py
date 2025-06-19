class IntegrationConfig:
    """Central configuration for Integration Layer components and processing"""
    
    KAFKA_BOOTSTRAP_SERVERS = ['localhost:9092']
    KAFKA_TOPICS = {
        'input': {
            'product-events': 'trustsight.products',
            'review-events': 'trustsight.reviews',
            'seller-events': 'trustsight.sellers',
            'listing-events': 'trustsight.listings',
            'price-events': 'trustsight.prices'
        },
        'output': {
            'fraud-detections': 'trustsight.fraud.detected',
            'trust-scores': 'trustsight.trust.scores',
            'alerts': 'trustsight.alerts',
            'actions': 'trustsight.actions',
            'decisions': 'trustsight.decisions'
        }
    }
    
    REDIS_HOST = 'localhost'
    REDIS_PORT = 6379
    REDIS_DB = 0
    CACHE_TTL = 3600
    
    MODEL_PATHS = {
        'review_fraud': 'models/review_fraud_model.pt',
        'seller_network': 'models/seller_network_model.pt',
        'listing_fraud': 'models/listing_fraud_model.pkl',
        'counterfeit': 'models/counterfeit_model.pt'
    }
    
    MAX_PARALLEL_DETECTIONS = 10
    BATCH_SIZE = 100
    PRIORITY_QUEUE_SIZE = 10000
    
    HIGH_PRIORITY_THRESHOLD = 0.8
    MEDIUM_PRIORITY_THRESHOLD = 0.5
    CROSS_INTEL_THRESHOLD = 0.7
    
    API_HOST = "0.0.0.0"
    API_PORT = 8000