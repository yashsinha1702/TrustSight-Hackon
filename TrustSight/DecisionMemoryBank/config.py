class MemoryBankConfig:
    """Configuration settings for Decision Memory Bank operations and storage"""
    
    DYNAMODB_TABLE = "trustsight-decisions"
    S3_BUCKET = "trustsight-evidence"
    GLACIER_VAULT = "trustsight-archive"
    
    DATABASE_URL = "sqlite:///./decision_memory_bank.db"
    
    RETENTION_POLICIES = {
        'confirmed_fraud': {
            'hot': 30,
            'warm': 730,
            'cold': 2555
        },
        'false_positive': {
            'hot': 90,
            'warm': -1,
            'cold': -1
        },
        'pending_verification': {
            'hot': 30,
            'warm': 0,
            'cold': 0
        },
        'low_confidence': {
            'hot': 7,
            'warm': 0,
            'cold': 0
        }
    }
    
    TTL_HIGH_CONFIDENCE = 7
    TTL_MEDIUM_CONFIDENCE = 14
    TTL_LOW_CONFIDENCE = 30
    
    VERIFICATION_WEIGHTS = {
        'seller_appeal': 0.9,
        'manual_review': 1.0,
        'customer_reports': 0.8,
        'behavioral_verification': 0.7,
        'time_based': 0.6
    }
    
    VERIFICATION_THRESHOLD = 0.75
    APPEAL_WINDOW = 48
    
    MIN_VERIFICATIONS_FOR_TRAINING = 1000
    TRAINING_CONFIDENCE_THRESHOLD = 0.85
    MODEL_VERSION_RETENTION = 10
    
    API_HOST = "0.0.0.0"
    API_PORT = 8001