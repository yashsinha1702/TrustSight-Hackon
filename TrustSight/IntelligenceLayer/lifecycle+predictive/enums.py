from enum import Enum

class LifecycleStage(Enum):
    """Product and seller lifecycle stages for monitoring progression"""
    LISTING_CREATED = "listing_created"
    FIRST_SALE = "first_sale"
    GROWTH_PHASE = "growth_phase"
    MATURE_PHASE = "mature_phase"
    DECLINE_PHASE = "decline_phase"
    DELISTED = "delisted"
    
    ACCOUNT_CREATED = "account_created"
    FIRST_LISTING = "first_listing"
    ESTABLISHED = "established"
    HIGH_VOLUME = "high_volume"
    DORMANT = "dormant"
    SUSPENDED = "suspended"

class FraudPattern(Enum):
    """Types of predictable fraud patterns for forecasting"""
    REVIEW_BOMB = "review_bomb"
    EXIT_SCAM = "exit_scam"
    SEASONAL_FRAUD = "seasonal_fraud"
    PRICE_MANIPULATION = "price_manipulation"
    INVENTORY_DUMP = "inventory_dump"
    ACCOUNT_TAKEOVER = "account_takeover"
    RETURN_FRAUD = "return_fraud"

class PredictionConfidence(Enum):
    """Prediction confidence levels for fraud forecasting"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    VERY_HIGH = "very_high"