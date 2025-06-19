from enum import Enum

class SignalType(Enum):
    """Types of fraud signals that can trigger network investigations"""
    FAKE_REVIEW = "fake_review"
    COUNTERFEIT_PRODUCT = "counterfeit_product"
    SUSPICIOUS_SELLER = "suspicious_seller"
    LISTING_FRAUD = "listing_fraud"
    PRICE_ANOMALY = "price_anomaly"
    NETWORK_PATTERN = "network_pattern"

class NetworkType(Enum):
    """Types of fraud networks that can be detected and classified"""
    REVIEW_FARM = "review_farm"
    SELLER_CARTEL = "seller_cartel"
    COUNTERFEIT_RING = "counterfeit_ring"
    HYBRID_OPERATION = "hybrid_operation"
    COMPETITOR_ATTACK = "competitor_attack"
    EXIT_SCAM_NETWORK = "exit_scam_network"