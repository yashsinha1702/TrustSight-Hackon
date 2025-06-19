from enum import Enum

class DecisionStatus(Enum):
    """Status values for fraud detection decisions throughout their lifecycle"""
    PENDING_VERIFICATION = "PENDING_VERIFICATION"
    CONFIRMED_FRAUD = "CONFIRMED_FRAUD"
    FALSE_POSITIVE = "FALSE_POSITIVE"
    EXPIRED_NO_DISPUTE = "EXPIRED_NO_DISPUTE"
    UNDER_APPEAL = "UNDER_APPEAL"
    APPEAL_UPHELD = "APPEAL_UPHELD"
    MANUAL_OVERRIDE = "MANUAL_OVERRIDE"

class VerificationMethod(Enum):
    """Methods used to verify fraud detection decisions"""
    SELLER_APPEAL = "SELLER_APPEAL"
    MANUAL_REVIEW = "MANUAL_REVIEW"
    CUSTOMER_REPORTS = "CUSTOMER_REPORTS"
    BEHAVIORAL_VERIFICATION = "BEHAVIORAL_VERIFICATION"
    TIME_BASED = "TIME_BASED"
    AUTOMATED_CHECK = "AUTOMATED_CHECK"

class StorageTier(Enum):
    """Storage tiers for decision data lifecycle management"""
    HOT = "hot"
    WARM = "warm"
    COLD = "cold"