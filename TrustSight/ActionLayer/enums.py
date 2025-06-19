from enum import Enum

class ActionType(Enum):
    WARNING_BANNER = "warning_banner"
    REVIEW_HOLD = "review_hold"
    LISTING_SUPPRESSION = "listing_suppression"
    PAYMENT_DELAY = "payment_delay"
    INTERNAL_ALERT = "internal_alert"
    TRUST_SCORE_REDUCTION = "trust_score_reduction"
    ACCOUNT_SUSPENSION = "account_suspension"
    LISTING_TAKEDOWN = "listing_takedown"
    AUTO_REFUND = "auto_refund"
    PERMANENT_BAN = "permanent_ban"
    PAYMENT_FREEZE = "payment_freeze"
    NETWORK_SHUTDOWN = "network_shutdown"

class ActionPhase(Enum):
    SOFT = "soft"
    HARD = "hard"

class AlertPriority(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

class AlertRecipient(Enum):
    BUYER = "buyer"
    SELLER = "seller"
    INTERNAL_TEAM = "internal_team"
    BRAND_OWNER = "brand_owner"
    LEGAL_TEAM = "legal_team"