from datetime import datetime
from typing import Dict, Any
from data_classes import TrustScore

class TrustScoreCalculator:
    """
    Calculates multi-dimensional trust scores based on fraud signals and risk indicators
    """
    
    def __init__(self):
        self.weights = {
            "review_authenticity": 0.25,
            "product_legitimacy": 0.25,
            "seller_reliability": 0.20,
            "pricing_fairness": 0.15,
            "network_isolation": 0.15
        }
        
    def calculate_trust_score(
        self,
        entity_type: str,
        entity_id: str,
        fraud_signals: Dict[str, Any]
    ) -> TrustScore:
        trust_score = TrustScore()
        
        if "review_fraud_score" in fraud_signals:
            trust_score.dimensions["review_authenticity"] = max(
                0, 100 * (1 - fraud_signals["review_fraud_score"])
            )
        
        if "counterfeit_score" in fraud_signals:
            trust_score.dimensions["product_legitimacy"] = max(
                0, 100 * (1 - fraud_signals["counterfeit_score"])
            )
        
        if "seller_fraud_score" in fraud_signals:
            trust_score.dimensions["seller_reliability"] = max(
                0, 100 * (1 - fraud_signals["seller_fraud_score"])
            )
        
        if "price_anomaly_score" in fraud_signals:
            trust_score.dimensions["pricing_fairness"] = max(
                0, 100 * (1 - fraud_signals["price_anomaly_score"])
            )
        
        if "network_connection_score" in fraud_signals:
            trust_score.dimensions["network_isolation"] = max(
                0, 100 * (1 - fraud_signals["network_connection_score"])
            )
        
        trust_score.overall_score = sum(
            score * self.weights[dimension]
            for dimension, score in trust_score.dimensions.items()
        )
        
        trust_score.history.append({
            "timestamp": datetime.now().isoformat(),
            "overall_score": trust_score.overall_score,
            "dimensions": trust_score.dimensions.copy(),
            "fraud_signals": fraud_signals
        })
        
        return trust_score
    
    def get_risk_level(self, trust_score: float) -> str:
        if trust_score >= 80:
            return "low_risk"
        elif trust_score >= 60:
            return "medium_risk"
        elif trust_score >= 40:
            return "high_risk"
        else:
            return "critical_risk"