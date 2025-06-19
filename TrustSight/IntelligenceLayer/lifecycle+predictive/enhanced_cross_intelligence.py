from dataclasses import asdict
from typing import Dict, List, Any
from .lifecycle_monitoring_engine import LifecycleMonitoringEngine
from .predictive_fraud_engine import PredictiveFraudEngine
from .models import FraudPrediction

class EnhancedCrossIntelligence:
    """Enhanced Cross Intelligence with Lifecycle and Predictive capabilities for comprehensive fraud analysis"""
    
    def __init__(self, existing_cross_intelligence=None):
        self.lifecycle_engine = LifecycleMonitoringEngine()
        self.predictive_engine = PredictiveFraudEngine()
        self.cross_intel = existing_cross_intelligence
        
    async def analyze_with_intelligence(
        self,
        entity_type: str,
        entity_id: str,
        detection_result: Dict[str, Any]
    ) -> Dict[str, Any]:
        lifecycle_profile = await self.lifecycle_engine.track_entity(
            entity_type,
            entity_id,
            {
                "type": "detection_event",
                "fraud_score": detection_result.get("fraud_score", 0),
                "detection_type": detection_result.get("fraud_type", "unknown")
            }
        )
        
        lifecycle_insights = await self.lifecycle_engine.get_lifecycle_insights(
            entity_type,
            entity_id
        )
        
        predictions = await self.predictive_engine.predict_fraud(
            entity_type,
            entity_id,
            detection_result,
            lifecycle_profile
        )
        
        enhanced_result = {
            "original_detection": detection_result,
            "lifecycle_analysis": lifecycle_insights,
            "fraud_predictions": [asdict(p) for p in predictions],
            "risk_score": max(
                detection_result.get("fraud_score", 0),
                lifecycle_profile.risk_score,
                max([p.probability for p in predictions]) if predictions else 0
            ),
            "recommended_actions": self._combine_recommendations(
                detection_result,
                lifecycle_insights,
                predictions
            )
        }
        
        return enhanced_result
    
    def _combine_recommendations(
        self,
        detection: Dict,
        lifecycle: Dict,
        predictions: List[FraudPrediction]
    ) -> List[str]:
        recommendations = []
        
        if detection.get("fraud_score", 0) > 0.7:
            recommendations.append("Immediate action required - high fraud score")
        
        recommendations.extend(lifecycle.get("recommendations", []))
        
        for prediction in predictions:
            if prediction.probability > 0.8:
                recommendations.extend(prediction.recommended_actions)
        
        seen = set()
        unique_recommendations = []
        for rec in recommendations:
            if rec not in seen:
                seen.add(rec)
                unique_recommendations.append(rec)
        
        return unique_recommendations