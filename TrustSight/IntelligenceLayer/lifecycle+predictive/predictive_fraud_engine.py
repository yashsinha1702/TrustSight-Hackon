import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from .enums import FraudPattern, PredictionConfidence, LifecycleStage
from .models import FraudPrediction, SeasonalPattern, LifecycleProfile

logger = logging.getLogger(__name__)

class PredictiveFraudEngine:
    """Predicts future fraud patterns based on historical data and current signals"""
    
    def __init__(self):
        self.prediction_models = self._initialize_models()
        self.seasonal_patterns = self._load_seasonal_patterns()
        self.prediction_history: List[FraudPrediction] = []
        
    def _initialize_models(self) -> Dict:
        return {
            FraudPattern.REVIEW_BOMB: self._create_review_bomb_predictor(),
            FraudPattern.EXIT_SCAM: self._create_exit_scam_predictor(),
            FraudPattern.SEASONAL_FRAUD: self._create_seasonal_predictor(),
            FraudPattern.PRICE_MANIPULATION: self._create_price_predictor()
        }
    
    def _create_review_bomb_predictor(self):
        return {
            "features": [
                "competitor_activity",
                "price_changes",
                "review_velocity",
                "product_ranking",
                "category_competition"
            ],
            "threshold": 0.7
        }
    
    def _create_exit_scam_predictor(self):
        return {
            "features": [
                "account_age",
                "sudden_discount",
                "inventory_increase",
                "shipping_time_increase",
                "customer_complaints"
            ],
            "threshold": 0.8
        }
    
    def _create_seasonal_predictor(self):
        return {
            "features": [
                "days_to_event",
                "category_risk",
                "price_point",
                "seller_history",
                "market_demand"
            ],
            "threshold": 0.6
        }
    
    def _create_price_predictor(self):
        return {
            "features": [
                "competitor_prices",
                "inventory_level",
                "market_volatility",
                "seller_margin",
                "demand_forecast"
            ],
            "threshold": 0.65
        }
    
    def _load_seasonal_patterns(self) -> List[SeasonalPattern]:
        return [
            SeasonalPattern(
                pattern_name="Black Friday Surge",
                season="black_friday",
                fraud_multiplier=4.5,
                common_tactics=["fake_discounts", "counterfeit_electronics", "review_flooding"],
                high_risk_categories=["Electronics", "Luxury", "Toys"]
            ),
            SeasonalPattern(
                pattern_name="Holiday Rush",
                season="holiday_season",
                fraud_multiplier=3.2,
                common_tactics=["gift_card_scams", "shipping_fraud", "fake_urgency"],
                high_risk_categories=["Gifts", "Electronics", "Jewelry"]
            ),
            SeasonalPattern(
                pattern_name="Prime Day Frenzy",
                season="prime_day",
                fraud_multiplier=5.0,
                common_tactics=["listing_hijacking", "fake_prime_deals", "bait_switch"],
                high_risk_categories=["All"]
            )
        ]
    
    async def predict_fraud(
        self,
        entity_type: str,
        entity_id: str,
        current_signals: Dict[str, Any],
        lifecycle_profile: Optional[LifecycleProfile] = None
    ) -> List[FraudPrediction]:
        predictions = []
        
        for pattern, model in self.prediction_models.items():
            probability = await self._calculate_fraud_probability(
                pattern,
                model,
                entity_type,
                entity_id,
                current_signals,
                lifecycle_profile
            )
            
            if probability > model["threshold"]:
                prediction = FraudPrediction(
                    prediction_id=f"pred_{datetime.now().timestamp()}",
                    entity_type=entity_type,
                    entity_id=entity_id,
                    fraud_pattern=pattern,
                    probability=probability,
                    confidence=self._calculate_confidence(probability, model["threshold"]),
                    predicted_timeframe=self._estimate_timeframe(pattern),
                    risk_factors=self._identify_risk_factors(pattern, current_signals),
                    recommended_actions=self._recommend_preventive_actions(pattern, probability)
                )
                predictions.append(prediction)
                self.prediction_history.append(prediction)
        
        return predictions
    
    async def _calculate_fraud_probability(
        self,
        pattern: FraudPattern,
        model: Dict,
        entity_type: str,
        entity_id: str,
        signals: Dict[str, Any],
        lifecycle: Optional[LifecycleProfile]
    ) -> float:
        probability = 0.0
        
        if pattern == FraudPattern.REVIEW_BOMB:
            if signals.get("competitor_new_product", False):
                probability += 0.2
            if signals.get("recent_price_drop", 0) > 20:
                probability += 0.15
            if signals.get("review_velocity_change", 0) > 200:
                probability += 0.25
            if signals.get("negative_review_ratio", 0) > 0.3:
                probability += 0.2
            if lifecycle and lifecycle.current_stage == LifecycleStage.GROWTH_PHASE:
                probability += 0.1
                
        elif pattern == FraudPattern.EXIT_SCAM:
            if lifecycle:
                if lifecycle.metrics.get("age_days", 0) < 180:
                    probability += 0.2
                if any(a["type"] == "rapid_sales_growth" for a in lifecycle.anomalies):
                    probability += 0.3
            if signals.get("discount_percentage", 0) > 70:
                probability += 0.25
            if signals.get("inventory_spike", 0) > 500:
                probability += 0.15
            if signals.get("shipping_time_increase", 0) > 5:
                probability += 0.1
                
        elif pattern == FraudPattern.SEASONAL_FRAUD:
            days_to_black_friday = self._days_to_event("black_friday")
            if days_to_black_friday < 30:
                probability += 0.3 * (30 - days_to_black_friday) / 30
            
            if signals.get("category") in ["Electronics", "Luxury"]:
                probability += 0.2
            
            if signals.get("recent_listing_increase", 0) > 50:
                probability += 0.15
            if signals.get("new_seller", False):
                probability += 0.15
                
        elif pattern == FraudPattern.PRICE_MANIPULATION:
            if signals.get("price_volatility", 0) > 30:
                probability += 0.3
            if signals.get("competitor_price_war", False):
                probability += 0.2
            if signals.get("inventory_low", False):
                probability += 0.15
            if signals.get("demand_spike", 0) > 200:
                probability += 0.2
        
        return min(probability, 1.0)
    
    def _calculate_confidence(self, probability: float, threshold: float) -> PredictionConfidence:
        confidence_score = probability / threshold
        
        if confidence_score > 1.5:
            return PredictionConfidence.VERY_HIGH
        elif confidence_score > 1.2:
            return PredictionConfidence.HIGH
        elif confidence_score > 1.0:
            return PredictionConfidence.MEDIUM
        else:
            return PredictionConfidence.LOW
    
    def _estimate_timeframe(self, pattern: FraudPattern) -> str:
        timeframes = {
            FraudPattern.REVIEW_BOMB: "next_48_hours",
            FraudPattern.EXIT_SCAM: "next_7_days",
            FraudPattern.SEASONAL_FRAUD: "next_14_days",
            FraudPattern.PRICE_MANIPULATION: "next_24_hours"
        }
        return timeframes.get(pattern, "next_7_days")
    
    def _identify_risk_factors(
        self,
        pattern: FraudPattern,
        signals: Dict[str, Any]
    ) -> List[str]:
        risk_factors = []
        
        if pattern == FraudPattern.REVIEW_BOMB:
            if signals.get("competitor_new_product"):
                risk_factors.append("New competitor product launched")
            if signals.get("recent_price_drop", 0) > 20:
                risk_factors.append(f"Recent {signals['recent_price_drop']}% price drop")
                
        elif pattern == FraudPattern.EXIT_SCAM:
            if signals.get("discount_percentage", 0) > 70:
                risk_factors.append(f"Extreme discount: {signals['discount_percentage']}%")
            if signals.get("inventory_spike", 0) > 500:
                risk_factors.append("Sudden inventory increase")
                
        return risk_factors
    
    def _recommend_preventive_actions(
        self,
        pattern: FraudPattern,
        probability: float
    ) -> List[str]:
        actions = []
        
        if pattern == FraudPattern.REVIEW_BOMB:
            actions.extend([
                "Enable real-time review monitoring",
                "Set velocity limits on review acceptance",
                "Prepare response templates"
            ])
            if probability > 0.8:
                actions.append("Pre-emptively flag for manual review")
                
        elif pattern == FraudPattern.EXIT_SCAM:
            actions.extend([
                "Hold payments for verification",
                "Monitor shipping confirmations",
                "Enable buyer protection warnings"
            ])
            if probability > 0.85:
                actions.append("Consider preventive suspension")
                
        elif pattern == FraudPattern.SEASONAL_FRAUD:
            actions.extend([
                "Increase monitoring frequency",
                "Lower auto-approval thresholds",
                "Deploy additional ML models"
            ])
            
        elif pattern == FraudPattern.PRICE_MANIPULATION:
            actions.extend([
                "Lock pricing for stability",
                "Monitor competitor actions",
                "Alert pricing team"
            ])
        
        return actions
    
    def _days_to_event(self, event: str) -> int:
        event_dates = {
            "black_friday": datetime(2024, 11, 29),
            "prime_day": datetime(2024, 7, 15),
            "holiday_season": datetime(2024, 12, 15)
        }
        
        target_date = event_dates.get(event, datetime.now() + timedelta(days=365))
        days_until = (target_date - datetime.now()).days
        
        return max(0, days_until)
    
    async def validate_predictions(
        self,
        prediction_id: str,
        actual_outcome: bool
    ):
        prediction = next((p for p in self.prediction_history if p.prediction_id == prediction_id), None)
        
        if prediction:
            validation = {
                "prediction_id": prediction_id,
                "predicted_pattern": prediction.fraud_pattern.value,
                "predicted_probability": prediction.probability,
                "actual_outcome": actual_outcome,
                "timestamp": datetime.now()
            }
            
            logger.info(f"Prediction validation: {validation}")
            
            if actual_outcome:
                logger.info(f"Correct prediction for {prediction.fraud_pattern.value}")
            else:
                logger.info(f"False positive for {prediction.fraud_pattern.value}")