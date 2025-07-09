"""
TRUSTSIGHT INTELLIGENCE ENGINES
Lifecycle Monitoring Engine - Tracks products/sellers from creation to removal
Predictive Fraud Engine - Forecasts fraud patterns before they occur
"""

import asyncio
import json
import logging
from collections import defaultdict
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Optional, Set, Tuple, Any
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class LifecyclePredictiveInterface:
    """
    Provides an inference wrapper for EnhancedCrossIntelligence
    """
    def __init__(self):
        self.enhanced_engine = EnhancedCrossIntelligence()

    async def infer(self, entity_type: str, entity_id: str, detection_result: dict):
        """
        Runs enhanced lifecycle + predictive fraud analysis
        """
        result = await self.enhanced_engine.analyze_with_intelligence(
            entity_type, entity_id, detection_result
        )
        return result


# ============= Enums =============

class LifecycleStage(Enum):
    """Product/Seller lifecycle stages"""
    # Product stages
    LISTING_CREATED = "listing_created"
    FIRST_SALE = "first_sale"
    GROWTH_PHASE = "growth_phase"
    MATURE_PHASE = "mature_phase"
    DECLINE_PHASE = "decline_phase"
    DELISTED = "delisted"
    
    # Seller stages
    ACCOUNT_CREATED = "account_created"
    FIRST_LISTING = "first_listing"
    ESTABLISHED = "established"
    HIGH_VOLUME = "high_volume"
    DORMANT = "dormant"
    SUSPENDED = "suspended"

class FraudPattern(Enum):
    """Types of predictable fraud patterns"""
    REVIEW_BOMB = "review_bomb"
    EXIT_SCAM = "exit_scam"
    SEASONAL_FRAUD = "seasonal_fraud"
    PRICE_MANIPULATION = "price_manipulation"
    INVENTORY_DUMP = "inventory_dump"
    ACCOUNT_TAKEOVER = "account_takeover"
    RETURN_FRAUD = "return_fraud"

class PredictionConfidence(Enum):
    """Prediction confidence levels"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    VERY_HIGH = "very_high"

# ============= Data Classes =============

@dataclass
class LifecycleEvent:
    """Event in entity lifecycle"""
    event_id: str
    entity_type: str  # product, seller
    entity_id: str
    event_type: str
    timestamp: datetime
    metadata: Dict[str, Any] = field(default_factory=dict)
    anomaly_score: float = 0.0

@dataclass
class LifecycleProfile:
    """Complete lifecycle profile of an entity"""
    entity_type: str
    entity_id: str
    current_stage: LifecycleStage
    creation_date: datetime
    events: List[LifecycleEvent] = field(default_factory=list)
    metrics: Dict[str, Any] = field(default_factory=dict)
    anomalies: List[Dict] = field(default_factory=list)
    risk_score: float = 0.0
    last_updated: datetime = field(default_factory=datetime.now)

@dataclass
class FraudPrediction:
    """Prediction of future fraud"""
    prediction_id: str
    entity_type: str
    entity_id: str
    fraud_pattern: FraudPattern
    probability: float
    confidence: PredictionConfidence
    predicted_timeframe: str  # e.g., "next_7_days"
    risk_factors: List[str] = field(default_factory=list)
    recommended_actions: List[str] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.now)

@dataclass
class SeasonalPattern:
    """Seasonal fraud pattern"""
    pattern_name: str
    season: str  # "black_friday", "prime_day", "holiday_season"
    fraud_multiplier: float
    common_tactics: List[str] = field(default_factory=list)
    high_risk_categories: List[str] = field(default_factory=list)

# ============= Lifecycle Monitoring Engine =============

class LifecycleMonitoringEngine:
    """
    Tracks entities throughout their lifecycle to identify abuse patterns
    """
    
    def __init__(self):
        self.lifecycle_data: Dict[str, LifecycleProfile] = {}
        self.stage_transitions = self._define_stage_transitions()
        self.anomaly_thresholds = self._define_anomaly_thresholds()
        
    def _define_stage_transitions(self) -> Dict:
        """Define normal stage transition patterns"""
        return {
            "product": {
                LifecycleStage.LISTING_CREATED: [LifecycleStage.FIRST_SALE],
                LifecycleStage.FIRST_SALE: [LifecycleStage.GROWTH_PHASE],
                LifecycleStage.GROWTH_PHASE: [LifecycleStage.MATURE_PHASE],
                LifecycleStage.MATURE_PHASE: [LifecycleStage.DECLINE_PHASE],
                LifecycleStage.DECLINE_PHASE: [LifecycleStage.DELISTED]
            },
            "seller": {
                LifecycleStage.ACCOUNT_CREATED: [LifecycleStage.FIRST_LISTING],
                LifecycleStage.FIRST_LISTING: [LifecycleStage.ESTABLISHED],
                LifecycleStage.ESTABLISHED: [LifecycleStage.HIGH_VOLUME, LifecycleStage.DORMANT],
                LifecycleStage.HIGH_VOLUME: [LifecycleStage.DORMANT, LifecycleStage.SUSPENDED],
                LifecycleStage.DORMANT: [LifecycleStage.ESTABLISHED, LifecycleStage.SUSPENDED]
            }
        }
    
    def _define_anomaly_thresholds(self) -> Dict:
        """Define what constitutes anomalous behavior at each stage"""
        return {
            "rapid_growth": {
                "sales_spike": 500,  # 500% increase
                "review_spike": 1000,  # 1000% increase
                "timeframe": 7  # days
            },
            "sudden_dormancy": {
                "activity_drop": 90,  # 90% decrease
                "timeframe": 3  # days
            },
            "price_volatility": {
                "max_change": 50,  # 50% price change
                "frequency": 3  # times per week
            }
        }
    
    async def track_entity(
        self,
        entity_type: str,
        entity_id: str,
        event: Dict[str, Any]
    ) -> LifecycleProfile:
        """
        Track entity lifecycle event
        """
        # Get or create profile
        profile_key = f"{entity_type}_{entity_id}"
        if profile_key not in self.lifecycle_data:
            self.lifecycle_data[profile_key] = LifecycleProfile(
                entity_type=entity_type,
                entity_id=entity_id,
                current_stage=LifecycleStage.LISTING_CREATED if entity_type == "product" else LifecycleStage.ACCOUNT_CREATED,
                creation_date=datetime.now()
            )
        
        profile = self.lifecycle_data[profile_key]
        
        # Create lifecycle event
        lifecycle_event = LifecycleEvent(
            event_id=f"evt_{datetime.now().timestamp()}",
            entity_type=entity_type,
            entity_id=entity_id,
            event_type=event.get("type", "unknown"),
            timestamp=datetime.now(),
            metadata=event
        )
        
        # Check for anomalies
        anomaly_score = await self._detect_lifecycle_anomalies(profile, lifecycle_event)
        lifecycle_event.anomaly_score = anomaly_score
        
        # Add event to profile
        profile.events.append(lifecycle_event)
        
        # Update metrics
        await self._update_lifecycle_metrics(profile)
        
        # Check for stage transition
        new_stage = await self._check_stage_transition(profile)
        if new_stage != profile.current_stage:
            profile.current_stage = new_stage
            logger.info(f"Entity {entity_id} transitioned to {new_stage.value}")
        
        # Calculate risk score
        profile.risk_score = await self._calculate_lifecycle_risk(profile)
        profile.last_updated = datetime.now()
        
        return profile
    
    async def _detect_lifecycle_anomalies(
        self,
        profile: LifecycleProfile,
        event: LifecycleEvent
    ) -> float:
        """
        Detect anomalies in lifecycle patterns
        """
        anomaly_score = 0.0
        
        # Check for rapid growth
        if event.event_type == "sales_update":
            recent_sales = [e for e in profile.events[-10:] if e.event_type == "sales_update"]
            if len(recent_sales) >= 2:
                old_sales = recent_sales[0].metadata.get("daily_sales", 0)
                new_sales = event.metadata.get("daily_sales", 0)
                if old_sales > 0:
                    growth_rate = (new_sales - old_sales) / old_sales * 100
                    if growth_rate > self.anomaly_thresholds["rapid_growth"]["sales_spike"]:
                        anomaly_score += 0.3
                        profile.anomalies.append({
                            "type": "rapid_sales_growth",
                            "growth_rate": growth_rate,
                            "timestamp": datetime.now()
                        })
        
        # Check for sudden review influx
        if event.event_type == "review_added":
            recent_reviews = [e for e in profile.events[-50:] if e.event_type == "review_added"]
            daily_reviews = defaultdict(int)
            for review in recent_reviews:
                day = review.timestamp.date()
                daily_reviews[day] += 1
            
            if len(daily_reviews) >= 2:
                avg_daily = sum(daily_reviews.values()) / len(daily_reviews)
                today_count = daily_reviews[datetime.now().date()]
                if avg_daily > 0 and today_count / avg_daily > 10:
                    anomaly_score += 0.4
                    profile.anomalies.append({
                        "type": "review_spike",
                        "spike_factor": today_count / avg_daily,
                        "timestamp": datetime.now()
                    })
        
        # Check for price manipulation
        if event.event_type == "price_change":
            recent_prices = [e for e in profile.events[-20:] if e.event_type == "price_change"]
            if len(recent_prices) >= 3:
                price_changes = []
                for i in range(1, len(recent_prices)):
                    old_price = recent_prices[i-1].metadata.get("price", 0)
                    new_price = recent_prices[i].metadata.get("price", 0)
                    if old_price > 0:
                        change = abs(new_price - old_price) / old_price * 100
                        price_changes.append(change)
                
                if len([c for c in price_changes if c > self.anomaly_thresholds["price_volatility"]["max_change"]]) >= 3:
                    anomaly_score += 0.5
                    profile.anomalies.append({
                        "type": "price_manipulation",
                        "volatility": np.mean(price_changes),
                        "timestamp": datetime.now()
                    })
        
        return min(anomaly_score, 1.0)
    
    async def _update_lifecycle_metrics(self, profile: LifecycleProfile):
        """
        Update lifecycle metrics for the entity
        """
        # Calculate age
        age_days = (datetime.now() - profile.creation_date).days
        profile.metrics["age_days"] = age_days
        
        # Calculate activity metrics
        events_last_7d = [e for e in profile.events if (datetime.now() - e.timestamp).days <= 7]
        events_last_30d = [e for e in profile.events if (datetime.now() - e.timestamp).days <= 30]
        
        profile.metrics["events_last_7d"] = len(events_last_7d)
        profile.metrics["events_last_30d"] = len(events_last_30d)
        
        # Calculate stage duration
        if profile.current_stage:
            stage_events = [e for e in profile.events if e.metadata.get("stage") == profile.current_stage.value]
            if stage_events:
                stage_duration = (datetime.now() - stage_events[0].timestamp).days
                profile.metrics[f"{profile.current_stage.value}_duration"] = stage_duration
        
        # Calculate anomaly rate
        total_anomalies = len(profile.anomalies)
        profile.metrics["anomaly_rate"] = total_anomalies / max(len(profile.events), 1)
    
    async def _check_stage_transition(self, profile: LifecycleProfile) -> LifecycleStage:
        """
        Check if entity should transition to a new lifecycle stage
        """
        current_stage = profile.current_stage
        
        # Product stage transitions
        if profile.entity_type == "product":
            if current_stage == LifecycleStage.LISTING_CREATED:
                # Check for first sale
                if any(e.event_type == "sale" for e in profile.events):
                    return LifecycleStage.FIRST_SALE
            
            elif current_stage == LifecycleStage.FIRST_SALE:
                # Check for growth (>10 sales in 7 days)
                recent_sales = [e for e in profile.events if e.event_type == "sale" and (datetime.now() - e.timestamp).days <= 7]
                if len(recent_sales) > 10:
                    return LifecycleStage.GROWTH_PHASE
            
            elif current_stage == LifecycleStage.GROWTH_PHASE:
                # Check for maturity (stable sales for 30 days)
                if profile.metrics.get("age_days", 0) > 90:
                    return LifecycleStage.MATURE_PHASE
            
            elif current_stage == LifecycleStage.MATURE_PHASE:
                # Check for decline (50% drop in activity)
                if profile.metrics.get("events_last_7d", 0) < profile.metrics.get("events_last_30d", 0) / 4 * 0.5:
                    return LifecycleStage.DECLINE_PHASE
        
        # Seller stage transitions
        elif profile.entity_type == "seller":
            if current_stage == LifecycleStage.ACCOUNT_CREATED:
                # Check for first listing
                if any(e.event_type == "listing_created" for e in profile.events):
                    return LifecycleStage.FIRST_LISTING
            
            elif current_stage == LifecycleStage.FIRST_LISTING:
                # Check for establishment (>20 products, >50 sales)
                products = len(set(e.metadata.get("product_id") for e in profile.events if e.event_type == "listing_created"))
                sales = len([e for e in profile.events if e.event_type == "sale"])
                if products > 20 and sales > 50:
                    return LifecycleStage.ESTABLISHED
        
        return current_stage
    
    async def _calculate_lifecycle_risk(self, profile: LifecycleProfile) -> float:
        """
        Calculate risk score based on lifecycle patterns
        """
        risk_score = 0.0
        
        # Young account with high activity
        if profile.metrics.get("age_days", 0) < 30 and profile.metrics.get("events_last_7d", 0) > 100:
            risk_score += 0.3
        
        # Rapid stage transitions
        if len(set(e.metadata.get("stage") for e in profile.events if "stage" in e.metadata)) > 3:
            if profile.metrics.get("age_days", 0) < 60:
                risk_score += 0.2
        
        # High anomaly rate
        anomaly_rate = profile.metrics.get("anomaly_rate", 0)
        risk_score += anomaly_rate * 0.4
        
        # Dormancy after high activity
        if profile.current_stage == LifecycleStage.DORMANT:
            prev_activity = profile.metrics.get("events_last_30d", 0)
            curr_activity = profile.metrics.get("events_last_7d", 0)
            if prev_activity > 50 and curr_activity < 5:
                risk_score += 0.3
        
        return min(risk_score, 1.0)
    
    async def get_lifecycle_insights(
        self,
        entity_type: str,
        entity_id: str
    ) -> Dict[str, Any]:
        """
        Get comprehensive lifecycle insights for an entity
        """
        profile_key = f"{entity_type}_{entity_id}"
        profile = self.lifecycle_data.get(profile_key)
        
        if not profile:
            return {"error": "Entity not found"}
        
        return {
            "entity_id": entity_id,
            "entity_type": entity_type,
            "current_stage": profile.current_stage.value,
            "age_days": profile.metrics.get("age_days", 0),
            "risk_score": profile.risk_score,
            "anomalies": profile.anomalies[-5:],  # Last 5 anomalies
            "metrics": profile.metrics,
            "stage_progression": self._get_stage_progression(profile),
            "recommendations": self._get_lifecycle_recommendations(profile)
        }
    
    def _get_stage_progression(self, profile: LifecycleProfile) -> List[Dict]:
        """Get entity's progression through lifecycle stages"""
        progression = []
        current_stage = None
        
        for event in profile.events:
            if "stage" in event.metadata and event.metadata["stage"] != current_stage:
                current_stage = event.metadata["stage"]
                progression.append({
                    "stage": current_stage,
                    "timestamp": event.timestamp.isoformat(),
                    "duration_days": 0
                })
        
        # Calculate durations
        for i in range(len(progression) - 1):
            duration = (progression[i+1]["timestamp"] - progression[i]["timestamp"]).days
            progression[i]["duration_days"] = duration
        
        return progression
    
    def _get_lifecycle_recommendations(self, profile: LifecycleProfile) -> List[str]:
        """Get recommendations based on lifecycle analysis"""
        recommendations = []
        
        if profile.risk_score > 0.7:
            recommendations.append("High risk - Enable enhanced monitoring")
        
        if profile.current_stage == LifecycleStage.GROWTH_PHASE and profile.anomalies:
            recommendations.append("Unusual growth pattern - Verify authenticity")
        
        if profile.current_stage == LifecycleStage.DORMANT:
            recommendations.append("Account dormant - Check for exit scam preparation")
        
        return recommendations

# ============= Predictive Fraud Engine =============

class PredictiveFraudEngine:
    """
    Predicts future fraud patterns based on historical data and current signals
    """
    
    def __init__(self):
        self.prediction_models = self._initialize_models()
        self.seasonal_patterns = self._load_seasonal_patterns()
        self.prediction_history: List[FraudPrediction] = []
        
    def _initialize_models(self) -> Dict:
        """Initialize prediction models for different fraud types"""
        return {
            FraudPattern.REVIEW_BOMB: self._create_review_bomb_predictor(),
            FraudPattern.EXIT_SCAM: self._create_exit_scam_predictor(),
            FraudPattern.SEASONAL_FRAUD: self._create_seasonal_predictor(),
            FraudPattern.PRICE_MANIPULATION: self._create_price_predictor()
        }
    
    def _create_review_bomb_predictor(self):
        """Model to predict review bombing attacks"""
        # In production, this would be a trained ML model
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
        """Model to predict exit scams"""
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
        """Model to predict seasonal fraud spikes"""
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
        """Model to predict price manipulation"""
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
        """Load known seasonal fraud patterns"""
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
        """
        Predict potential fraud patterns for an entity
        """
        predictions = []
        
        # Check each fraud pattern
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
        """
        Calculate probability of specific fraud pattern
        """
        probability = 0.0
        
        if pattern == FraudPattern.REVIEW_BOMB:
            # Check for review bomb indicators
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
            # Check for exit scam indicators
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
            # Check proximity to major events
            days_to_black_friday = self._days_to_event("black_friday")
            if days_to_black_friday < 30:
                probability += 0.3 * (30 - days_to_black_friday) / 30
            
            # Check category risk
            if signals.get("category") in ["Electronics", "Luxury"]:
                probability += 0.2
            
            # Check seller preparation
            if signals.get("recent_listing_increase", 0) > 50:
                probability += 0.15
            if signals.get("new_seller", False):
                probability += 0.15
                
        elif pattern == FraudPattern.PRICE_MANIPULATION:
            # Check for price manipulation indicators
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
        """Calculate prediction confidence"""
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
        """Estimate when fraud will occur"""
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
        """Identify specific risk factors"""
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
        """Recommend preventive actions"""
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
        """Calculate days to major shopping event"""
        # Simplified - in production would use actual calendar
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
        """
        Validate past predictions to improve model
        """
        prediction = next((p for p in self.prediction_history if p.prediction_id == prediction_id), None)
        
        if prediction:
            # Log validation result
            validation = {
                "prediction_id": prediction_id,
                "predicted_pattern": prediction.fraud_pattern.value,
                "predicted_probability": prediction.probability,
                "actual_outcome": actual_outcome,
                "timestamp": datetime.now()
            }
            
            # In production, this would update model weights
            logger.info(f"Prediction validation: {validation}")
            
            # Calculate accuracy metrics
            if actual_outcome:
                logger.info(f"Correct prediction for {prediction.fraud_pattern.value}")
            else:
                logger.info(f"False positive for {prediction.fraud_pattern.value}")

# ============= Integration with Cross Intelligence =============

class EnhancedCrossIntelligence:
    """
    Enhanced Cross Intelligence with Lifecycle and Predictive capabilities
    """
    
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
        """
        Enhance detection with lifecycle and predictive analysis
        """
        # Track lifecycle event
        lifecycle_profile = await self.lifecycle_engine.track_entity(
            entity_type,
            entity_id,
            {
                "type": "detection_event",
                "fraud_score": detection_result.get("fraud_score", 0),
                "detection_type": detection_result.get("fraud_type", "unknown")
            }
        )
        
        # Get lifecycle insights
        lifecycle_insights = await self.lifecycle_engine.get_lifecycle_insights(
            entity_type,
            entity_id
        )
        
        # Predict future fraud
        predictions = await self.predictive_engine.predict_fraud(
            entity_type,
            entity_id,
            detection_result,
            lifecycle_profile
        )
        
        # Combine all intelligence
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
        """Combine recommendations from all sources"""
        recommendations = []
        
        # Add detection-based recommendations
        if detection.get("fraud_score", 0) > 0.7:
            recommendations.append("Immediate action required - high fraud score")
        
        # Add lifecycle recommendations
        recommendations.extend(lifecycle.get("recommendations", []))
        
        # Add predictive recommendations
        for prediction in predictions:
            if prediction.probability > 0.8:
                recommendations.extend(prediction.recommended_actions)
        
        # Remove duplicates while preserving order
        seen = set()
        unique_recommendations = []
        for rec in recommendations:
            if rec not in seen:
                seen.add(rec)
                unique_recommendations.append(rec)
        
        return unique_recommendations

# ============= Example Usage =============

async def main():
    """Example usage of Intelligence Engines"""
    
    # Initialize engines
    lifecycle_engine = LifecycleMonitoringEngine()
    predictive_engine = PredictiveFraudEngine()
    
    # Example 1: Track product lifecycle
    product_id = "PROD123"
    
    # Simulate lifecycle events
    events = [
        {"type": "listing_created", "price": 99.99},
        {"type": "sale", "quantity": 1},
        {"type": "sale", "quantity": 5},
        {"type": "review_added", "rating": 5},
        {"type": "sales_update", "daily_sales": 50},
        {"type": "price_change", "price": 149.99},
        {"type": "sales_update", "daily_sales": 500}  # Suspicious spike
    ]
    
    for event in events:
        profile = await lifecycle_engine.track_entity("product", product_id, event)
        await asyncio.sleep(0.1)
    
    # Get lifecycle insights
    insights = await lifecycle_engine.get_lifecycle_insights("product", product_id)
    print(f"Lifecycle Insights: {json.dumps(insights, indent=2, default=str)}")
    
    # Example 2: Predict fraud patterns
    signals = {
        "competitor_new_product": True,
        "recent_price_drop": 25,
        "review_velocity_change": 300,
        "category": "Electronics",
        "discount_percentage": 75,
        "inventory_spike": 1000
    }
    
    predictions = await predictive_engine.predict_fraud(
        "product",
        product_id,
        signals,
        profile
    )
    
    print(f"\nFraud Predictions:")
    for pred in predictions:
        print(f"- {pred.fraud_pattern.value}: {pred.probability:.2%} ({pred.confidence.value})")
        print(f"  Timeframe: {pred.predicted_timeframe}")
        print(f"  Risk Factors: {pred.risk_factors}")
        print(f"  Recommended Actions: {pred.recommended_actions}")
    
    # Example 3: Integrated intelligence
    enhanced_intel = EnhancedCrossIntelligence()
    
    detection_result = {
        "fraud_type": "review_fraud",
        "fraud_score": 0.75,
        "network_size": 12
    }
    
    enhanced_result = await enhanced_intel.analyze_with_intelligence(
        "product",
        product_id,
        detection_result
    )
    
    print(f"\nEnhanced Intelligence Result:")
    print(f"Combined Risk Score: {enhanced_result['risk_score']:.2%}")
    print(f"Recommendations: {enhanced_result['recommended_actions']}")

if __name__ == "__main__":
    asyncio.run(main())