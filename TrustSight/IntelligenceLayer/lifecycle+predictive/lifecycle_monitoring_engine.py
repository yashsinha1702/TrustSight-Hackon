import logging
import numpy as np
from collections import defaultdict
from datetime import datetime, timedelta
from typing import Dict, List, Any
from .enums import LifecycleStage
from .models import LifecycleEvent, LifecycleProfile

logger = logging.getLogger(__name__)

class LifecycleMonitoringEngine:
    """Tracks entities throughout their lifecycle to identify abuse patterns and anomalous behavior"""
    
    def __init__(self):
        self.lifecycle_data: Dict[str, LifecycleProfile] = {}
        self.stage_transitions = self._define_stage_transitions()
        self.anomaly_thresholds = self._define_anomaly_thresholds()
        
    def _define_stage_transitions(self) -> Dict:
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
        return {
            "rapid_growth": {
                "sales_spike": 500,
                "review_spike": 1000,
                "timeframe": 7
            },
            "sudden_dormancy": {
                "activity_drop": 90,
                "timeframe": 3
            },
            "price_volatility": {
                "max_change": 50,
                "frequency": 3
            }
        }
    
    async def track_entity(
        self,
        entity_type: str,
        entity_id: str,
        event: Dict[str, Any]
    ) -> LifecycleProfile:
        profile_key = f"{entity_type}_{entity_id}"
        if profile_key not in self.lifecycle_data:
            self.lifecycle_data[profile_key] = LifecycleProfile(
                entity_type=entity_type,
                entity_id=entity_id,
                current_stage=LifecycleStage.LISTING_CREATED if entity_type == "product" else LifecycleStage.ACCOUNT_CREATED,
                creation_date=datetime.now()
            )
        
        profile = self.lifecycle_data[profile_key]
        
        lifecycle_event = LifecycleEvent(
            event_id=f"evt_{datetime.now().timestamp()}",
            entity_type=entity_type,
            entity_id=entity_id,
            event_type=event.get("type", "unknown"),
            timestamp=datetime.now(),
            metadata=event
        )
        
        anomaly_score = await self._detect_lifecycle_anomalies(profile, lifecycle_event)
        lifecycle_event.anomaly_score = anomaly_score
        
        profile.events.append(lifecycle_event)
        
        await self._update_lifecycle_metrics(profile)
        
        new_stage = await self._check_stage_transition(profile)
        if new_stage != profile.current_stage:
            profile.current_stage = new_stage
            logger.info(f"Entity {entity_id} transitioned to {new_stage.value}")
        
        profile.risk_score = await self._calculate_lifecycle_risk(profile)
        profile.last_updated = datetime.now()
        
        return profile
    
    async def _detect_lifecycle_anomalies(
        self,
        profile: LifecycleProfile,
        event: LifecycleEvent
    ) -> float:
        anomaly_score = 0.0
        
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
        age_days = (datetime.now() - profile.creation_date).days
        profile.metrics["age_days"] = age_days
        
        events_last_7d = [e for e in profile.events if (datetime.now() - e.timestamp).days <= 7]
        events_last_30d = [e for e in profile.events if (datetime.now() - e.timestamp).days <= 30]
        
        profile.metrics["events_last_7d"] = len(events_last_7d)
        profile.metrics["events_last_30d"] = len(events_last_30d)
        
        if profile.current_stage:
            stage_events = [e for e in profile.events if e.metadata.get("stage") == profile.current_stage.value]
            if stage_events:
                stage_duration = (datetime.now() - stage_events[0].timestamp).days
                profile.metrics[f"{profile.current_stage.value}_duration"] = stage_duration
        
        total_anomalies = len(profile.anomalies)
        profile.metrics["anomaly_rate"] = total_anomalies / max(len(profile.events), 1)
    
    async def _check_stage_transition(self, profile: LifecycleProfile) -> LifecycleStage:
        current_stage = profile.current_stage
        
        if profile.entity_type == "product":
            if current_stage == LifecycleStage.LISTING_CREATED:
                if any(e.event_type == "sale" for e in profile.events):
                    return LifecycleStage.FIRST_SALE
            
            elif current_stage == LifecycleStage.FIRST_SALE:
                recent_sales = [e for e in profile.events if e.event_type == "sale" and (datetime.now() - e.timestamp).days <= 7]
                if len(recent_sales) > 10:
                    return LifecycleStage.GROWTH_PHASE
            
            elif current_stage == LifecycleStage.GROWTH_PHASE:
                if profile.metrics.get("age_days", 0) > 90:
                    return LifecycleStage.MATURE_PHASE
            
            elif current_stage == LifecycleStage.MATURE_PHASE:
                if profile.metrics.get("events_last_7d", 0) < profile.metrics.get("events_last_30d", 0) / 4 * 0.5:
                    return LifecycleStage.DECLINE_PHASE
        
        elif profile.entity_type == "seller":
            if current_stage == LifecycleStage.ACCOUNT_CREATED:
                if any(e.event_type == "listing_created" for e in profile.events):
                    return LifecycleStage.FIRST_LISTING
            
            elif current_stage == LifecycleStage.FIRST_LISTING:
                products = len(set(e.metadata.get("product_id") for e in profile.events if e.event_type == "listing_created"))
                sales = len([e for e in profile.events if e.event_type == "sale"])
                if products > 20 and sales > 50:
                    return LifecycleStage.ESTABLISHED
        
        return current_stage
    
    async def _calculate_lifecycle_risk(self, profile: LifecycleProfile) -> float:
        risk_score = 0.0
        
        if profile.metrics.get("age_days", 0) < 30 and profile.metrics.get("events_last_7d", 0) > 100:
            risk_score += 0.3
        
        if len(set(e.metadata.get("stage") for e in profile.events if "stage" in e.metadata)) > 3:
            if profile.metrics.get("age_days", 0) < 60:
                risk_score += 0.2
        
        anomaly_rate = profile.metrics.get("anomaly_rate", 0)
        risk_score += anomaly_rate * 0.4
        
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
            "anomalies": profile.anomalies[-5:],
            "metrics": profile.metrics,
            "stage_progression": self._get_stage_progression(profile),
            "recommendations": self._get_lifecycle_recommendations(profile)
        }
    
    def _get_stage_progression(self, profile: LifecycleProfile) -> List[Dict]:
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
        
        for i in range(len(progression) - 1):
            duration = (progression[i+1]["timestamp"] - progression[i]["timestamp"]).days
            progression[i]["duration_days"] = duration
        
        return progression
    
    def _get_lifecycle_recommendations(self, profile: LifecycleProfile) -> List[str]:
        recommendations = []
        
        if profile.risk_score > 0.7:
            recommendations.append("High risk - Enable enhanced monitoring")
        
        if profile.current_stage == LifecycleStage.GROWTH_PHASE and profile.anomalies:
            recommendations.append("Unusual growth pattern - Verify authenticity")
        
        if profile.current_stage == LifecycleStage.DORMANT:
            recommendations.append("Account dormant - Check for exit scam preparation")
        
        return recommendations