"""
TRUSTSIGHT ACTION LAYER - Two-Phase Response System
Implements Soft Actions (immediate) and Hard Actions (post-verification)
Includes Trust Score Calculator, Real-time Alerts, and Network Action Engine
"""

import asyncio
import json
import logging
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Optional, Set, Tuple, Any
from uuid import uuid4
import numpy as np
from concurrent.futures import ThreadPoolExecutor

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ============= Enums and Constants =============

class ActionType(Enum):
    """Types of actions that can be taken"""
    # Soft Actions (Immediate, Reversible)
    WARNING_BANNER = "warning_banner"
    REVIEW_HOLD = "review_hold"
    LISTING_SUPPRESSION = "listing_suppression"
    PAYMENT_DELAY = "payment_delay"
    INTERNAL_ALERT = "internal_alert"
    TRUST_SCORE_REDUCTION = "trust_score_reduction"
    
    # Hard Actions (Post-Verification, Severe)
    ACCOUNT_SUSPENSION = "account_suspension"
    LISTING_TAKEDOWN = "listing_takedown"
    AUTO_REFUND = "auto_refund"
    PERMANENT_BAN = "permanent_ban"
    PAYMENT_FREEZE = "payment_freeze"
    NETWORK_SHUTDOWN = "network_shutdown"

class ActionPhase(Enum):
    """Action phases"""
    SOFT = "soft"  # Immediate, reversible
    HARD = "hard"  # Post-verification, severe

class AlertPriority(Enum):
    """Alert priority levels"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

class AlertRecipient(Enum):
    """Who receives alerts"""
    BUYER = "buyer"
    SELLER = "seller"
    INTERNAL_TEAM = "internal_team"
    BRAND_OWNER = "brand_owner"
    LEGAL_TEAM = "legal_team"

# ============= Data Classes =============

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from uuid import uuid4
from datetime import datetime

@dataclass
class ActionRequest:
    request_id: str = field(default_factory=lambda: str(uuid4()))
    fraud_type: str = ""
    entity_type: str = ""  # product, seller, review, network
    entity_id: str = ""
    fraud_score: float = 0.0
    confidence: float = 0.0
    evidence: Dict[str, Any] = field(default_factory=dict)
    network_size: int = 1
    affected_entities: List[str] = field(default_factory=list)
    decision_id: Optional[str] = None
    timestamp: datetime = field(default_factory=datetime.now)
    entity_data: Optional[Dict[str, Any]] = field(default=None)
    detection_timestamp: Optional[datetime] = None
    network_indicators: Optional[Dict[str, Any]] = field(default_factory=dict)
    lifecycle_stage: Optional[str] = None
    predicted_risks: Optional[List[str]] = field(default_factory=list)  # ✅ FINAL field







@dataclass
class ActionResult:
    """Result of action execution"""
    action_id: str = field(default_factory=lambda: str(uuid4()))
    request_id: str = ""
    actions_taken: List[Dict[str, Any]] = field(default_factory=list)
    phase: ActionPhase = ActionPhase.SOFT
    reversible: bool = True
    rollback_info: Optional[Dict] = None
    alerts_sent: List[Dict] = field(default_factory=list)
    trust_score_impact: float = 0.0
    success: bool = True
    error_message: Optional[str] = None
    timestamp: datetime = field(default_factory=datetime.now)

@dataclass
class TrustScore:
    """Multi-dimensional trust score (0-100)"""
    overall_score: float = 100.0
    dimensions: Dict[str, float] = field(default_factory=lambda: {
        "review_authenticity": 100.0,
        "product_legitimacy": 100.0,
        "seller_reliability": 100.0,
        "pricing_fairness": 100.0,
        "network_isolation": 100.0  # Not connected to fraud networks
    })
    history: List[Dict] = field(default_factory=list)
    last_updated: datetime = field(default_factory=datetime.now)

@dataclass
class Alert:
    """Alert notification"""
    alert_id: str = field(default_factory=lambda: str(uuid4()))
    priority: AlertPriority = AlertPriority.MEDIUM
    recipient_type: AlertRecipient = AlertRecipient.INTERNAL_TEAM
    recipient_id: Optional[str] = None
    title: str = ""
    message: str = ""
    action_required: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)

# ============= Trust Score Calculator =============

class TrustScoreCalculator:
    """
    Calculates multi-dimensional trust scores based on various signals
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
        """
        Calculate comprehensive trust score
        """
        trust_score = TrustScore()
        
        # Update dimensions based on fraud signals
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
        
        # Calculate weighted overall score
        trust_score.overall_score = sum(
            score * self.weights[dimension]
            for dimension, score in trust_score.dimensions.items()
        )
        
        # Add to history
        trust_score.history.append({
            "timestamp": datetime.now().isoformat(),
            "overall_score": trust_score.overall_score,
            "dimensions": trust_score.dimensions.copy(),
            "fraud_signals": fraud_signals
        })
        
        return trust_score
    
    def get_risk_level(self, trust_score: float) -> str:
        """Determine risk level from trust score"""
        if trust_score >= 80:
            return "low_risk"
        elif trust_score >= 60:
            return "medium_risk"
        elif trust_score >= 40:
            return "high_risk"
        else:
            return "critical_risk"

# ============= Real-time Alert System =============

class RealTimeAlertSystem:
    """
    Manages real-time alerts for buyers, sellers, and internal teams
    """
    
    def __init__(self):
        self.alert_templates = self._load_alert_templates()
        self.alert_queue = asyncio.Queue()
        self.alert_history: List[Alert] = []
        
    def _load_alert_templates(self) -> Dict:
        """Load predefined alert templates"""
        return {
            "buyer_counterfeit_warning": {
                "title": "⚠️ Potential Counterfeit Product",
                "message": "This product may not be authentic. Our AI has detected {confidence}% probability of counterfeit. Consider purchasing from verified sellers.",
                "priority": AlertPriority.HIGH,
                "action_required": True
            },
            "seller_network_detection": {
                "title": "🔍 Unusual Activity Detected",
                "message": "Your account has been flagged for review due to unusual patterns. Please verify your account to continue selling.",
                "priority": AlertPriority.CRITICAL,
                "action_required": True
            },
            "internal_fraud_network": {
                "title": "🚨 Fraud Network Detected",
                "message": "Network of {network_size} interconnected entities detected. Fraud score: {fraud_score}%. Immediate action recommended.",
                "priority": AlertPriority.CRITICAL,
                "action_required": True
            },
            "buyer_review_warning": {
                "title": "📋 Review Authenticity Warning",
                "message": "Multiple reviews for this product show signs of manipulation. Trust score: {trust_score}/100.",
                "priority": AlertPriority.MEDIUM,
                "action_required": False
            },
            "seller_payment_delay": {
                "title": "💳 Payment Processing Delayed",
                "message": "Your payment has been delayed for additional verification. This is a temporary measure to ensure marketplace safety.",
                "priority": AlertPriority.MEDIUM,
                "action_required": False
            }
        }
    
    async def send_alert(
        self,
        recipient_type: AlertRecipient,
        recipient_id: str,
        alert_type: str,
        metadata: Dict[str, Any]
    ) -> Alert:
        """
        Send real-time alert to specified recipient
        """
        template = self.alert_templates.get(alert_type, {})
        
        # Format message with metadata
        message = template.get("message", "").format(**metadata)
        
        alert = Alert(
            priority=template.get("priority", AlertPriority.MEDIUM),
            recipient_type=recipient_type,
            recipient_id=recipient_id,
            title=template.get("title", "Alert"),
            message=message,
            action_required=template.get("action_required", False),
            metadata=metadata
        )
        
        # Queue for async sending
        await self.alert_queue.put(alert)
        self.alert_history.append(alert)
        
        logger.info(f"Alert sent: {alert.alert_id} to {recipient_type.value}")
        return alert
    
    async def process_alert_queue(self):
        """Process queued alerts"""
        while True:
            try:
                alert = await self.alert_queue.get()
                # In production, this would send to SNS/SQS/Email
                logger.info(f"Processing alert: {alert.title}")
                await asyncio.sleep(0.1)  # Simulate sending
            except Exception as e:
                logger.error(f"Error processing alert: {e}")

# ============= Network Action Engine =============

class NetworkActionEngine:
    """
    Handles bulk actions on fraud networks
    """
    
    def __init__(self):
        self.executor = ThreadPoolExecutor(max_workers=10)
        
    async def execute_network_action(
        self,
        network_entities: List[Dict[str, Any]],
        action_type: ActionType,
        confidence: float
    ) -> Dict[str, Any]:
        """
        Execute bulk action on entire fraud network
        """
        results = {
            "total_entities": len(network_entities),
            "successful_actions": 0,
            "failed_actions": 0,
            "action_details": []
        }
        
        # Group entities by type
        entities_by_type = {}
        for entity in network_entities:
            entity_type = entity.get("type", "unknown")
            if entity_type not in entities_by_type:
                entities_by_type[entity_type] = []
            entities_by_type[entity_type].append(entity)
        
        # Execute actions in parallel
        tasks = []
        for entity_type, entities in entities_by_type.items():
            if action_type == ActionType.NETWORK_SHUTDOWN:
                # Comprehensive network shutdown
                tasks.extend([
                    self._shutdown_seller(e["id"]) for e in entities if e["type"] == "seller"
                ])
                tasks.extend([
                    self._remove_product(e["id"]) for e in entities if e["type"] == "product"
                ])
                tasks.extend([
                    self._quarantine_review(e["id"]) for e in entities if e["type"] == "review"
                ])
            elif action_type == ActionType.LISTING_SUPPRESSION:
                tasks.extend([
                    self._suppress_listing(e["id"]) for e in entities if e["type"] == "product"
                ])
        
        # Wait for all actions to complete
        completed_tasks = await asyncio.gather(*tasks, return_exceptions=True)
        
        for i, result in enumerate(completed_tasks):
            if isinstance(result, Exception):
                results["failed_actions"] += 1
                logger.error(f"Action failed: {result}")
            else:
                results["successful_actions"] += 1
                results["action_details"].append(result)
        
        return results
    
    async def _shutdown_seller(self, seller_id: str) -> Dict:
        """Shutdown seller account"""
        await asyncio.sleep(0.1)  # Simulate API call
        return {
            "action": "seller_shutdown",
            "seller_id": seller_id,
            "status": "suspended",
            "timestamp": datetime.now().isoformat()
        }
    
    async def _remove_product(self, product_id: str) -> Dict:
        """Remove product listing"""
        await asyncio.sleep(0.1)  # Simulate API call
        return {
            "action": "product_removal",
            "product_id": product_id,
            "status": "removed",
            "timestamp": datetime.now().isoformat()
        }
    
    async def _quarantine_review(self, review_id: str) -> Dict:
        """Quarantine suspicious review"""
        await asyncio.sleep(0.1)  # Simulate API call
        return {
            "action": "review_quarantine",
            "review_id": review_id,
            "status": "hidden",
            "timestamp": datetime.now().isoformat()
        }
    
    async def _suppress_listing(self, product_id: str) -> Dict:
        """Suppress product from search results"""
        await asyncio.sleep(0.1)  # Simulate API call
        return {
            "action": "listing_suppression",
            "product_id": product_id,
            "status": "suppressed",
            "timestamp": datetime.now().isoformat()
        }

# ============= Main Action Layer =============

class TrustSightActionLayer:
    """
    Main Action Layer orchestrator - implements two-phase response system
    """
    
    def __init__(self, decision_memory_bank=None):
        self.trust_calculator = TrustScoreCalculator()
        self.alert_system = RealTimeAlertSystem()
        self.network_engine = NetworkActionEngine()
        self.decision_memory = decision_memory_bank
        
        # Action thresholds
        self.thresholds = {
            "soft_action": 0.60,      # 60% fraud score triggers soft action
            "hard_action": 0.80,      # 80% fraud score triggers hard action
            "network_action": 0.75,   # 75% for network-wide actions
            "auto_action": 0.95       # 95% for automatic hard actions
        }
        
        # Track actions for rollback
        self.action_history: Dict[str, ActionResult] = {}
        
    async def process_fraud_detection(
        self,
        action_request: ActionRequest
    ) -> ActionResult:
        """
        Main entry point - process fraud detection and take appropriate action
        """
        # logger.info(f"[DEBUG] Received action_request: {action_request}")
        logger.info(f"Processing action request for entity: {action_request.entity_id} with signal: {action_request.fraud_type}")

        # Calculate trust score
        trust_score = self.trust_calculator.calculate_trust_score(
            action_request.entity_type,
            action_request.entity_id,
            {
                f"{action_request.fraud_type}_score": action_request.fraud_score,
                "network_connection_score": min(action_request.network_size / 100, 1.0)
            }
        )

        # Determine action phase based on score and confidence
        if action_request.fraud_score >= self.thresholds["auto_action"]:
            phase = ActionPhase.HARD
            immediate_action = True
        elif action_request.fraud_score >= self.thresholds["hard_action"]:
            phase = ActionPhase.HARD
            immediate_action = False  # Wait for verification
        else:
            phase = ActionPhase.SOFT
            immediate_action = True
        
        # Execute appropriate actions
        if phase == ActionPhase.SOFT or immediate_action:
            result = await self._execute_soft_actions(action_request, trust_score)
        else:
            # Queue for verification before hard actions
            result = await self._queue_for_verification(action_request, trust_score)
        
        # Store action history
        self.action_history[result.action_id] = result
        
        # Send alerts
        await self._send_alerts(action_request, result, trust_score)
        
        return result
    
    async def _execute_soft_actions(
        self,
        request: ActionRequest,
        trust_score: TrustScore
    ) -> ActionResult:
        """
        Execute immediate, reversible soft actions
        """
        result = ActionResult(
            request_id=request.request_id,
            phase=ActionPhase.SOFT,
            reversible=True,
            trust_score_impact=100 - trust_score.overall_score
        )
        
        actions_taken = []
        
        # Determine which soft actions to take
        if request.fraud_type == "counterfeit":
            # Add warning banner
            actions_taken.append({
                "type": ActionType.WARNING_BANNER.value,
                "entity_id": request.entity_id,
                "message": "⚠️ This product is under review for authenticity",
                "visibility": "public"
            })
            
            # Suppress from recommendations
            actions_taken.append({
                "type": ActionType.LISTING_SUPPRESSION.value,
                "entity_id": request.entity_id,
                "scope": "recommendations_only",
                "duration": "until_verified"
            })
            
        elif request.fraud_type == "review_fraud":
            # Hold suspicious reviews
            for review_id in request.affected_entities[:10]:  # Limit to 10
                actions_taken.append({
                    "type": ActionType.REVIEW_HOLD.value,
                    "entity_id": review_id,
                    "reason": "Pending authenticity verification",
                    "visible_to_seller": True
                })
            
        elif request.fraud_type == "seller_fraud":
            # Delay payments
            actions_taken.append({
                "type": ActionType.PAYMENT_DELAY.value,
                "entity_id": request.entity_id,
                "duration": "7_days",
                "reason": "Additional verification required"
            })
            
            # Internal alert
            actions_taken.append({
                "type": ActionType.INTERNAL_ALERT.value,
                "team": "trust_safety",
                "priority": "high",
                "seller_id": request.entity_id
            })
        
        # Always reduce trust score
        actions_taken.append({
            "type": ActionType.TRUST_SCORE_REDUCTION.value,
            "entity_id": request.entity_id,
            "old_score": 100,
            "new_score": trust_score.overall_score,
            "reason": f"{request.fraud_type} detected"
        })
        
        result.actions_taken = actions_taken
        
        # Create rollback information
        result.rollback_info = {
            "actions": actions_taken,
            "timestamp": datetime.now().isoformat(),
            "can_rollback_until": (datetime.now() + timedelta(days=30)).isoformat()
        }
        
        return result
    
    async def _queue_for_verification(
        self,
        request: ActionRequest,
        trust_score: TrustScore
    ) -> ActionResult:
        """
        Queue high-confidence detections for verification before hard actions
        """
        result = ActionResult(
            request_id=request.request_id,
            phase=ActionPhase.HARD,
            reversible=False,
            trust_score_impact=100 - trust_score.overall_score
        )
        
        # Create verification request for Decision Memory Bank
        if self.decision_memory:
            verification_request = {
                "decision_id": request.decision_id,
                "fraud_score": request.fraud_score,
                "confidence": request.confidence,
                "proposed_actions": self._determine_hard_actions(request),
                "priority": "high" if request.fraud_score > 0.9 else "medium"
            }
            
            # Queue for verification
            result.actions_taken.append({
                "type": "verification_queued",
                "decision_id": request.decision_id,
                "estimated_verification_time": "24-48 hours",
                "verification_request": verification_request
            })
        
        return result
    
    def _determine_hard_actions(self, request: ActionRequest) -> List[str]:
        """
        Determine which hard actions to take based on fraud type
        """
        actions = []
        
        if request.fraud_type == "counterfeit":
            actions.extend([
                ActionType.LISTING_TAKEDOWN.value,
                ActionType.AUTO_REFUND.value
            ])
        elif request.fraud_type == "seller_fraud":
            actions.extend([
                ActionType.ACCOUNT_SUSPENSION.value,
                ActionType.PAYMENT_FREEZE.value
            ])
        elif request.fraud_type == "review_fraud" and request.network_size > 10:
            actions.append(ActionType.NETWORK_SHUTDOWN.value)
        
        return actions
    
    async def execute_verified_hard_actions(
        self,
        decision_id: str,
        verification_result: str
    ) -> ActionResult:
        """
        Execute hard actions after verification from Decision Memory Bank
        """
        # Find original request (in production, this would be from database)
        original_request = None  # Would be retrieved
        
        result = ActionResult(
            phase=ActionPhase.HARD,
            reversible=False
        )
        
        if verification_result == "CONFIRMED_FRAUD":
            # Execute all proposed hard actions
            if original_request and original_request.network_size > 1:
                # Network-wide action
                network_result = await self.network_engine.execute_network_action(
                    original_request.affected_entities,
                    ActionType.NETWORK_SHUTDOWN,
                    original_request.confidence
                )
                result.actions_taken.append({
                    "type": ActionType.NETWORK_SHUTDOWN.value,
                    "network_result": network_result
                })
            else:
                # Individual entity actions
                result.actions_taken.extend([
                    {
                        "type": ActionType.ACCOUNT_SUSPENSION.value,
                        "duration": "permanent",
                        "reason": "Verified fraud"
                    },
                    {
                        "type": ActionType.AUTO_REFUND.value,
                        "affected_orders": "last_30_days",
                        "estimated_refund_amount": "$X,XXX"
                    }
                ])
        
        return result
    
    async def _send_alerts(
        self,
        request: ActionRequest,
        result: ActionResult,
        trust_score: TrustScore
    ):
        """
        Send appropriate alerts based on actions taken
        """
        alerts_to_send = []
        
        # Determine alert recipients and types
        if request.fraud_type == "counterfeit":
            # Alert buyers who might have purchased
            alerts_to_send.append((
                AlertRecipient.BUYER,
                "buyer_id_placeholder",
                "buyer_counterfeit_warning",
                {
                    "confidence": int(request.confidence * 100),
                    "product_id": request.entity_id
                }
            ))
            
            # Alert brand owner
            alerts_to_send.append((
                AlertRecipient.BRAND_OWNER,
                "brand_id_placeholder",
                "brand_counterfeit_alert",
                {"product_id": request.entity_id}
            ))
        
        # Internal alerts for high-risk detections
        if request.fraud_score > self.thresholds["hard_action"]:
            alerts_to_send.append((
                AlertRecipient.INTERNAL_TEAM,
                "trust_safety_team",
                "internal_fraud_network",
                {
                    "network_size": request.network_size,
                    "fraud_score": int(request.fraud_score * 100)
                }
            ))
        
        # Send all alerts
        for recipient_type, recipient_id, alert_type, metadata in alerts_to_send:
            alert = await self.alert_system.send_alert(
                recipient_type,
                recipient_id,
                alert_type,
                metadata
            )
            result.alerts_sent.append(asdict(alert))
    
    async def rollback_action(self, action_id: str) -> bool:
        """
        Rollback a soft action (e.g., after appeal approval)
        """
        if action_id not in self.action_history:
            return False
        
        action_result = self.action_history[action_id]
        
        if not action_result.reversible:
            logger.warning(f"Action {action_id} is not reversible")
            return False
        
        # Reverse each action
        for action in action_result.actions_taken:
            if action["type"] == ActionType.WARNING_BANNER.value:
                # Remove warning banner
                logger.info(f"Removing warning banner from {action['entity_id']}")
            elif action["type"] == ActionType.LISTING_SUPPRESSION.value:
                # Restore listing visibility
                logger.info(f"Restoring listing {action['entity_id']}")
            elif action["type"] == ActionType.PAYMENT_DELAY.value:
                # Release payment
                logger.info(f"Releasing payment for {action['entity_id']}")
            elif action["type"] == ActionType.TRUST_SCORE_REDUCTION.value:
                # Restore trust score
                logger.info(f"Restoring trust score for {action['entity_id']}")
        
        return True

# ============= Example Usage =============

async def main():
    """Example usage of Action Layer"""
    
    # Initialize action layer
    action_layer = TrustSightActionLayer()
    
    # Start alert processor
    alert_task = asyncio.create_task(
        action_layer.alert_system.process_alert_queue()
    )
    
    # Example 1: Counterfeit detection
    counterfeit_request = ActionRequest(
        fraud_type="counterfeit",
        entity_type="product",
        entity_id="PROD123",
        fraud_score=0.85,
        confidence=0.92,
        evidence={
            "image_analysis": "Stock photo detected",
            "price_anomaly": "70% below market",
            "seller_authority": "Not authorized dealer"
        },
        network_size=1
    )
    
    result1 = await action_layer.process_fraud_detection(counterfeit_request)
    print(f"Counterfeit Action Result: {result1.actions_taken}")
    
    # Example 2: Review fraud network
    review_fraud_request = ActionRequest(
        fraud_type="review_fraud",
        entity_type="network",
        entity_id="NET456",
        fraud_score=0.78,
        confidence=0.85,
        network_size=47,
        affected_entities=[f"REVIEW{i}" for i in range(47)],
        evidence={
            "timing_pattern": "All posted 3-4 AM",
            "text_similarity": "85% similar phrases",
            "reviewer_connections": "Shared IP addresses"
        }
    )
    
    result2 = await action_layer.process_fraud_detection(review_fraud_request)
    print(f"Review Network Action Result: {result2.actions_taken}")
    
    # Example 3: High-confidence seller fraud
    seller_fraud_request = ActionRequest(
        fraud_type="seller_fraud",
        entity_type="seller",
        entity_id="SELLER789",
        fraud_score=0.96,
        confidence=0.98,
        network_size=12,
        affected_entities=[f"SELLER{i}" for i in range(12)],
        evidence={
            "registration_pattern": "All registered same day",
            "inventory_overlap": "95% identical products",
            "pricing_coordination": "Synchronized price changes"
        }
    )
    
    result3 = await action_layer.process_fraud_detection(seller_fraud_request)
    print(f"Seller Network Action Result: {result3.actions_taken}")
    
    # Simulate verification callback
    await asyncio.sleep(2)
    
    # Example 4: Execute hard actions after verification
    verified_result = await action_layer.execute_verified_hard_actions(
        "decision_123",
        "CONFIRMED_FRAUD"
    )
    print(f"Verified Hard Actions: {verified_result.actions_taken}")
    
    # Cancel alert task
    alert_task.cancel()

if __name__ == "__main__":
    asyncio.run(main())