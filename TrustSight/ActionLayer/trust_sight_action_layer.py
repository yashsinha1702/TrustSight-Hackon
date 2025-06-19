import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List
from dataclasses import asdict

from enums import ActionType, ActionPhase, AlertRecipient
from data_classes import ActionRequest, ActionResult, TrustScore
from trust_score_calculator import TrustScoreCalculator
from real_time_alert_system import RealTimeAlertSystem
from network_action_engine import NetworkActionEngine

logger = logging.getLogger(__name__)

class TrustSightActionLayer:
    """
    Main orchestrator implementing two-phase response system for fraud detection and response
    """
    
    def __init__(self, decision_memory_bank=None):
        self.trust_calculator = TrustScoreCalculator()
        self.alert_system = RealTimeAlertSystem()
        self.network_engine = NetworkActionEngine()
        self.decision_memory = decision_memory_bank
        
        self.thresholds = {
            "soft_action": 0.60,
            "hard_action": 0.80,
            "network_action": 0.75,
            "auto_action": 0.95
        }
        
        self.action_history: Dict[str, ActionResult] = {}
        
    async def process_fraud_detection(
        self,
        action_request: ActionRequest
    ) -> ActionResult:
        logger.info(f"Processing action request: {action_request.request_id}")
        
        trust_score = self.trust_calculator.calculate_trust_score(
            action_request.entity_type,
            action_request.entity_id,
            {
                f"{action_request.fraud_type}_score": action_request.fraud_score,
                "network_connection_score": min(action_request.network_size / 100, 1.0)
            }
        )
        
        if action_request.fraud_score >= self.thresholds["auto_action"]:
            phase = ActionPhase.HARD
            immediate_action = True
        elif action_request.fraud_score >= self.thresholds["hard_action"]:
            phase = ActionPhase.HARD
            immediate_action = False
        else:
            phase = ActionPhase.SOFT
            immediate_action = True
        
        if phase == ActionPhase.SOFT or immediate_action:
            result = await self._execute_soft_actions(action_request, trust_score)
        else:
            result = await self._queue_for_verification(action_request, trust_score)
        
        self.action_history[result.action_id] = result
        await self._send_alerts(action_request, result, trust_score)
        
        return result
    
    async def _execute_soft_actions(
        self,
        request: ActionRequest,
        trust_score: TrustScore
    ) -> ActionResult:
        result = ActionResult(
            request_id=request.request_id,
            phase=ActionPhase.SOFT,
            reversible=True,
            trust_score_impact=100 - trust_score.overall_score
        )
        
        actions_taken = []
        
        if request.fraud_type == "counterfeit":
            actions_taken.append({
                "type": ActionType.WARNING_BANNER.value,
                "entity_id": request.entity_id,
                "message": "⚠️ This product is under review for authenticity",
                "visibility": "public"
            })
            
            actions_taken.append({
                "type": ActionType.LISTING_SUPPRESSION.value,
                "entity_id": request.entity_id,
                "scope": "recommendations_only",
                "duration": "until_verified"
            })
            
        elif request.fraud_type == "review_fraud":
            for review_id in request.affected_entities[:10]:
                actions_taken.append({
                    "type": ActionType.REVIEW_HOLD.value,
                    "entity_id": review_id,
                    "reason": "Pending authenticity verification",
                    "visible_to_seller": True
                })
            
        elif request.fraud_type == "seller_fraud":
            actions_taken.append({
                "type": ActionType.PAYMENT_DELAY.value,
                "entity_id": request.entity_id,
                "duration": "7_days",
                "reason": "Additional verification required"
            })
            
            actions_taken.append({
                "type": ActionType.INTERNAL_ALERT.value,
                "team": "trust_safety",
                "priority": "high",
                "seller_id": request.entity_id
            })
        
        actions_taken.append({
            "type": ActionType.TRUST_SCORE_REDUCTION.value,
            "entity_id": request.entity_id,
            "old_score": 100,
            "new_score": trust_score.overall_score,
            "reason": f"{request.fraud_type} detected"
        })
        
        result.actions_taken = actions_taken
        
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
        result = ActionResult(
            request_id=request.request_id,
            phase=ActionPhase.HARD,
            reversible=False,
            trust_score_impact=100 - trust_score.overall_score
        )
        
        if self.decision_memory:
            verification_request = {
                "decision_id": request.decision_id,
                "fraud_score": request.fraud_score,
                "confidence": request.confidence,
                "proposed_actions": self._determine_hard_actions(request),
                "priority": "high" if request.fraud_score > 0.9 else "medium"
            }
            
            result.actions_taken.append({
                "type": "verification_queued",
                "decision_id": request.decision_id,
                "estimated_verification_time": "24-48 hours",
                "verification_request": verification_request
            })
        
        return result
    
    def _determine_hard_actions(self, request: ActionRequest) -> List[str]:
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
        original_request = None
        
        result = ActionResult(
            phase=ActionPhase.HARD,
            reversible=False
        )
        
        if verification_result == "CONFIRMED_FRAUD":
            if original_request and original_request.network_size > 1:
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
        alerts_to_send = []
        
        if request.fraud_type == "counterfeit":
            alerts_to_send.append((
                AlertRecipient.BUYER,
                "buyer_id_placeholder",
                "buyer_counterfeit_warning",
                {
                    "confidence": int(request.confidence * 100),
                    "product_id": request.entity_id
                }
            ))
            
            alerts_to_send.append((
                AlertRecipient.BRAND_OWNER,
                "brand_id_placeholder",
                "brand_counterfeit_alert",
                {"product_id": request.entity_id}
            ))
        
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
        
        for recipient_type, recipient_id, alert_type, metadata in alerts_to_send:
            alert = await self.alert_system.send_alert(
                recipient_type,
                recipient_id,
                alert_type,
                metadata
            )
            result.alerts_sent.append(asdict(alert))
    
    async def rollback_action(self, action_id: str) -> bool:
        if action_id not in self.action_history:
            return False
        
        action_result = self.action_history[action_id]
        
        if not action_result.reversible:
            logger.warning(f"Action {action_id} is not reversible")
            return False
        
        for action in action_result.actions_taken:
            if action["type"] == ActionType.WARNING_BANNER.value:
                logger.info(f"Removing warning banner from {action['entity_id']}")
            elif action["type"] == ActionType.LISTING_SUPPRESSION.value:
                logger.info(f"Restoring listing {action['entity_id']}")
            elif action["type"] == ActionType.PAYMENT_DELAY.value:
                logger.info(f"Releasing payment for {action['entity_id']}")
            elif action["type"] == ActionType.TRUST_SCORE_REDUCTION.value:
                logger.info(f"Restoring trust score for {action['entity_id']}")
        
        return True