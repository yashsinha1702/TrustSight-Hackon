import asyncio
import logging
from typing import Dict, List, Any
from enums import AlertPriority, AlertRecipient
from data_classes import Alert

logger = logging.getLogger(__name__)

class RealTimeAlertSystem:
    """
    Manages real-time alert notifications for buyers, sellers, and internal teams
    """
    
    def __init__(self):
        self.alert_templates = self._load_alert_templates()
        self.alert_queue = asyncio.Queue()
        self.alert_history: List[Alert] = []
        
    def _load_alert_templates(self) -> Dict:
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
        template = self.alert_templates.get(alert_type, {})
        
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
        
        await self.alert_queue.put(alert)
        self.alert_history.append(alert)
        
        logger.info(f"Alert sent: {alert.alert_id} to {recipient_type.value}")
        return alert
    
    async def process_alert_queue(self):
        while True:
            try:
                alert = await self.alert_queue.get()
                logger.info(f"Processing alert: {alert.title}")
                await asyncio.sleep(0.1)
            except Exception as e:
                logger.error(f"Error processing alert: {e}")