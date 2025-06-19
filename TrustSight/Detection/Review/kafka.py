import json
from datetime import datetime
from inference import FraudDetectionInference

class KafkaFraudDetector:
    """Real-time fraud detection integration with Kafka streaming for production deployment"""
    
    def __init__(self, model_inference: FraudDetectionInference):
        self.detector = model_inference
        
    async def process_review_stream(self, kafka_consumer, kafka_producer):
        batch = []
        
        async for message in kafka_consumer:
            review = json.loads(message.value)
            batch.append(review)
            
            if len(batch) >= self.detector.config.batch_size:
                results = self.detector.predict_batch(batch)
                
                for review, result in zip(batch, results):
                    if result['fraud_probability'] > 0.5:
                        await self.send_fraud_alert(
                            kafka_producer,
                            review,
                            result
                        )
                
                batch = []
    
    async def send_fraud_alert(self, producer, review, result):
        alert = {
            'detection_id': f"FRAUD_{review['review_id']}",
            'timestamp': datetime.now().isoformat(),
            'review_id': review['review_id'],
            'fraud_probability': result['fraud_probability'],
            'confidence': result['confidence'],
            'fraud_signals': result['fraud_signals'],
            'explanation': result['explanation'],
            'action_required': 'INVESTIGATE' if result['confidence'] > 0.8 else 'MONITOR'
        }
        
        await producer.send(
            'fraud-detection-events',
            value=json.dumps(alert).encode()
        )