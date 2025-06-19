import asyncio
import json
import logging
from typing import Dict
from aiokafka import AIOKafkaConsumer
from prometheus_client import Counter
from .models import DetectionRequest

logger = logging.getLogger(__name__)

kafka_messages_processed = Counter('trustsight_kafka_messages', 'Kafka messages processed', ['topic'])

class KafkaStreamProcessor:
    """Processes incoming Kafka streams for real-time fraud detection"""
    
    def __init__(self, integration_engine):
        self.engine = integration_engine
        self.config = integration_engine.config
        self.running = False
        
    async def start(self):
        self.running = True
        
        tasks = []
        for topic_name, topic in self.config.KAFKA_TOPICS['input'].items():
            task = asyncio.create_task(self._consume_topic(topic_name, topic))
            tasks.append(task)
        
        logger.info("Kafka stream processors started")
        await asyncio.gather(*tasks)
    
    async def stop(self):
        self.running = False
    
    async def _consume_topic(self, topic_name: str, topic: str):
        consumer = AIOKafkaConsumer(
            topic,
            bootstrap_servers=self.config.KAFKA_BOOTSTRAP_SERVERS,
            value_deserializer=lambda m: json.loads(m.decode('utf-8')),
            group_id=f'trustsight-{topic_name}'
        )
        
        await consumer.start()
        logger.info(f"Started consuming from {topic}")
        
        try:
            async for message in consumer:
                if not self.running:
                    break
                
                kafka_messages_processed.labels(topic=topic_name).inc()
                
                await self._process_message(topic_name, message.value)
                
        finally:
            await consumer.stop()
    
    async def _process_message(self, topic_name: str, message: Dict):
        try:
            entity_type_map = {
                'product-events': 'product',
                'review-events': 'review',
                'seller-events': 'seller',
                'listing-events': 'listing'
            }
            
            entity_type = entity_type_map.get(topic_name, 'unknown')
            
            request = DetectionRequest(
                entity_type=entity_type,
                entity_id=message.get('entity_id', ''),
                entity_data=message,
                source='kafka',
                metadata={'kafka_topic': topic_name}
            )
            
            request.priority = self.engine.calculate_priority(request)
            
            await self.engine.priority_queue.put((request.priority.value, request))
            
        except Exception as e:
            logger.error(f"Error processing Kafka message: {e}")