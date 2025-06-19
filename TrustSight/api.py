import asyncio
import logging
from typing import Dict
from fastapi import FastAPI, HTTPException
from .api_models import DetectionRequestAPI, BatchDetectionRequestAPI
from .models import DetectionRequest
from .enums import Priority
from .integration_engine import TrustSightIntegrationEngine

logger = logging.getLogger(__name__)

app = FastAPI(title="TrustSight Integration API", version="1.0.0")

engine = None

async def process_priority_queue(engine: TrustSightIntegrationEngine):
    """Processes requests from priority queue"""
    logger.info("Priority queue processor started")
    
    while True:
        try:
            priority, request = await engine.priority_queue.get()
            
            asyncio.create_task(engine.process_detection_request(request))
            
        except asyncio.CancelledError:
            break
        except Exception as e:
            logger.error(f"Queue processor error: {e}")
            await asyncio.sleep(1)

@app.on_event("startup")
async def startup_event():
    """Initialize engine on API startup"""
    global engine
    engine = TrustSightIntegrationEngine()
    
    engine.queue_processor_task = asyncio.create_task(process_priority_queue(engine))
    
    logger.info("API server started")

@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup resources on API shutdown"""
    if engine and engine.queue_processor_task:
        engine.queue_processor_task.cancel()

@app.get("/health")
async def health_check():
    """API health check endpoint"""
    return {
        "status": "healthy",
        "active_requests": len(engine.active_requests) if engine else 0,
        "detectors": list(engine.detectors.keys()) if engine else []
    }

@app.post("/detect", response_model=Dict)
async def detect_fraud(request: DetectionRequestAPI):
    """Single entity fraud detection endpoint"""
    detection_request = DetectionRequest(
        entity_type=request.entity_type,
        entity_id=request.entity_id,
        entity_data=request.entity_data,
        priority=Priority[request.priority],
        source='api'
    )
    
    result = await engine.process_detection_request(detection_request)
    
    return {
        "request_id": result.request_id,
        "entity_id": result.entity_id,
        "trust_score": result.trust_score,
        "fraud_score": result.overall_fraud_score,
        "fraud_detected": result.overall_fraud_score > 0.5,
        "cross_intel_triggered": result.cross_intel_triggered,
        "network_size": result.cross_intel_result.get('network_size', 0) if result.cross_intel_result else 0,
        "recommendations": result.recommendations,
        "priority_actions": result.priority_actions,
        "processing_time": result.processing_time
    }

@app.post("/detect/batch")
async def detect_fraud_batch(batch_request: BatchDetectionRequestAPI):
    """Batch fraud detection endpoint"""
    tasks = []
    
    for req in batch_request.requests:
        detection_request = DetectionRequest(
            entity_type=req.entity_type,
            entity_id=req.entity_id,
            entity_data=req.entity_data,
            priority=Priority[req.priority],
            source='api_batch'
        )
        
        if batch_request.process_async:
            await engine.priority_queue.put((detection_request.priority.value, detection_request))
            tasks.append(detection_request.request_id)
        else:
            task = asyncio.create_task(engine.process_detection_request(detection_request))
            tasks.append(task)
    
    if batch_request.process_async:
        return {
            "message": f"Queued {len(tasks)} requests for processing",
            "request_ids": tasks
        }
    else:
        results = await asyncio.gather(*tasks)
        return {
            "results": [
                {
                    "request_id": r.request_id,
                    "entity_id": r.entity_id,
                    "trust_score": r.trust_score,
                    "fraud_score": r.overall_fraud_score
                }
                for r in results
            ]
        }

@app.get("/status/{request_id}")
async def get_request_status(request_id: str):
    """Get status of a detection request"""
    if request_id in engine.active_requests:
        return {"status": "processing", "request_id": request_id}
    elif request_id in engine.completed_requests:
        return {"status": "completed", "request_id": request_id}
    else:
        raise HTTPException(status_code=404, detail="Request not found")

@app.get("/metrics")
async def get_metrics():
    """Get system operational metrics"""
    return {
        "active_requests": len(engine.active_requests),
        "completed_requests": len(engine.completed_requests),
        "queue_size": engine.priority_queue.qsize(),
        "cache_size": len(engine.memory_cache) if hasattr(engine, 'memory_cache') else 0,
        "detectors": {
            name: "active" for name in engine.detectors.keys()
        }
    }