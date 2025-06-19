import logging
from fastapi import FastAPI, HTTPException
from .api_models import AppealRequest, DecisionQuery
from .models import Appeal
from .database_models import DecisionRecord
from .decision_memory_bank import FraudDecisionMemoryBank

logger = logging.getLogger(__name__)

app = FastAPI(title="TrustSight Decision Memory Bank API", version="1.0.0")

memory_bank = None

@app.on_event("startup")
async def startup_event():
    """Initialize memory bank on API startup"""
    global memory_bank
    memory_bank = FraudDecisionMemoryBank()
    logger.info("Decision Memory Bank API started")

@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup resources on API shutdown"""
    if memory_bank:
        memory_bank.scheduler.shutdown()

@app.post("/appeal")
async def submit_appeal(appeal_request: AppealRequest):
    """Submit an appeal for a fraud detection decision"""
    try:
        appeal = Appeal(
            decision_id=appeal_request.decision_id,
            seller_id=appeal_request.seller_id,
            reason=appeal_request.reason,
            evidence=appeal_request.evidence
        )
        
        appeal_id = await memory_bank.submit_appeal(appeal)
        
        return {
            "appeal_id": appeal_id,
            "status": "submitted",
            "message": "Your appeal has been submitted and will be reviewed within 24 hours"
        }
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Appeal submission error: {e}")
        raise HTTPException(status_code=500, detail="Appeal submission failed")

@app.get("/decision/{decision_id}")
async def get_decision(decision_id: str):
    """Get decision details by decision ID"""
    db = memory_bank.SessionLocal()
    try:
        decision = db.query(DecisionRecord).filter_by(decision_id=decision_id).first()
        
        if not decision:
            raise HTTPException(status_code=404, detail="Decision not found")
        
        return {
            "decision_id": decision.decision_id,
            "entity_id": decision.entity_id,
            "status": decision.status,
            "fraud_score": decision.fraud_score,
            "confidence_score": decision.confidence_score,
            "timestamp": decision.timestamp,
            "verification_result": decision.verification_result,
            "appeal_status": "submitted" if decision.appeal_submitted else "none",
            "appeal_outcome": decision.appeal_outcome
        }
        
    finally:
        db.close()

@app.post("/query")
async def query_decisions(query: DecisionQuery):
    """Query decisions with optional filters"""
    db = memory_bank.SessionLocal()
    try:
        q = db.query(DecisionRecord)
        
        if query.entity_id:
            q = q.filter(DecisionRecord.entity_id == query.entity_id)
        if query.entity_type:
            q = q.filter(DecisionRecord.entity_type == query.entity_type)
        if query.date_from:
            q = q.filter(DecisionRecord.timestamp >= query.date_from)
        if query.date_to:
            q = q.filter(DecisionRecord.timestamp <= query.date_to)
        if query.status:
            q = q.filter(DecisionRecord.status == query.status)
        
        decisions = q.limit(100).all()
        
        return {
            "count": len(decisions),
            "decisions": [
                {
                    "decision_id": d.decision_id,
                    "entity_id": d.entity_id,
                    "status": d.status,
                    "fraud_score": d.fraud_score,
                    "timestamp": d.timestamp
                }
                for d in decisions
            ]
        }
        
    finally:
        db.close()

@app.get("/metrics")
async def get_metrics():
    """Get memory bank operational metrics"""
    return memory_bank.get_metrics()

@app.get("/health")
async def health_check():
    """API health check endpoint"""
    return {"status": "healthy", "component": "decision_memory_bank"}