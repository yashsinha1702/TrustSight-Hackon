from datetime import datetime
from sqlalchemy import Column, String, Float, DateTime, JSON, Boolean, Integer
from sqlalchemy.ext.declarative import declarative_base

Base = declarative_base()

class DecisionRecord(Base):
    """SQLAlchemy model for persisting fraud detection decisions"""
    __tablename__ = 'decisions'
    
    decision_id = Column(String, primary_key=True)
    timestamp = Column(DateTime, default=datetime.utcnow)
    detection_type = Column(String)
    entity_type = Column(String)
    entity_id = Column(String)
    fraud_score = Column(Float)
    confidence_score = Column(Float)
    
    primary_evidence = Column(JSON)
    supporting_signals = Column(JSON)
    network_map = Column(JSON)
    model_versions = Column(JSON)
    
    status = Column(String, default='PENDING_VERIFICATION')
    verification_result = Column(String)
    verification_timestamp = Column(DateTime)
    verification_details = Column(JSON)
    
    action_taken = Column(String)
    automated_actions = Column(JSON)
    manual_actions = Column(JSON)
    
    feedback_applied = Column(Boolean, default=False)
    model_update_version = Column(String)
    
    storage_tier = Column(String, default='hot')
    ttl = Column(Integer)
    expires_at = Column(DateTime)
    
    appeal_submitted = Column(Boolean, default=False)
    appeal_timestamp = Column(DateTime)
    appeal_details = Column(JSON)
    appeal_outcome = Column(String)