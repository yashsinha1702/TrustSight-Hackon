from typing import Dict, List, Any
from pydantic import BaseModel, Field

class DetectionRequestAPI(BaseModel):
    """Pydantic model for API fraud detection requests"""
    entity_type: str = Field(..., description="Type of entity: product, review, seller, listing")
    entity_id: str = Field(..., description="Unique identifier for the entity")
    entity_data: Dict[str, Any] = Field(..., description="Entity data for detection")
    priority: str = Field(default="MEDIUM", description="Processing priority")

class BatchDetectionRequestAPI(BaseModel):
    """Pydantic model for batch fraud detection requests"""
    requests: List[DetectionRequestAPI]
    process_async: bool = Field(default=True, description="Process asynchronously")