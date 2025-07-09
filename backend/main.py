# Enhanced main.py with additional endpoints
from fastapi import FastAPI, Request, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import subprocess
import uuid
import json
import os
import sys
import asyncio
from datetime import datetime
from typing import Dict, List, Optional
from pydantic import BaseModel

app = FastAPI(
    title="TrustSight Fraud Detection API",
    description="Advanced e-commerce fraud detection and prevention system",
    version="2.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# WebSocket connection manager for real-time alerts
class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)

    async def send_personal_message(self, message: str, websocket: WebSocket):
        await websocket.send_text(message)

    async def broadcast(self, message: str):
        for connection in self.active_connections:
            try:
                await connection.send_text(message)
            except:
                # Remove dead connections
                self.active_connections.remove(connection)

manager = ConnectionManager()

# Pydantic models for API requests
class ProductTrustRequest(BaseModel):
    product_id: str
    
class SellerRiskRequest(BaseModel):
    seller_id: str

class SuspiciousActivityReport(BaseModel):
    product_id: Optional[str] = None
    seller_id: Optional[str] = None
    reviewer_id: Optional[str] = None
    reason: str
    details: Optional[Dict] = None
    reporter_id: Optional[str] = None

class FraudAlert(BaseModel):
    alert_id: str
    alert_type: str
    severity: str
    message: str
    entity_id: str
    entity_type: str
    timestamp: datetime
    details: Optional[Dict] = None

# In-memory storage for demo purposes (use a proper database in production)
fraud_reports = []
fraud_alerts = []
trust_scores_cache = {}

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "service": "TrustSight API", "version": "2.0.0"}

@app.post("/investigate")
async def run_pipeline(request: Request):
    """Main fraud investigation endpoint"""
    data = await request.json()
    print("Received investigation request:", data)
    
    # Save input to temp JSON
    input_path = f"input_{uuid.uuid4().hex}.json"
    with open(input_path, "w") as f:
        json.dump(data, f)

    print(f"Processing with input file: {input_path}")
    
    try:
        # Get the directory where main.py is located
        script_dir = os.path.dirname(os.path.abspath(__file__))
        script_path = os.path.join(script_dir, "trustsight_inference_pipeline.py")
        
        # Use absolute path for input file too
        abs_input_path = os.path.abspath(input_path)
        
        # Run the subprocess with proper environment
        result = subprocess.run(
            [sys.executable, script_path, abs_input_path],
            cwd=script_dir,  # Set working directory
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=120
        )

        # Clean up input file
        if os.path.exists(input_path):
            os.remove(input_path)
        if os.path.exists(abs_input_path) and abs_input_path != input_path:
            os.remove(abs_input_path)

        # Check return code
        if result.returncode != 0:
            return {
                "error": f"Script failed with return code {result.returncode}", 
                "stderr": result.stderr,
                "stdout": result.stdout
            }

        # Check if we have output
        if not result.stdout or not result.stdout.strip():
            return {
                "error": "Script produced no output", 
                "stderr": result.stderr,
                "return_code": result.returncode
            }

        # Try to parse JSON, handling any non-JSON prefix
        try:
            stdout_content = result.stdout.strip()
            
            # Find the first '{' character (start of JSON)
            json_start = stdout_content.find('{')
            if json_start > 0:
                print(f"Found non-JSON content before JSON: {stdout_content[:json_start]}")
                stdout_content = stdout_content[json_start:]
            elif json_start == -1:
                return {
                    "error": "No JSON found in output",
                    "stdout": result.stdout,
                    "stderr": result.stderr
                }
            
            parsed_result = json.loads(stdout_content)
            
            # Send real-time alert if high risk detected
            await send_fraud_alert_if_needed(parsed_result, data)
            
            print("Successfully parsed JSON result")
            return parsed_result
            
        except json.JSONDecodeError as e:
            print(f"JSON decode error: {e}")
            return {
                "error": f"Invalid JSON output: {str(e)}", 
                "stdout": result.stdout,
                "stderr": result.stderr,
                "json_error_position": e.pos if hasattr(e, 'pos') else None
            }

    except subprocess.TimeoutExpired:
        print("Script timed out")
        if os.path.exists(input_path):
            os.remove(input_path)
        return {"error": "Script execution timed out (120 seconds)"}
        
    except FileNotFoundError as e:
        print(f"File not found error: {e}")
        if os.path.exists(input_path):
            os.remove(input_path)
        return {"error": f"Script file not found: {str(e)}"}
        
    except Exception as e:
        print(f"Unexpected error: {e}")
        if os.path.exists(input_path):
            os.remove(input_path)
        return {"error": f"Unexpected error: {str(e)}"}

@app.get("/api/v1/trust-score/product/{product_id}")
async def get_product_trust_score(product_id: str):
    """Get trust score for a specific product"""
    
    # Check cache first
    if product_id in trust_scores_cache:
        return trust_scores_cache[product_id]
    
    # Generate mock trust score based on product ID patterns
    trust_score = 75  # Default
    fraud_indicators = []
    
    if "COUNTERFEIT" in product_id:
        trust_score = 25
        fraud_indicators.append({
            "type": "COUNTERFEIT_RISK",
            "severity": "HIGH",
            "description": "Product flagged as potential counterfeit"
        })
    elif "EXIT_SCAM" in product_id:
        trust_score = 30
        fraud_indicators.append({
            "type": "EXIT_SCAM_RISK", 
            "severity": "HIGH",
            "description": "Seller associated with exit scam patterns"
        })
    elif "REVIEW_FARM" in product_id:
        trust_score = 60
        fraud_indicators.append({
            "type": "FAKE_REVIEWS",
            "severity": "MEDIUM", 
            "description": "Suspicious review patterns detected"
        })
    elif "LEGIT" in product_id:
        trust_score = 90
    
    result = {
        "product_id": product_id,
        "trust_score": trust_score,
        "fraud_indicators": fraud_indicators,
        "verification_status": "VERIFIED" if trust_score >= 80 else "CAUTION" if trust_score >= 60 else "HIGH_RISK",
        "last_updated": datetime.now().isoformat()
    }
    
    # Cache the result
    trust_scores_cache[product_id] = result
    return result

@app.get("/api/v1/seller/risk/{seller_id}")
async def get_seller_risk(seller_id: str):
    """Get risk assessment for a specific seller"""
    
    # Generate mock seller risk based on seller ID patterns
    trust_score = 80  # Default
    risk_level = "LOW"
    is_authorized = True
    
    if "COUNTERFEIT" in seller_id or "EXIT_SCAM" in seller_id:
        trust_score = 20
        risk_level = "CRITICAL"
        is_authorized = False
    elif "REVIEW_FARM" in seller_id:
        trust_score = 55
        risk_level = "MEDIUM"
        is_authorized = True
    
    return {
        "seller_id": seller_id,
        "trust_score": trust_score,
        "risk_level": risk_level,
        "is_authorized": is_authorized,
        "risk_factors": [
            "High return rate" if trust_score < 50 else None,
            "Suspicious pricing patterns" if "COUNTERFEIT" in seller_id else None,
            "Review manipulation" if "REVIEW_FARM" in seller_id else None
        ],
        "last_assessment": datetime.now().isoformat()
    }

@app.post("/api/v1/report")
async def report_suspicious_activity(report: SuspiciousActivityReport):
    """Report suspicious activity"""
    
    report_id = str(uuid.uuid4())
    report_data = {
        "report_id": report_id,
        "timestamp": datetime.now().isoformat(),
        "status": "received",
        **report.dict()
    }
    
    fraud_reports.append(report_data)
    
    # Send alert to connected clients
    alert = FraudAlert(
        alert_id=str(uuid.uuid4()),
        alert_type="USER_REPORT",
        severity="MEDIUM",
        message=f"User reported suspicious activity: {report.reason}",
        entity_id=report.product_id or report.seller_id or "unknown",
        entity_type="product" if report.product_id else "seller" if report.seller_id else "unknown",
        timestamp=datetime.now(),
        details={"report_id": report_id}
    )
    
    await broadcast_alert(alert)
    
    return {
        "success": True,
        "report_id": report_id,
        "message": "Report received and will be investigated"
    }

@app.get("/api/v1/stats/fraud")
async def get_fraud_stats():
    """Get fraud detection statistics"""
    
    # Mock statistics - in production, these would come from your database
    return {
        "total_products_scanned": 1500,
        "fraudulent_products_detected": 287,
        "counterfeit_products": 156,
        "fake_reviews_blocked": 1243,
        "suspicious_sellers": 45,
        "fraud_networks_disrupted": 12,
        "monetary_fraud_prevented": 2450000,
        "last_24h": {
            "new_fraud_cases": 23,
            "products_analyzed": 156,
            "alerts_generated": 8
        },
        "detection_accuracy": 94.2,
        "false_positive_rate": 2.1
    }

@app.get("/api/v1/alerts")
async def get_recent_alerts(limit: int = 50):
    """Get recent fraud alerts"""
    return {
        "alerts": fraud_alerts[-limit:],
        "total_count": len(fraud_alerts)
    }

@app.websocket("/ws/fraud-alerts")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for real-time fraud alerts"""
    await manager.connect(websocket)
    try:
        while True:
            # Keep connection alive and wait for client messages
            data = await websocket.receive_text()
            # Echo back for testing
            await manager.send_personal_message(f"Echo: {data}", websocket)
    except WebSocketDisconnect:
        manager.disconnect(websocket)

async def send_fraud_alert_if_needed(analysis_result, original_data):
    """Send real-time alert if fraud is detected"""
    
    detections = analysis_result.get("detections", {})
    
    # Check for high-risk conditions
    high_risk_detected = False
    alert_message = ""
    severity = "LOW"
    
    if detections.get("counterfeit", {}).get("score", 0) > 50:
        high_risk_detected = True
        alert_message = "High counterfeit risk detected"
        severity = "HIGH"
    elif detections.get("seller", {}).get("fraud_probability", 0) > 0.7:
        high_risk_detected = True
        alert_message = "Seller fraud probability high"
        severity = "HIGH"
    elif detections.get("review", {}).get("avg_fraud_score", 0) > 0.8:
        high_risk_detected = True
        alert_message = "Significant fake review activity detected"
        severity = "MEDIUM"
    
    if high_risk_detected:
        alert = FraudAlert(
            alert_id=str(uuid.uuid4()),
            alert_type="FRAUD_DETECTED",
            severity=severity,
            message=alert_message,
            entity_id=original_data.get("product", {}).get("title", "unknown"),
            entity_type="product",
            timestamp=datetime.now(),
            details={"analysis_summary": detections}
        )
        
        await broadcast_alert(alert)

async def broadcast_alert(alert: FraudAlert):
    """Broadcast alert to all connected WebSocket clients"""
    
    # Add to alerts list
    fraud_alerts.append(alert.dict())
    
    # Broadcast to WebSocket clients
    alert_message = json.dumps({
        "type": "fraud_alert",
        "data": alert.dict()
    }, default=str)
    
    await manager.broadcast(alert_message)

@app.get("/api/v1/test-pipeline")
async def test_pipeline():
    """Test endpoint to verify pipeline script exists and can be called"""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    script_path = os.path.join(script_dir, "trustsight_inference_pipeline.py")
    
    return {
        "script_exists": os.path.exists(script_path),
        "script_path": script_path,
        "current_directory": os.getcwd(),
        "script_directory": script_dir,
        "python_executable": sys.executable
    }

@app.get("/api/v1/dashboard/summary")
async def get_dashboard_summary():
    """Get summary data for the TrustSight dashboard"""
    
    return {
        "system_status": "operational",
        "active_investigations": 15,
        "fraud_networks_monitored": 8,
        "real_time_analysis": True,
        "last_update": datetime.now().isoformat(),
        "performance_metrics": {
            "analysis_speed": "1.2s avg",
            "accuracy": "94.2%",
            "uptime": "99.7%"
        }
    }

# Error handlers
@app.exception_handler(404)
async def not_found_handler(request: Request, exc):
    return JSONResponse(
        status_code=404,
        content={"error": "Endpoint not found", "available_endpoints": [
            "/health", "/investigate", "/api/v1/trust-score/product/{id}",
            "/api/v1/seller/risk/{id}", "/api/v1/report", "/api/v1/stats/fraud"
        ]}
    )

@app.exception_handler(500)
async def internal_error_handler(request: Request, exc):
    return JSONResponse(
        status_code=500,
        content={"error": "Internal server error", "message": str(exc)}
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)