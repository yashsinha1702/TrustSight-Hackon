from fastapi import FastAPI, WebSocket
from fastapi.middleware.cors import CORSMiddleware
import json
import random
from datetime import datetime
import asyncio

app = FastAPI()

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load your data
with open('../src/data/products.json', 'r') as f:
    products = json.load(f)

with open('../src/data/sellers.json', 'r') as f:
    sellers = json.load(f)

@app.get("/api/v1/trust-score/product/{product_id}")
async def get_product_trust_score(product_id: str):
    # Calculate trust score based on product data
    product = next((p for p in products if p['product_id'] == product_id), None)
    
    if not product:
        return {"error": "Product not found"}
    
    trust_score = 100
    fraud_indicators = []
    
    if 'COUNTERFEIT' in product_id:
        trust_score -= 50
        fraud_indicators.append({
            "type": "COUNTERFEIT_DETECTED",
            "description": "Product appears to be counterfeit"
        })
    
    if product.get('market_price') and product['price'] < product['market_price'] * 0.5:
        trust_score -= 20
        fraud_indicators.append({
            "type": "PRICE_ANOMALY",
            "description": f"Price is {round((1 - product['price']/product['market_price']) * 100)}% below market"
        })
    
    return {
        "product_id": product_id,
        "trust_score": max(0, trust_score),
        "fraud_indicators": fraud_indicators,
        "verification_status": "VERIFIED" if trust_score > 80 else "UNVERIFIED",
        "timestamp": datetime.now().isoformat()
    }

@app.get("/api/v1/seller/risk/{seller_id}")
async def get_seller_risk(seller_id: str):
    seller = next((s for s in sellers if s['seller_id'] == seller_id), None)
    
    if not seller:
        return {"error": "Seller not found"}
    
    trust_score = 100
    
    if not seller.get('seller_authorized'):
        trust_score -= 30
    
    if seller['seller_metrics']['return_rate'] > 0.15:
        trust_score -= 20
    
    if seller['seller_metrics']['customer_complaints'] > 200:
        trust_score -= 15
    
    return {
        "seller_id": seller_id,
        "trust_score": max(0, trust_score),
        "is_authorized": seller.get('seller_authorized', False),
        "risk_level": "LOW" if trust_score > 80 else "MEDIUM" if trust_score > 50 else "HIGH"
    }

@app.websocket("/ws/fraud-alerts")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            # Simulate real-time alerts
            await asyncio.sleep(5)
            alert = {
                "id": random.randint(1000, 9999),
                "type": random.choice(["COUNTERFEIT_DETECTED", "FAKE_REVIEW_WAVE", "SELLER_ANOMALY"]),
                "message": "Suspicious activity detected",
                "severity": random.choice(["HIGH", "MEDIUM", "CRITICAL"]),
                "timestamp": datetime.now().isoformat()
            }
            await websocket.send_json(alert)
    except Exception:
        pass

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8002)