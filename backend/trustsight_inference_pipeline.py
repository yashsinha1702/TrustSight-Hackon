# trustsight_inference_pipeline.py

import asyncio
import json
from datetime import datetime
from typing import Dict, Any
# --- Detection Layer ---
from detector import HybridCounterfeitDetector
from seller import SellerNetworkInterface, SellerNetworkConfig
from listing import ListingFraudInference
from review import FraudDetectionInference, ModelConfig

# --- Intelligence & Action Layer ---
from cross_intelligence import CrossIntelligenceEngine, FraudSignal, SignalType
from lifecycle_predictive import LifecyclePredictiveInterface
from action_layer import TrustSightActionLayer,ActionRequest
from decision_memory_bank import FraudDecisionMemoryBank
import os
import sys
class TrustSightPipeline:
    def __init__(self):
        # Detection
        self.counterfeit = HybridCounterfeitDetector()
        self.seller = SellerNetworkInterface("seller_network_best_f1_0.5707.pt", SellerNetworkConfig())
        self.listing = ListingFraudInference("listing_fraud_ensemble_model.pkl", "listing_fraud_scaler.pkl")
        self.review = FraudDetectionInference("best_model_f1_1.0000.pt", ModelConfig())

        # Intelligence & Action
        self.cross_intel = CrossIntelligenceEngine()
        self.lifecycle = LifecyclePredictiveInterface()
        self.actions = TrustSightActionLayer()

        # Memory
        self.memory = FraudDecisionMemoryBank()

    async def run(self, input_data: dict):
        product = input_data.get("product", {})
        seller = input_data.get("seller", {})
        listing = input_data.get("listing", {})
        reviews = input_data.get("reviews", [])

        # --- Detection Layer ---
        detection = {
            "counterfeit": self.counterfeit.detect(product),
            "seller": self.seller.analyze_seller(seller),
            "listing": self.listing.predict(listing),
            "review": self.review.predict_batch(reviews)
        }

        # --- Intelligence: Build Signals ---
        signals = []
        if detection["review"]["avg_fraud_score"] > 0.5:
            signals.append(FraudSignal(
                signal_id=f"REV_{product['title']}",
                signal_type=SignalType.FAKE_REVIEW,
                entity_id=seller["seller_id"],
                confidence=detection["review"]["avg_fraud_score"],
                timestamp=datetime.now(),
                metadata={"reviews": reviews}
            ))
        if detection["seller"]["fraud_probability"] > 0.5:
            signals.append(FraudSignal(
                signal_id=f"SELLER_{seller['seller_id']}",
                signal_type=SignalType.SELLER_FRAUD,
                entity_id=seller["seller_id"],
                confidence=detection["seller"]["fraud_probability"],
                timestamp=datetime.now(),
                metadata=detection["seller"]
            ))
        if detection["listing"]["label"] == "1" or detection["listing"]["label"] == 1:
            signals.append(FraudSignal(
                signal_id=f"LIST_{listing['listing_id']}",
                signal_type=SignalType.LISTING_FRAUD,
                entity_id=listing["listing_id"],
                confidence=detection["listing"]["confidence"],
                timestamp=datetime.now(),
                metadata=listing
            ))

        # --- Run Cross Intelligence ---
        intel_results = []
        for sig in signals:
            res = await self.cross_intel.trace_fraud_network(sig)
            intel_results.append(res)

        # print("Intel results: ",intel_results)
        # --- Lifecycle Intelligence ---
        lifecycle_results = []
        for sig in signals:
            lifecycle = await self.lifecycle.infer(sig.signal_type.name, sig.entity_id, detection)
            lifecycle_results.append(lifecycle)

        # --- Action Layer ---
        final_actions = []
        for idx, sig in enumerate(signals):
            investigation = intel_results[idx]
            lifecycle = lifecycle_results[idx]

            action_input = ActionRequest(
                fraud_type=sig.signal_type.name.lower(),
                entity_type="seller" if "SELLER" in sig.signal_id else "product",
                entity_id=sig.entity_id,
                fraud_score=sig.confidence,
                confidence=sig.confidence,
                network_size=len(investigation.network_graph.nodes),
                affected_entities=list(investigation.network_graph.nodes),
                evidence=detection,
                decision_id=str(sig.signal_id),
                network_indicators={
                    "financial_impact": getattr(investigation, "financial_impact", 0),
                    "key_players": getattr(investigation, "key_players", [])
                },
                lifecycle_stage=lifecycle.get("current_stage", None),
                predicted_risks=lifecycle.get("recommended_actions", []),
                entity_data=seller if sig.signal_type.name == "SELLER_FRAUD" else product,
                detection_timestamp=sig.timestamp
            )

            action = await self.actions.process_fraud_detection(action_input)
            final_actions.append(action)


        # --- Decision Memory ---
        memory_ids = []
        for idx, sig in enumerate(signals):
            decision_id = await self.memory.record_decision(
                {
    "signal_id": sig.signal_id,
    "signal_type": sig.signal_type.name,
    "entity_id": sig.entity_id,
    "confidence": sig.confidence,
    "timestamp": str(sig.timestamp),
    "metadata": sig.metadata,
    "source_detector": sig.source_detector
},
                {
                    "overall_fraud_score": sig.confidence,
                    "confidence": sig.confidence,
                    "primary_evidence": [detection],
                    "model_versions": {"TrustSight v2.0": 1.0},
                    "action": final_actions[idx]
                }
            )
            memory_ids.append(decision_id)

        return {
            "detections": detection,
            "cross_intel": intel_results,
            "lifecycle": lifecycle_results,
            "actions": final_actions,
            "decisions": memory_ids
        }


# Example usage
# if __name__ == "__main__":
#     input_data = {
#         "product": {
#             "title": "Nike shoes - Official Edition",
#             "description": "Top quality shoes",
#             "price": 89.5,
#             "mrp": 129,
#             "image_1": "img_001.jpg",
#             "images": ["img_001.jpg"],
#             "seller_name": "FastTraders",
#             "brand_raw": "Nike",
#             "brand_fuzzy": "Nike"
#         },
#         "seller": {
#             "seller_id": "S_123",
#             "business_name": "Speedy Supplies Inc.",
#             "display_name": "FastTraders",
#             "account_age_days": 250,
#             "seller_metrics": {
#                 "total_products": 700,
#                 "active_products": 350,
#                 "avg_product_price": 29.99,
#                 "avg_rating": 4.5,
#                 "response_time_hours": 12,
#                 "fulfillment_rate": 0.95,
#                 "return_rate": 0.1,
#                 "customer_complaints": 50
#             },
#             "pricing_behavior": {
#                 "avg_price_change_frequency_days": 5,
#                 "max_price_drop_percent": 60,
#                 "synchronized_changes_count": 12,
#                 "competitor_price_matching": True,
#                 "dynamic_pricing_detected": False
#             },
#             "inventory_patterns": {
#                 "avg_stock_level": 300,
#                 "max_stock_spike": 8000,
#                 "inventory_turnover_days": 40
#             },
#             "network_features": {
#                 "shared_product_count": 10,
#                 "address_similarity_score": 0.5,
#                 "customer_overlap_score": 0.7,
#                 "reviewer_overlap_score": 0.8
#             },
#             "temporal_features": {
#                 "registration_hour": 10
#             }
#         },
#         "listing": {
#             "listing_id": "LIST_001",
#             "feature_mismatch_score": 0.25,
#             "feature_evolution_score": 0.3,
#             "feature_seo_score": 0.55,
#             "feature_variation_score": 0.1,
#             "feature_hijack_score": 0.3,
#             "feature_review_count": 0.4,
#             "feature_price_variance": 0.5,
#             "feature_seller_age": 0.3,
#             "feature_category_competitiveness": 0.4
#         },
#         "reviews": [
#             {
#                 "review_id": "R001",
#                 "review_text": "Best product! So useful and good one!!",
#                 "review_date": "2024-09-10"
#             },
#             {
#                 "review_id": "R002",
#                 "review_text": "I love it. Very nice quality. You can use it!",
#                 "review_date": "2024-10-02"
#             }
#         ]
#     }

#     pipeline = TrustSightPipeline()
#     result = asyncio.run(pipeline.run(input_data))


#     def serialize(obj):
#         if hasattr(obj, "to_dict"):
#             return obj.to_dict()
#         elif isinstance(obj, (datetime,)):
#             return obj.isoformat()
#         elif isinstance(obj, set):
#             return list(obj)
#         else:
#             return str(obj)

#     print(json.dumps(result, default=serialize, indent=2))


def serialize(obj):
    """Custom JSON serializer for complex objects"""
    if hasattr(obj, "to_dict"):
        return obj.to_dict()
    elif isinstance(obj, datetime):
        return obj.isoformat()
    elif isinstance(obj, set):
        return list(obj)
    else:
        return str(obj)


if __name__ == "__main__":
    import os
    
    # Redirect stdout temporarily to capture only JSON output
    original_stdout = sys.stdout
    original_stderr = sys.stderr
    
    try:
        # Check if input file argument is provided
        if len(sys.argv) < 2:
            error_response = {"error": "No input file provided"}
            print(json.dumps(error_response), file=original_stdout)
            original_stdout.flush()
            sys.exit(1)

        input_path = sys.argv[1]
        
        # Check if input file exists
        if not os.path.exists(input_path):
            error_response = {"error": f"Input file not found: {input_path}"}
            print(json.dumps(error_response), file=original_stdout)
            original_stdout.flush()
            sys.exit(1)

        # Load input data
        try:
            with open(input_path, "r") as f:
                input_data = json.load(f)
        except json.JSONDecodeError as e:
            error_response = {"error": f"Invalid JSON in input file: {str(e)}"}
            print(json.dumps(error_response), file=original_stdout)
            original_stdout.flush()
            sys.exit(1)
        except Exception as e:
            error_response = {"error": f"Error reading input file: {str(e)}"}
            print(json.dumps(error_response), file=original_stdout)
            original_stdout.flush()
            sys.exit(1)

        # Redirect stdout to stderr to prevent contamination of JSON output
        # This ensures any print statements from imported modules go to stderr
        sys.stdout = sys.stderr
        
        # Initialize and run pipeline (any prints will go to stderr now)
        pipeline = TrustSightPipeline()
        result = asyncio.run(pipeline.run(input_data))

        # Restore stdout and output clean JSON
        sys.stdout = original_stdout
        
        # Output result as JSON to original stdout
        json_output = json.dumps(result, default=serialize, ensure_ascii=False)
        print(json_output, file=original_stdout)
        original_stdout.flush()

    except Exception as e:
        # Restore stdout for error output
        sys.stdout = original_stdout
        
        # Handle any unexpected errors
        import traceback
        error_response = {
            "error": f"Pipeline execution failed: {str(e)}",
            "traceback": traceback.format_exc()
        }
        print(json.dumps(error_response), file=original_stdout)
        original_stdout.flush()
        sys.exit(1)
    
    finally:
        # Ensure stdout is always restored
        sys.stdout = original_stdout