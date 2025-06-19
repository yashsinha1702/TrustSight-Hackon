# -------------------------
# File: dummy_llm.py
# -------------------------

class DummyLLM:
    def analyze_description(self, description):
        # Simple rule: if it's too short or very generic
        if len(description.split()) < 5 or "experience the quality" in description.lower():
            return {
                "verdict": "suspicious",
                "reason": "Description too short or uses generic phrases"
            }
        return {
            "verdict": "clean",
            "reason": "Description appears detailed and brand-specific"
        }

    def detect_translated_content(self, description):
        # Rule-based flag for poor grammar or typical translation artifacts
        translated_phrases = ["this product is very nice", "you can use it", "good one", "very useful item"]
        flags = [phrase for phrase in translated_phrases if phrase in description.lower()]
        return {
            "verdict": "suspicious" if flags else "clean",
            "reason": f"Possible translation artifacts: {flags}" if flags else "No translation signs"
        }

    def calculate_text_fraud_score(self, all_analyses):
        # If ≥2 components suspicious → suspicious
        suspicious_count = sum(1 for a in all_analyses.values() if a["verdict"] == "suspicious")
        return {
            "verdict": "suspicious" if suspicious_count >= 2 else "clean",
            "components": all_analyses
        }

def load_model(model_name):
    return DummyLLM()
