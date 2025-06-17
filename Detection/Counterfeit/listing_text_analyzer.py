# -------------------------
# File: listing_text_analyzer.py
# -------------------------

from dummy_llm import load_model

class KeywordManipulationDetector:
    def __init__(self):
        self.red_flags = ["100% real", "officially certified", "cheapest", "guaranteed", "limited offer", "hot deal"]

    def detect(self, description):
        description_lower = description.lower()
        flagged = [kw for kw in self.red_flags if kw in description_lower]
        return {
            "flags": flagged,
            "verdict": "suspicious" if flagged else "clean",
            "reason": f"Found manipulative phrases: {flagged}" if flagged else "No manipulative keywords found"
        }

class BrandLanguageValidator:
    def check_brand_voice(self, product):
        brand = product["title"].split()[0]
        description = product["description"].lower()

        # If brand name is not mentioned at all, or over-mentioned (spammy)
        mention_count = description.count(brand.lower())
        if mention_count == 0:
            return {
                "verdict": "suspicious",
                "reason": f"Brand name '{brand}' not mentioned in description"
            }
        elif mention_count > 3:
            return {
                "verdict": "suspicious",
                "reason": f"Brand name '{brand}' mentioned {mention_count} times — may be unnatural"
            }
        else:
            return {
                "verdict": "clean",
                "reason": f"Brand name mentioned {mention_count} time(s), looks okay"
            }

class ListingTextAnalyzer:
    def __init__(self):
        self.llm = load_model("amazon-fraud-bert")
        self.keyword_detector = KeywordManipulationDetector()
        self.brand_validator = BrandLanguageValidator()

    def analyze_listing(self, product):
        description_analysis = self.llm.analyze_description(product["description"])
        brand_analysis = self.brand_validator.check_brand_voice(product)
        translation_check = self.llm.detect_translated_content(product["description"])

        all_analyses = {
            "description": description_analysis,
            "brand": brand_analysis,
            "translation": translation_check
        }

        return self.llm.calculate_text_fraud_score(all_analyses)
