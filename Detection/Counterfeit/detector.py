from product_image_analyzer import ProductImageAnalyzer
from listing_text_analyzer import ListingTextAnalyzer
from price_analyzer import PriceAnomalyDetector
from seller_check import SellerAuthorityChecker

class HybridCounterfeitDetector:
    def __init__(self):
        self.image_analyzer = ProductImageAnalyzer()
        self.text_analyzer = ListingTextAnalyzer()
        self.price_analyzer = PriceAnomalyDetector()
        self.seller_verifier = SellerAuthorityChecker()

    def detect(self, product):
        authority_check = self.seller_verifier.verify(product)
        image_results = self.image_analyzer.analyze_all_images(product)
        text_results = self.text_analyzer.analyze_listing(product)
        price_results = self.price_analyzer.check_pricing(product)

        all_results = {
            "seller": authority_check,
            "image": image_results,
            "text": text_results,
            "price": price_results,
        }

        return {
            "score": self.calculate_final_score(all_results),
            "confidence": self.calculate_confidence(all_results),
            "evidence": self.compile_evidence(all_results)
        }

    def calculate_final_score(self, results):
        score = 100
        penalties = {
            "image": 30,
            "text": 20,
            "price": 15,
            "seller": 35
        }

        for key, penalty in penalties.items():
            verdict = results[key].get("verdict")
            if verdict in ["suspicious", "unauthorized", "reused", "ai-detected", "stock-like", "logo-mismatch", "counterfeit"]:
                score -= penalty

        return max(0, score)

    def calculate_confidence(self, results):
        decisive = 0
        for result in results.values():
            if result.get("verdict") not in ["error", "unknown"]:
                decisive += 1
        return round((decisive / 4) * 100)

    def compile_evidence(self, results):
        evidence = {}
        for category, result in results.items():
            reason = result.get("reason", "No reason provided")
            evidence[category] = {
                "verdict": result.get("verdict"),
                "reason": reason
            }
        return evidence
