# -------------------------
# File: price_analyzer.py
# -------------------------

class PriceAnomalyDetector:
    def __init__(self, pricing_reference=None):
        self.reference_data = pricing_reference or self.load_real_reference()

    def load_real_reference(self):
        # Reference from genuine products in dataset (precomputed)
        return {
            "Adidas": 127.08,
            "Apple": 124.46,
            "Nike": 119.87,
            "Puma": 121.4,
            "Samsung": 127.26
        }

    def extract_brand(self, title):
        return title.split()[0].strip()

    def check_pricing(self, product):
        title = product.get("title", "")
        price = float(product.get("price", 0))
        mrp = float(product.get("mrp", 0))

        brand = self.extract_brand(title)
        base_price = self.reference_data.get(brand)

        if base_price is None:
            return {
                "verdict": "unknown",
                "reason": f"No reference pricing available for brand '{brand}'"
            }

        threshold = 0.75 * base_price
        if price < threshold:
            return {
                "verdict": "suspicious",
                "brand": brand,
                "price": price,
                "reference_price": base_price,
                "reason": f"Price (${price}) is too low compared to reference (${base_price})"
            }
        else:
            return {
                "verdict": "acceptable",
                "brand": brand,
                "price": price,
                "reference_price": base_price,
                "reason": "Price is within normal range"
            }
