# -------------------------
# File: seller_check.py
# -------------------------

class SellerAuthorityChecker:
    def __init__(self):
        self.authorized_sellers = {
            "Nike": ["Nike Official Store"],
            "Adidas": ["Adidas Official Store"],
            "Apple": ["Apple Official Store", "Apple Online Store"],
            "Puma": ["Puma Official Store", "Puma India"],
            "Samsung": ["Samsung Official Store", "Samsung Authorized Dealer"]
        }

    def extract_brand(self, title):
        # Assumes title is like: "Nike shoes - Official Edition"
        return title.split()[0].strip()

    def verify(self, product):
        title = product.get("title", "")
        seller_name = product.get("seller_name", "")

        brand = self.extract_brand(title)
        authorized_list = self.authorized_sellers.get(brand, [])

        if seller_name in authorized_list:
            return {
                "verdict": "authorized",
                "brand": brand,
                "seller": seller_name,
                "reason": f"{seller_name} is an authorized seller for {brand}"
            }
        else:
            return {
                "verdict": "unauthorized",
                "brand": brand,
                "seller": seller_name,
                "reason": f"{seller_name} is not listed as an authorized seller for {brand}"
            }
