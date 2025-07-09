import json
import random
import datetime
from faker import Faker

fake = Faker()

def random_date(start, end):
    return start + datetime.timedelta(seconds=random.randint(0, int((end - start).total_seconds())))

def generate_seller_id(fraud_type, index):
    prefix = "S_LEGIT" if fraud_type is None else f"S_FRAUD_{fraud_type.upper()}"
    return f"{prefix}_{str(index).zfill(6)}"

def generate_seller_profile(index, fraud_type):
    today = datetime.datetime(2025, 6, 15)
    reg_date = random_date(datetime.datetime(2023, 1, 1), today)
    address = fake.address().split('\n')
    business_address = {
        "street": address[0],
        "city": fake.city(),
        "state": fake.state_abbr(),
        "zip": fake.zipcode(),
        "country": "US",
        "latitude": float(fake.latitude()),
        "longitude": float(fake.longitude())
    }

    categories = random.sample(["Electronics", "Home & Kitchen", "Sports", "Books", "Toys", "Clothing"], k=3)
    brands = {"Nike": random.randint(20, 200), "Adidas": random.randint(20, 200), "Generic": random.randint(300, 800)}

    return {
        "seller_id": generate_seller_id(fraud_type, index),
        "business_name": fake.company(),
        "display_name": fake.company_suffix() + "_" + fake.country_code(),
        "registration_date": reg_date.isoformat() + "Z",
        "account_age_days": (today - reg_date).days,
        "business_address": business_address,

        "seller_metrics": {
            "total_products": random.randint(100, 1000),
            "active_products": random.randint(50, 900),
            "avg_product_price": round(random.uniform(5, 100), 2),
            "total_reviews": random.randint(100, 20000),
            "avg_rating": round(random.uniform(2.5, 5.0), 1),
            "response_time_hours": round(random.uniform(1, 24), 1),
            "fulfillment_rate": round(random.uniform(0.8, 1.0), 2),
            "return_rate": round(random.uniform(0.01, 0.2), 2),
            "customer_complaints": random.randint(0, 300)
        },

        "product_catalog_fingerprint": {
            "categories": categories,
            "brand_distribution": brands,
            "price_range": [round(random.uniform(5, 30), 2), round(random.uniform(100, 500), 2)],
            "common_keywords": random.sample(["premium", "fast shipping", "best quality", "limited offer", "discount"], 3),
            "listing_templates_used": random.randint(1, 5),
            "unique_products": random.randint(100, 600),
            "duplicated_listings": random.randint(0, 400)
        },

        "pricing_behavior": {
            "avg_price_change_frequency_days": round(random.uniform(1, 10), 1),
            "max_price_drop_percent": round(random.uniform(5, 70), 1),
            "synchronized_changes_count": random.randint(0, 50),
            "competitor_price_matching": random.choice([True, False]),
            "dynamic_pricing_detected": random.choice([True, False])
        },

        "inventory_patterns": {
            "avg_stock_level": random.randint(100, 1000),
            "max_stock_spike": random.randint(1000, 10000),
            "stock_spike_date": random_date(datetime.datetime(2023, 1, 1), today).isoformat() + "Z",
            "out_of_stock_frequency": round(random.uniform(0.0, 0.3), 2),
            "inventory_turnover_days": random.randint(5, 60)
        },

        "network_features": {
            "shared_products_with_sellers": [],
            "shared_product_count": 0,
            "registration_cluster_id": None,
            "same_day_registrations": [],
            "address_similarity_score": round(random.uniform(0.0, 1.0), 2),
            "customer_overlap_score": round(random.uniform(0.0, 1.0), 2),
            "reviewer_overlap_score": round(random.uniform(0.0, 1.0), 2)
        },

        "temporal_features": {
            "registration_hour": reg_date.hour,
            "registration_day_of_week": reg_date.strftime('%A'),
            "days_since_last_activity": random.randint(0, 30),
            "activity_gaps": [random.randint(1, 30) for _ in range(3)],
            "peak_activity_hours": random.sample(range(0, 24), k=5)
        },

        "financial_indicators": {
            "revenue_growth_rate": round(random.uniform(-1, 5), 2),
            "sudden_discount_rate": round(random.uniform(0, 0.5), 2),
            "payment_method_changes": random.randint(0, 5),
            "payout_frequency_days": random.choice([7, 14, 30]),
            "unusual_refund_spike": random.choice([True, False])
        },

        "labels": {
            "is_fraud": int(fraud_type is not None),
            "fraud_type": fraud_type if fraud_type else None,
            "network_role": "hub" if fraud_type == "network" and random.random() < 0.1 else "member",
            "risk_score": round(random.uniform(0.0, 1.0), 2) if fraud_type else 0,
            "specific_patterns": {
                "is_network_member": int(fraud_type == "network"),
                "has_price_coordination": int(fraud_type in ["price_coordination", "mixed"]),
                "has_inventory_sharing": int(fraud_type == "network"),
                "has_registration_cluster": int(fraud_type == "network"),
                "exit_scam_risk": int(fraud_type in ["exit_scam", "mixed"])
            }
        },

        "network_id": f"NETWORK_{random.randint(1, 200)}" if fraud_type == "network" else None,
        "connected_seller_count": random.randint(5, 200) if fraud_type == "network" else 0,
        "network_value_usd": random.randint(50000, 5000000) if fraud_type else None
    }

def generate_dataset():
    dataset = []
    current_index = 0
    fraud_types = {
        None: 60000,
        "network": 15000,
        "price_coordination": 10000,
        "exit_scam": 8000,
        "mixed": 7000
    }

    for fraud_type, count in fraud_types.items():
        for _ in range(count):
            dataset.append(generate_seller_profile(current_index, fraud_type))
            current_index += 1

    with open("sellers_dataset_100k.json", "w") as f:
        json.dump(dataset, f, indent=2)

    print("✅ Dataset generated and saved to sellers_dataset_100k.json")

# Run it
generate_dataset()
