import csv
from detector import HybridCounterfeitDetector

detector = HybridCounterfeitDetector()

# Load CSV
with open("products.csv", newline='', encoding='utf-8') as csvfile:
    reader = csv.DictReader(csvfile)
    for i, row in enumerate(reader):
        # Preprocess for image fields
        row["images"] = [row["image_1"]]  # required by image analyzer
        row["brand_fuzzy"] = row["title"].split()[0]
        row["brand_raw"] = row["title"].split()[0]

        result = detector.detect(row)

        print(f"----- Product #{i+1} -----")
        print(f"Title      : {row['title']}")
        print(f"Verdict    : {result['score']} / 100")
        print(f"Confidence : {result['confidence']}%")
        print(f"Evidence   : {result['evidence']}")
        print()
