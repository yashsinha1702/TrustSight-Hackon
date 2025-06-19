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

import os
import pandas as pd
from PIL import Image
import random
import shutil

# Load your CSV
df = pd.read_csv("products.csv")

# Define folders
SOURCE_FOLDER = "images"
DEST_FOLDER = "final_images"
os.makedirs(DEST_FOLDER, exist_ok=True)

# Get list of image filenames
image_files = os.listdir(SOURCE_FOLDER)

# Normalize filename mapping
def normalize_filename(name):
    base = os.path.splitext(name)[0]
    return base.lower().replace("-", " ").replace("_", " ").strip()

image_map = {normalize_filename(f): f for f in image_files}

# Normalize title to extract meaningful keywords
def extract_keywords(title):
    title = title.lower()
    ignore = {"buy", "now", "for", "cheap", "genuine", "official", "edition", "special", "offer", "hot", "deal"}
    tokens = [word for word in title.replace("-", " ").replace("!", "").replace(",", "").split() if word not in ignore]
    return set(tokens)


# Match title to image
def match_image(title_keywords):
    for norm_key, fname in image_map.items():
        filename_words = set(norm_key.split())
        # Check if at least 2 meaningful title keywords are in filename
        match_count = sum(1 for word in title_keywords if word in filename_words)
        if match_count >= 2:
            return fname
    return None


# Random rotation
def rotate_image(img):
    angle = random.choice([90, 180, 270])
    return img.rotate(angle)

# Process each row
unmatched = []

for _, row in df.iterrows():
    title_keywords = extract_keywords(row["title"])
    match = match_image(title_keywords)

    if match:
        try:
            img_path = os.path.join(SOURCE_FOLDER, match)
            with Image.open(img_path) as img:
                img = rotate_image(img).convert("RGB")  # Ensure it's in RGB
                dest_path = os.path.join(DEST_FOLDER, row["image_1"])
                img.save(dest_path, format="JPEG")
        except Exception as e:
            print(f"❌ Error processing {match} → {row['image_1']}: {e}")
    else:
        unmatched.append(row["title"])
        print(f"⚠️ No match found for: {row['title']}")

# Summary
print(f"\n✅ Completed with {len(unmatched)} unmatched entries.")
if unmatched:
    print("🔍 Titles with no match:")
    for t in unmatched:
        print(" -", t)

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
import os
from PIL import Image
import torch
from transformers import CLIPProcessor, CLIPModel
from sklearn.metrics.pairwise import cosine_similarity

class LogoVerifier:
    def __init__(self, logo_dir="brand_logos"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(self.device)
        self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        self.logo_dir = logo_dir
        self.logo_images = self._load_logo_images()

    def _load_logo_images(self):
        logos = {}
        for fname in os.listdir(self.logo_dir):
            if fname.lower().endswith((".jpg", ".png")):
                brand = os.path.splitext(fname)[0]
                img = Image.open(os.path.join(self.logo_dir, fname)).convert("RGB")
                logos[brand] = img
        return logos

    def verify_logo(self, product_image_path):
        try:
            prod_img = Image.open(product_image_path).convert("RGB")
            best_brand = None
            best_score = -1

            for brand, logo_img in self.logo_images.items():
                inputs = self.processor(images=[prod_img, logo_img], return_tensors="pt").to(self.device)
                with torch.no_grad():
                    features = self.model.get_image_features(**inputs)
                    prod_vec = features[0].unsqueeze(0).cpu().numpy()
                    logo_vec = features[1].unsqueeze(0).cpu().numpy()
                    sim = cosine_similarity(prod_vec, logo_vec)[0][0]
                    if sim > best_score:
                        best_score = sim
                        best_brand = brand

            verdict = "confirmed" if best_score >= 0.85 else "mismatch"
            return {
                "matched_logo": best_brand,
                "similarity": round(best_score, 3),
                "verdict": verdict
            }

        except Exception as e:
            return {
                "verdict": "error",
                "reason": str(e)
            }
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
# -------------------------
# File: product_image_analyzer.py
# -------------------------

import cv2
from logo_verifier import LogoVerifier

def calculate_sharpness(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    return cv2.Laplacian(image, cv2.CV_64F).var() if image is not None else 0

class ReverseImageSearchEngine:
    def __init__(self, image_to_sellers):
        self.image_to_sellers = image_to_sellers

    def trace_origins(self, product):
        img = product["image_1"]
        seller = product["seller_name"]

        reused_by_others = self.image_to_sellers.get(img, set()) - {seller}
        if reused_by_others:
            return {
                "verdict": "reused",
                "reason": f"Image '{img}' is reused by other sellers: {list(reused_by_others)}"
            }
        else:
            return {
                "verdict": "unique",
                "reason": f"Image '{img}' is unique to seller '{seller}'"
            }


import torch
import torchvision.transforms as transforms
from torchvision.models import efficientnet_b0
from PIL import Image
import os
from transformers import CLIPProcessor, CLIPModel
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

class MicroDetailAnalyzer:
    def __init__(self):
        # CNN Classifier
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.cnn_model = efficientnet_b0(pretrained=False, num_classes=2)
        self.cnn_model.load_state_dict(torch.load("models/efficientnet_model.pth", map_location=self.device))
        self.cnn_model.to(self.device)
        self.cnn_model.eval()

        self.cnn_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

        # CLIP
        self.clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(self.device)
        self.clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

    def deep_analyze(self, product):
        image_name = product["image_1"]
        brand = product["title"].split()[0]  # Extract brand name

        cnn_result = self.run_cnn_classifier(image_name)
        clip_result = self.run_clip_similarity(image_name, brand)

        final_verdict = "suspicious" if (
            cnn_result["verdict"] == "suspicious" or clip_result["verdict"] == "suspicious"
        ) else "clean"

        return {
            "verdict": final_verdict,
            "cnn": cnn_result,
            "clip": clip_result
        }

    def run_cnn_classifier(self, image_name):
        try:
            img_path = os.path.join("final_images", image_name)
            image = Image.open(img_path).convert("RGB")
            tensor = self.cnn_transform(image).unsqueeze(0).to(self.device)

            with torch.no_grad():
                output = self.cnn_model(tensor)
                probs = torch.softmax(output, dim=1).cpu().numpy()[0]
                pred = int(np.argmax(probs))
                label = "genuine" if pred == 1 else "fake"

            return {
                "verdict": "suspicious" if label == "fake" else "clean",
                "prediction": label,
                "confidence": round(probs[pred], 3)
            }

        except Exception as e:
            return {
                "verdict": "error",
                "reason": str(e)
            }

    def run_clip_similarity(self, image_name, title):
        try:
            tokens = title.split()
            if len(tokens) < 2:
                return {"verdict": "error", "reason": "Title must contain both brand and category."}
            
            brand = tokens[0]
            category = tokens[1]
            ref_base = f"{brand}_{category}"  # ✅ You missed this line
            ref_dir = "reference_images"
            available_refs = os.listdir(ref_dir)

            ref_file = None
            for ext in [".jpg", ".jpeg", ".png", ".webp"]:
                candidate = ref_base + ext
                if candidate in available_refs:
                    ref_file = candidate
                    break

            if not ref_file:
                return {
                    "verdict": "counterfeit",
                    "reason": f"No reference image found for {ref_base} with any valid extension. Likely a fake or unrecognized brand-category pair."
                }

            query_img = Image.open(os.path.join("final_images", image_name)).convert("RGB")
            reference_img = Image.open(os.path.join(ref_dir, ref_file)).convert("RGB")

            inputs = self.clip_processor(images=[query_img, reference_img], return_tensors="pt").to(self.device)

            with torch.no_grad():
                features = self.clip_model.get_image_features(**inputs)
                query_vec = features[0].unsqueeze(0).cpu().numpy()
                ref_vec = features[1].unsqueeze(0).cpu().numpy()

            sim_score = cosine_similarity(query_vec, ref_vec)[0][0]
            verdict = "suspicious" if sim_score < 0.85 else "clean"

            return {
                "verdict": verdict,
                "similarity": round(sim_score, 3),
                "reference_used": ref_file,
                "reason": "Low similarity to brand reference image" if verdict == "suspicious" else "Visually similar to official brand product"
            }

        except Exception as e:
            return {
                "verdict": "error",
                "reason": str(e)
            }



from transformers import AutoFeatureExtractor, AutoModelForImageClassification
from PIL import Image
import torch

from transformers import AutoImageProcessor, AutoModelForImageClassification
from PIL import Image
import torch
import os

class AIGeneratedImageDetector:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_name = "umm-maybe/AI-image-detector"
        self.processor = AutoImageProcessor.from_pretrained(self.model_name)
        self.model = AutoModelForImageClassification.from_pretrained(self.model_name).to(self.device)
        self.model.eval()

    def scan_for_ai(self, images):
        suspicious = []
        results = []

        for img_name in images:
            try:
                img_path = os.path.join("final_images", img_name)
                img = Image.open(img_path).convert("RGB")
                inputs = self.processor(images=img, return_tensors="pt").to(self.device)

                with torch.no_grad():
                    outputs = self.model(**inputs)
                    probs = torch.nn.functional.softmax(outputs.logits, dim=1)
                    prob_ai = probs[0][1].item()
                    prob_real = probs[0][0].item()

                    verdict = "ai-detected" if prob_ai > 0.7 else "clean"

                    result = {
                        "image": img_name,
                        "verdict": verdict,
                        "prob_ai": round(prob_ai, 3),
                        "prob_real": round(prob_real, 3)
                    }

                    results.append(result)
                    if verdict == "ai-detected":
                        suspicious.append(result)

            except Exception as e:
                results.append({
                    "image": img_name,
                    "verdict": "error",
                    "reason": str(e)
                })

        return {
            "verdict": "ai-detected" if suspicious else "clean",
            "details": results,
            "reason": f"{len(suspicious)} AI-generated image(s) detected" if suspicious else "All images look real"
        }



from PIL import Image
import os

class ImageMetadataForensics:
    def examine(self, images):
        suspicious = []
        for img_name in images:
            try:
                img_path = os.path.join("final_images", img_name)
                img = Image.open(img_path)
                exif = img._getexif()
                if not exif:
                    suspicious.append(img_name)
            except:
                suspicious.append(img_name)

        return {
            "verdict": "suspicious" if suspicious else "clean",
            "reason": f"No metadata found in: {suspicious}" if suspicious else "EXIF metadata looks fine"
        }


class ProductImageAnalyzer:
    def __init__(self):
        self.reverse_search = ReverseImageSearchEngine(image_to_sellers={})
        self.micro_analyzer = MicroDetailAnalyzer()
        self.ai_detector = AIGeneratedImageDetector()
        self.metadata_analyzer = ImageMetadataForensics()
        self.logo_verifier = LogoVerifier("brand_logos")

    def analyze_all_images(self, product):
        source_analysis = self.reverse_search.trace_origins(product)
        portfolio_analysis = self.analyze_image_portfolio(product)

        micro_analysis = None
        if portfolio_analysis.get("has_custom_photos"):
            micro_analysis = self.micro_analyzer.deep_analyze(product)

        ai_detection = self.ai_detector.scan_for_ai(product["images"])
        metadata_results = self.metadata_analyzer.examine(product["images"])

        logo_verification = self.analyze_logo_verification(
            product,
            product.get("brand_fuzzy", ""),
            product.get("brand_raw", "")
        )

        all_analyses = {
            "source": source_analysis,
            "portfolio": portfolio_analysis,
            "micro": micro_analysis,
            "ai": ai_detection,
            "metadata": metadata_results,
            "logo": logo_verification
        }


        return self.synthesize_image_verdict(all_analyses)

    def analyze_image_portfolio(self, product):
        image_name = product["image_1"]
        img_path = os.path.join("final_images", image_name)

        try:
            img = Image.open(img_path)
            width, height = img.size
            aspect = round(width / height, 2)
            sharpness = calculate_sharpness(img_path)

            is_perfect = width == height and sharpness > 1000

            return {
                "has_custom_photos": not is_perfect,
                "verdict": "stock-like" if is_perfect else "custom",
                "reason": "Very sharp, square image — likely studio shot" if is_perfect else "Image has custom traits"
            }

        except Exception as e:
            return {
                "has_custom_photos": False,
                "verdict": "unknown",
                "reason": f"Error loading image: {str(e)}"
            }


    def synthesize_image_verdict(self, analyses):
        red_flags = []

        for key, result in analyses.items():
            if result and result.get("verdict") in ["suspicious", "reused", "ai-detected", "stock-like", "logo-mismatch"]:
                red_flags.append(key)

        final_verdict = "suspicious" if len(red_flags) >= 2 else "clean"

        return {
            "verdict": final_verdict,
            "reason": f"Red flags in: {red_flags}" if red_flags else "All sub-checks passed",
            "components": analyses
        }


    def analyze_logo_verification(self, product, fuzzy_matched_brand, raw_brand):
        if raw_brand.lower() != fuzzy_matched_brand.lower():
            image_path = os.path.join("final_images", product["image_1"])
            result = self.logo_verifier.verify_logo(image_path)
            if result["verdict"] == "confirmed" and result["matched_logo"].lower() == fuzzy_matched_brand.lower():
                return {
                    "verdict": "logo-confirmed",
                    "reason": f"Fuzzy matched to {fuzzy_matched_brand}, logo also matched.",
                    "match": result
                }
            else:
                return {
                    "verdict": "logo-mismatch",
                    "reason": f"Fuzzy matched to {fuzzy_matched_brand}, but logo not matched (found: {result.get('matched_logo', 'none')})",
                    "match": result
                }
        else:
            return {
                "verdict": "not-required",
                "reason": "Exact brand match. Logo verification skipped."
            }


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
# Re-run necessary code due to kernel reset

import os
import torch
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split

# Custom Dataset
class ProductImageDataset(Dataset):
    def __init__(self, df, image_dir, transform=None):
        self.df = df.reset_index(drop=True)
        self.image_dir = image_dir
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        image_name = self.df.loc[idx, 'image_1']
        label = 1 if self.df.loc[idx, 'label'] == 'genuine' else 0
        image_path = os.path.join(self.image_dir, image_name)
        image = Image.open(image_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, label

# Load CSV
df = pd.read_csv("/mnt/c/users/Atharv/Desktop/Hackon/Counterfeit/products.csv")

# Image directory
image_dir = "final_images"

# Split data
# train_df, val_df = train_test_split(df, test_size=0.2, stratify=df['label'], random_state=42)
train_df = df.copy()

# Transforms
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Datasets and Loaders
# train_dataset = ProductImageDataset(train_df, image_dir, transform)
# val_dataset = ProductImageDataset(val_df, image_dir, transform)
# train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
# val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)

train_dataset = ProductImageDataset(train_df, image_dir, transform)
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)


# Model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:",device)
model = models.efficientnet_b0(pretrained=True)
model.classifier[1] = nn.Linear(model.classifier[1].in_features, 2)
model = model.to(device)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)

# Early stopping setup
best_val_loss = float('inf')
patience, patience_counter = 5, 0
num_epochs = 250
model_path = "models"

train_losses, val_losses = [], []

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * images.size(0)

    train_loss = running_loss / len(train_loader.dataset)
    train_losses.append(train_loss)
    
# Evaluate training accuracy
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Accuracy: {accuracy:.2f}%")


    # Manual early stopping based on training loss (optional logic)
    if epoch > 1 and abs(train_losses[-1] - train_losses[-2]) < 1e-4:
        patience_counter += 1
    else:
        patience_counter = 0

    if patience_counter >= patience:
        print("Early stopping triggered based on training loss plateau.")
        break


final_model_path = "models/efficientnet_model_final.pth"
os.makedirs(os.path.dirname(final_model_path), exist_ok=True)
torch.save(model.state_dict(), final_model_path)
print(f"✅ Final model saved to: {final_model_path}")
