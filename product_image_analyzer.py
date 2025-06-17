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


