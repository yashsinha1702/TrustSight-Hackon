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
