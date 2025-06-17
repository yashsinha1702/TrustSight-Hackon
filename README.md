# 🛡️ TrustSight — Counterfeit Detection Module

**This is the Counterfeit Detection submodule of TrustSight**, an AI-powered fraud detection system built for large-scale e-commerce platforms like Amazon.  
It detects counterfeit products using a multi-signal analysis pipeline combining image verification, pricing analysis, seller legitimacy, and listing text inspection.

---

## 📌 Scope

> 🔍 This module detects **counterfeit listings** only.  
> Modules for **review fraud**, **seller networks**, and **listing manipulation** are part of the broader TrustSight Detection Layer (not included here).

---

## 📂 Folder Structure

Detection/
└── Counterfeit/
├── product_image_analyzer.py # Image signal analysis
├── listing_text_analyzer.py # Text-based listing fraud
├── price_analyzer.py # Price manipulation checker
├── seller_check.py # Official seller verification
├── logo_verifier.py # CLIP-based logo check
├── dummy_llm.py # Simulated LLM logic
├── detector.py # Main orchestration logic
├── test_script.py # CLI tester script
├── requirements.txt # GPU-based dependencies
├── training_efficientnet.py # EfficientNet trainer (optional)
├── init.py
└── pycache/ # Python cache (ignored)

yaml
Copy
Edit

---

## 🧠 Detection Techniques

| Signal | Methods Used |
|--------|--------------|
| **Images** | ✅ Reverse image reuse detection  
✅ CLIP similarity vs brand reference  
✅ CNN micro-authenticity classifier  
✅ AI-generated image detection  
✅ EXIF metadata inspection  
✅ Logo verification using CLIP |
| **Text** | ✅ Brand name frequency check  
✅ Suspicious keywords  
✅ LLM-based translation artifact detection |
| **Price** | ✅ Brand reference thresholding |
| **Seller** | ✅ Match against verified seller registry |

---

## ⚙️ Setup Instructions (GPU Recommended)

### 1. 📦 Install dependencies

```bash
cd Detection/Counterfeit/
pip install -r requirements.txt
This setup uses:

torch with CUDA 11.8

transformers, opencv-python, scikit-learn

2. 📁 Add Required Files
The following are required but excluded from GitHub. Download and place them properly:

Resource	Destination Folder	Description	Link
efficientnet_model.pth	models/	CNN weights	🔗 Download
Brand reference images	reference_images/	Official brand+category images	🔗 Download
Brand logos	brand_logos/	Logos for logo matching	🔗 Download
Test image set	final_images/	Product images (img_001.jpg...)	🔗 Download
Sample metadata	products.csv	Product dataset	🔗 Download

🚀 How to Run
Run on all entries from products.csv:
bash
Copy
Edit
python test_script.py
Sample Output:
yaml
Copy
Edit
----- Product #1 -----
Title      : Nike shoes - Official Edition
Verdict    : 58 / 100
Confidence : 100%
Evidence   : {
  'seller': {'verdict': 'unauthorized', ...},
  'image': {'verdict': 'suspicious', ...},
  ...
}
🧾 Feature Summary
✅ CLIP-based visual similarity
✅ EfficientNet image classifier
✅ AI-generated image detection
✅ Logo match check using brand logos
✅ Spam keyword & brand mention text logic
✅ Translation artifact detector
✅ Price manipulation detection
✅ Seller validation using brand registry
✅ Unified fraud score (0–100) with confidence
✅ Modular code — ready for upstream integration
