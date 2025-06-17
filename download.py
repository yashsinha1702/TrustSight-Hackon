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
