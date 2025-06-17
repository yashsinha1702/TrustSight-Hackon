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
