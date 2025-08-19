# =========================================
#  Import libraries
# =========================================
import os, torch, numpy as np
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.utils import save_image   # for saving predictions as PNGs

# =========================================
#  Data preparation
# =========================================
# Transformations for training (augment + normalize)
train_tfms = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(8),
    transforms.ToTensor()                 # convert to tensor [0,1]
])

# Test set uses only basic ToTensor
test_tfms = transforms.ToTensor()

# Load FashionMNIST training & test datasets
train_ds = datasets.FashionMNIST(root="data", train=True, download=True, transform=train_tfms)
test_ds  = datasets.FashionMNIST(root="data", train=False, download=True, transform=test_tfms)

# Data loaders (batch the data for training/testing)
train_loader = DataLoader(train_ds, batch_size=128, shuffle=True, num_workers=0)
test_loader  = DataLoader(test_ds, batch_size=256, shuffle=False, num_workers=0)

# Human-readable class labels
CLASS_NAMES = [
    "T-shirt/top","Trouser","Pullover","Dress","Coat",
    "Sandal","Shirt","Sneaker","Bag","Ankle boot"
]

# =========================================
#  Model definition (CNN)
# =========================================

class FashionCNN(nn.Module):
    def __init__(self):
        super().__init__()

        # Convolutional layers (feature extractor)
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.Conv2d(32, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.MaxPool2d(2), nn.Dropout(0.25),

            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.MaxPool2d(2), nn.Dropout(0.25),
        )

        # Classifier (fully connected layers)
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),   # shrink to (B,64,1,1)
            nn.Flatten(),              # flatten to (B,64)
            nn.Linear(64, 128), nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(128, 10)         # 10 output classes
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

# =========================================
#  Training + Evaluation loop
# =========================================

if __name__ == "__main__":
    # Use GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = FashionCNN().to(device)

    # Loss function & optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    best_acc = 0.0
    for epoch in range(1, 21):  # up to 20 epochs
        # ----- Training -----
        model.train()
        running = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()
            running += loss.item() * xb.size(0)
        train_loss = running / len(train_loader.dataset)

        # ----- Evaluation -----
        model.eval()
        correct = total = 0
        with torch.no_grad():
            for xb, yb in test_loader:
                xb, yb = xb.to(device), yb.to(device)
                pred = model(xb).argmax(1)
                correct += (pred == yb).sum().item()
                total += yb.size(0)
        test_acc = correct / total
        print(f"Epoch {epoch:02d} | train_loss {train_loss:.4f} | test_acc {test_acc:.4f}")

        # Save best model checkpoint
        if test_acc > best_acc:
            best_acc = test_acc
            os.makedirs("models", exist_ok=True)
            torch.save(model.state_dict(), "models/fashion_cnn.pt")
            print(f"  Saved checkpoint (acc={best_acc:.4f})")

    print(f"Best Test Accuracy: {best_acc:.4f}")

    # =========================================
    #  Export predictions as PNG
    # =========================================
    os.makedirs("predicted_images", exist_ok=True)

    model.eval()
    with torch.no_grad():
        for i, (xb, yb) in enumerate(test_loader):
            xb, yb = xb.to(device), yb.to(device)
            preds = model(xb).argmax(1)

            # Save first 20 predictions as PNG
            for j in range(min(20, len(xb))):
                label = CLASS_NAMES[preds[j]]
                filename = f"predicted_images/{label}_{i*len(xb)+j}.png"
                save_image(xb[j], filename)

            break   # only first batch for demo

    print(" Export complete! PNGs saved in 'predicted_images/'")
