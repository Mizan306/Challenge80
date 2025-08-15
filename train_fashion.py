import os, torch, numpy as np
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# ----- Data -----
train_tfms = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(8),
    transforms.ToTensor()                 # [0,1], shape (1,28,28)
])
test_tfms = transforms.ToTensor()

train_ds = datasets.FashionMNIST(root="data", train=True, download=True, transform=train_tfms)
test_ds  = datasets.FashionMNIST(root="data", train=False, download=True, transform=test_tfms)

# Use num_workers=0 for Windows safety
train_loader = DataLoader(train_ds, batch_size=128, shuffle=True, num_workers=0)
test_loader  = DataLoader(test_ds,  batch_size=256, shuffle=False, num_workers=0)

CLASS_NAMES = [
    "T-shirt/top","Trouser","Pullover","Dress","Coat",
    "Sandal","Shirt","Sneaker","Bag","Ankle boot"
]

# ----- Model -----
class FashionCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.Conv2d(32, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.MaxPool2d(2), nn.Dropout(0.25),

            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.MaxPool2d(2), nn.Dropout(0.25),
        )
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),          # (B,64,1,1)
            nn.Flatten(),                     # (B,64)
            nn.Linear(64, 128), nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(128, 10)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = FashionCNN().to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # ----- Train -----
    best_acc = 0.0
    for epoch in range(1, 21):  # up to ~20 epochs; stop earlier if you like
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

        # eval
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

        if test_acc > best_acc:
            best_acc = test_acc
            os.makedirs("models", exist_ok=True)
            torch.save(model.state_dict(), "models/fashion_cnn.pt")
            print(f"  ✅ Saved checkpoint (acc={best_acc:.4f})")

    print(f"Best Test Accuracy: {best_acc:.4f}")
