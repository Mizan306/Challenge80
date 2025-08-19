# ==========================================
# Imports (Libraries we need)
# ==========================================
from flask import Flask, request, jsonify, render_template
import torch, numpy as np
from torch import nn
from PIL import Image, ImageOps
import io, os


# ==========================================
# Class Names (FashionMNIST labels)
# ==========================================
CLASS_NAMES = [
    "T-shirt/top","Trouser","Pullover","Dress","Coat",
    "Sandal","Shirt","Sneaker","Bag","Ankle boot"
]


# ==========================================
# CNN Model Architecture (same as training)
# ==========================================
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
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(64, 128), nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(128, 10)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


# ==========================================
# Load Trained Model Weights
# ==========================================
def load_model():
    m = FashionCNN()
    m.load_state_dict(torch.load("models/fashion_cnn.pt", map_location="cpu"))
    m.eval()   # set model to evaluation mode
    return m

model = load_model()


# ==========================================
# Flask App Initialization
# ==========================================
app = Flask(__name__)


# ==========================================
# Image Preprocessing Function
# (resize, grayscale, normalize, invert if needed)
# ==========================================
def _resample_bilinear():
    try:
        return Image.Resampling.BILINEAR   # Pillow >=10
    except AttributeError:
        return Image.BILINEAR

def preprocess_image(file_bytes):
    """Return tensor shape (1,1,28,28), float32 in [0,1], foreground light."""
    img = Image.open(io.BytesIO(file_bytes)).convert("L")           # grayscale
    img = ImageOps.fit(img, (28, 28), method=_resample_bilinear())  # resize
    arr = np.array(img).astype("float32") / 255.0                   # normalize
    if arr.mean() > 0.5:                                            # invert if needed
        arr = 1.0 - arr
    tensor = torch.from_numpy(arr)[None, None, :, :]                # add batch + channel
    return tensor


# ==========================================
# Flask Routes
# ==========================================

# Home page
@app.get("/")
def index():
    return render_template("index.html")


# Prediction route (POST an image here)
@app.post("/predict")
def predict():


    # --- File validation ---
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400
    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "Empty filename"}), 400



    # --- Preprocess + Predict ---
    x = preprocess_image(file.read())
    with torch.no_grad():
        probs = torch.softmax(model(x), dim=1).numpy()[0]   # probabilities
    top = int(probs.argmax())                               # best class
    top3_idx = probs.argsort()[-3:][::-1]                   # top-3 classes
    top3 = [{"label": CLASS_NAMES[i], "prob": float(probs[i])} for i in top3_idx]



    # --- Return response ---
    return jsonify({"prediction": CLASS_NAMES[top],
                    "probability": float(probs[top]),
                    "top3": top3})


# Health check (test if server is alive)
@app.get("/health")
def health():
    return {"status": "ok"}


# ==========================================
# Run the Flask app
# ==========================================
if __name__ == "__main__":
    app.run(debug=True)
