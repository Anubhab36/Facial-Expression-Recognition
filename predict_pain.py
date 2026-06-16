import sys
import torch
from PIL import Image
from torchvision import transforms

from model_pain import PainCNN

# ==========================
# Class Names
# ==========================

CLASS_NAMES = [
    "Distress",
    "Normal",
    "Pain"
]

# ==========================
# Device
# ==========================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

# ==========================
# Load Model
# ==========================

model = PainCNN(num_classes=3)

model.load_state_dict(
    torch.load(
        "models/pain_distress_cnn.pth",
        map_location=device
    )
)

model.to(device)

model.eval()

# ==========================
# Image Transform
# ==========================

transform = transforms.Compose([

    transforms.Resize((64, 64)),

    transforms.ToTensor()

])

# ==========================
# Check Input
# ==========================

if len(sys.argv) != 2:

    print("Usage:")
    print("python predict_pain.py image.jpg")

    sys.exit()

image_path = sys.argv[1]

# ==========================
# Load Image
# ==========================

image = Image.open(
    image_path
).convert("RGB")

image = transform(image)

image = image.unsqueeze(0).to(device)

# ==========================
# Prediction
# ==========================

with torch.no_grad():

    outputs = model(image)

    probabilities = torch.softmax(
        outputs,
        dim=1
    )

    confidence, prediction = torch.max(
        probabilities,
        dim=1
    )

predicted_class = CLASS_NAMES[
    prediction.item()
]

confidence = confidence.item() * 100

print("\n==============================")
print("Pain Classification Result")
print("==============================")
print(f"Prediction : {predicted_class}")
print(f"Confidence : {confidence:.2f}%")
print("==============================")