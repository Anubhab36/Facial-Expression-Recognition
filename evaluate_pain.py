import os
import torch
import matplotlib.pyplot as plt

from torchvision import datasets
from torchvision import transforms
from torch.utils.data import DataLoader

from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    ConfusionMatrixDisplay,
)

from model_pain import PainCNN

# =====================================
# Configuration
# =====================================

MODEL_PATH = "models/pain_distress_cnn.pth"
VAL_DIR = "final_dataset/validation"

BATCH_SIZE = 32

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

# =====================================
# Dataset
# =====================================

transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor()
])

dataset = datasets.ImageFolder(
    VAL_DIR,
    transform=transform
)

loader = DataLoader(
    dataset,
    batch_size=BATCH_SIZE,
    shuffle=False
)

class_names = dataset.classes

print("Classes:", class_names)

# =====================================
# Load Model
# =====================================

model = PainCNN(num_classes=3)

model.load_state_dict(
    torch.load(
        MODEL_PATH,
        map_location=device
    )
)

model.to(device)

model.eval()

# =====================================
# Prediction
# =====================================

y_true = []
y_pred = []

with torch.no_grad():

    for images, labels in loader:

        images = images.to(device)

        outputs = model(images)

        _, predicted = torch.max(outputs, 1)

        y_true.extend(labels.numpy())

        y_pred.extend(
            predicted.cpu().numpy()
        )

# =====================================
# Results
# =====================================

print("\nClassification Report\n")

print(

    classification_report(

        y_true,

        y_pred,

        target_names=class_names

    )

)

cm = confusion_matrix(
    y_true,
    y_pred
)

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=class_names
)

fig, ax = plt.subplots(figsize=(6,6))

disp.plot(ax=ax)

plt.title("Pain Classification Confusion Matrix")

os.makedirs("reports", exist_ok=True)

plt.savefig("reports/confusion_matrix.png")

plt.show()

print("\nConfusion matrix saved to:")

print("reports/confusion_matrix.png")