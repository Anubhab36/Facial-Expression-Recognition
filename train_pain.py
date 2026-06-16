import os
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

from torchvision import datasets
from torchvision import transforms
from torch.utils.data import DataLoader

from model_pain import PainCNN

# ==========================
# Configuration
# ==========================

TRAIN_DIR = "final_dataset/train"
VAL_DIR = "final_dataset/validation"

BATCH_SIZE = 32
EPOCHS = 30
LEARNING_RATE = 0.0005

MODEL_PATH = "models/pain_distress_cnn.pth"

os.makedirs("models", exist_ok=True)

# ==========================
# Data Augmentation
# ==========================

train_transform = transforms.Compose([

    transforms.Resize((64, 64)),

    transforms.RandomHorizontalFlip(p=0.5),

    transforms.RandomRotation(10),

    transforms.RandomAffine(
        degrees=0,
        translate=(0.1, 0.1)
    ),

    transforms.ColorJitter(
        brightness=0.2,
        contrast=0.2
    ),

    transforms.ToTensor(),

])

val_transform = transforms.Compose([

    transforms.Resize((64, 64)),

    transforms.ToTensor(),

])

# ==========================
# Load Dataset
# ==========================

print("Loading dataset...")

train_dataset = datasets.ImageFolder(
    TRAIN_DIR,
    transform=train_transform
)

val_dataset = datasets.ImageFolder(
    VAL_DIR,
    transform=val_transform
)

print("\nClasses Found:")

print(train_dataset.classes)

print("\nTraining Images:", len(train_dataset))

print("Validation Images:", len(val_dataset))

train_loader = DataLoader(

    train_dataset,

    batch_size=BATCH_SIZE,

    shuffle=True,

    num_workers=0

)

val_loader = DataLoader(

    val_dataset,

    batch_size=BATCH_SIZE,

    shuffle=False,

    num_workers=0

)

# ==========================
# Device
# ==========================

device = torch.device(

    "cuda"

    if torch.cuda.is_available()

    else "cpu"

)

print("\nUsing device:", device)

# ==========================
# Model
# ==========================

model = PainCNN(

    num_classes=3

).to(device)

criterion = nn.CrossEntropyLoss()

optimizer = optim.Adam(

    model.parameters(),

    lr=LEARNING_RATE

)

scheduler = optim.lr_scheduler.ReduceLROnPlateau(

    optimizer,

    mode="max",

    factor=0.5,

    patience=3

)

print("\nModel initialized successfully.")

print(model)

# ==========================
# Variables for Training
# ==========================

best_accuracy = 0.0

train_loss_history = []

val_accuracy_history = []

print("\nReady to start training...")

# ==========================
# Training Loop
# ==========================

PATIENCE = 7
patience_counter = 0

for epoch in range(EPOCHS):

    print("\n" + "=" * 60)
    print(f"Epoch {epoch + 1}/{EPOCHS}")

    #############################
    # TRAINING
    #############################

    model.train()

    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in train_loader:

        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(images)

        loss = criterion(outputs, labels)

        loss.backward()

        optimizer.step()

        running_loss += loss.item()

        _, predicted = torch.max(outputs, 1)

        total += labels.size(0)

        correct += (predicted == labels).sum().item()

    train_loss = running_loss / len(train_loader)

    train_accuracy = 100 * correct / total

    train_loss_history.append(train_loss)

    #############################
    # VALIDATION
    #############################

    model.eval()

    correct = 0
    total = 0

    with torch.no_grad():

        for images, labels in val_loader:

            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)

            _, predicted = torch.max(outputs, 1)

            total += labels.size(0)

            correct += (predicted == labels).sum().item()

    val_accuracy = 100 * correct / total

    val_accuracy_history.append(val_accuracy)

    #############################
    # Scheduler
    #############################

    scheduler.step(val_accuracy)

    #############################
    # Print Statistics
    #############################

    current_lr = optimizer.param_groups[0]["lr"]

    print(f"Training Loss      : {train_loss:.4f}")
    print(f"Training Accuracy  : {train_accuracy:.2f}%")
    print(f"Validation Accuracy: {val_accuracy:.2f}%")
    print(f"Learning Rate      : {current_lr:.6f}")

    #############################
    # Save Best Model
    #############################

    if val_accuracy > best_accuracy:

        best_accuracy = val_accuracy

        patience_counter = 0

        torch.save(

            model.state_dict(),

            MODEL_PATH

        )

        print("\n✅ Best model saved!")

    else:

        patience_counter += 1

        print(

            f"No improvement "

            f"({patience_counter}/{PATIENCE})"

        )

    #############################
    # Early Stopping
    #############################

    if patience_counter >= PATIENCE:

        print("\nEarly stopping triggered.")

        break

print("\nTraining Finished.")

print(

    f"\nBest Validation Accuracy: "

    f"{best_accuracy:.2f}%"

)

# =====================================
# Save Training Graphs
# =====================================

os.makedirs("reports", exist_ok=True)

# -------- Training Loss --------

plt.figure(figsize=(8, 5))

plt.plot(
    train_loss_history,
    marker="o",
    linewidth=2
)

plt.title("Training Loss")

plt.xlabel("Epoch")

plt.ylabel("Loss")

plt.grid(True)

plt.tight_layout()

plt.savefig("reports/training_loss.png")

plt.close()

# -------- Validation Accuracy --------

plt.figure(figsize=(8, 5))

plt.plot(
    val_accuracy_history,
    marker="o",
    linewidth=2
)

plt.title("Validation Accuracy")

plt.xlabel("Epoch")

plt.ylabel("Accuracy (%)")

plt.grid(True)

plt.tight_layout()

plt.savefig("reports/validation_accuracy.png")

plt.close()

print("\nTraining graphs saved successfully.")

print("reports/training_loss.png")

print("reports/validation_accuracy.png")