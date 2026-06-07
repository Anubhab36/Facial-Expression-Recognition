import torch
import torch.nn as nn
import torch.optim as optim

from torchvision import datasets
from torchvision import transforms
from torch.utils.data import DataLoader

from model import EmotionCNN


# Configuration

BATCH_SIZE = 16
EPOCHS = 3
LEARNING_RATE = 0.001

TRAIN_DIR = "dataset/images/train"
VAL_DIR = "dataset/images/validation"


# Transforms

transform = transforms.Compose([
    transforms.Resize((96, 96)),
    transforms.ToTensor()
])


print("Loading datasets...")

train_dataset = datasets.ImageFolder(
    TRAIN_DIR,
    transform=transform
)

val_dataset = datasets.ImageFolder(
    VAL_DIR,
    transform=transform
)

print(f"Training Images: {len(train_dataset)}")
print(f"Validation Images: {len(val_dataset)}")


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


device = torch.device("cpu")

print(f"Using device: {device}")

model = EmotionCNN().to(device)

criterion = nn.CrossEntropyLoss()

optimizer = optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE
)

print("Starting training...")


for epoch in range(EPOCHS):

    model.train()

    running_loss = 0.0

    for batch_idx, (images, labels) in enumerate(train_loader):

        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(images)

        loss = criterion(outputs, labels)

        loss.backward()

        optimizer.step()

        running_loss += loss.item()

        if batch_idx % 100 == 0:

            print(
                f"Epoch {epoch + 1}/{EPOCHS} | "
                f"Batch {batch_idx}/{len(train_loader)} | "
                f"Loss: {loss.item():.4f}"
            )

    avg_loss = running_loss / len(train_loader)

    print(
        f"\nEpoch {epoch + 1} Training Loss: "
        f"{avg_loss:.4f}"
    )

    model.eval()

    correct = 0
    total = 0

    with torch.no_grad():

        for images, labels in val_loader:

            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)

            _, predicted = torch.max(
                outputs.data,
                1
            )

            total += labels.size(0)

            correct += (
                predicted == labels
            ).sum().item()

    accuracy = (
        100 * correct / total
    )

    print(
        f"Validation Accuracy: "
        f"{accuracy:.2f}%\n"
    )


torch.save(
    model.state_dict(),
    "models/emotion_cnn.pth"
)

print("Training Complete")
print("Model saved to models/emotion_cnn.pth")