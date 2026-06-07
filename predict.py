import sys

import torch
from PIL import Image
from torchvision import transforms

from model import EmotionCNN


CLASS_NAMES = [
    "angry",
    "disgust",
    "fear",
    "happy",
    "neutral",
    "sad",
    "surprise"
]


device = torch.device("cpu")

model = EmotionCNN()

model.load_state_dict(
    torch.load(
        "models/emotion_cnn.pth",
        map_location=device
    )
)

model.eval()


transform = transforms.Compose([
    transforms.Resize((96, 96)),
    transforms.ToTensor()
])


if len(sys.argv) != 2:

    print(
        "Usage: python predict.py image.jpg"
    )

    sys.exit()


image_path = sys.argv[1]

image = Image.open(
    image_path
).convert("RGB")

image = transform(image)

image = image.unsqueeze(0)

with torch.no_grad():

    output = model(image)

    prediction = torch.argmax(
        output,
        dim=1
    ).item()

emotion = CLASS_NAMES[prediction]

print(
    f"Predicted Emotion: {emotion}"
)