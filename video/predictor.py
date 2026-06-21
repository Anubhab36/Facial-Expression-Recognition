import os

os.environ["OMP_NUM_THREADS"] = "1"

import torch
import warnings

warnings.filterwarnings("ignore")

from torchvision import transforms
from PIL import Image
import cv2

from model import EmotionCNN


CLASS_NAMES = [
    "Angry",
    "Disgust",
    "Fear",
    "Happy",
    "Neutral",
    "Sad",
    "Surprise"
]


class EmotionPredictor:

    def __init__(self):

        self.device = torch.device("cpu")

        self.model = EmotionCNN()

        self.model.load_state_dict(
            torch.load(
                "models/emotion_cnn.pth",
                map_location=self.device
            )
        )

        self.model.eval()

        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((64, 64)),
            transforms.ToTensor()
        ])

    def predict(self, face):

        rgb = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)

        image = self.transform(rgb)

        image = image.unsqueeze(0)

        with torch.no_grad():

            outputs = self.model(image)

            probabilities = torch.softmax(outputs, dim=1)

            confidence, prediction = torch.max(
                probabilities,
                dim=1
            )

        emotion = CLASS_NAMES[prediction.item()]

        confidence = confidence.item() * 100

        return emotion, confidence