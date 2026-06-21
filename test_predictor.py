import cv2

from video.predictor import EmotionPredictor

predictor = EmotionPredictor()

image = cv2.imread(
    "dataset/images/validation/neutral/10048.jpg"
)

emotion, confidence = predictor.predict(image)

print("Emotion :", emotion)
print(f"Confidence : {confidence:.2f}%")