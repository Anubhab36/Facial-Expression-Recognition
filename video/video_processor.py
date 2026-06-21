import cv2

from video.detector import FaceDetector
from video.predictor import EmotionPredictor


class VideoProcessor:

    def __init__(self):

        self.detector = FaceDetector()

        self.predictor = EmotionPredictor()

    def process(
        self,
        input_video,
        output_video
    ):

        cap = cv2.VideoCapture(input_video)

        if not cap.isOpened():

            print("Error opening video.")

            return

        width = int(
            cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        )

        height = int(
            cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        )

        fps = cap.get(cv2.CAP_PROP_FPS)

        if fps <= 1:
            fps = 30

        writer = cv2.VideoWriter(
            output_video,
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps,
            (width, height)
        )

        frame_count = 0

        while True:

            ret, frame = cap.read()

            if not ret:
                break

            faces = self.detector.detect(frame)

            for (x, y, w, h) in faces:

                face = frame[
                    y:y+h,
                    x:x+w
                ]

                if face.size == 0:
                    continue

                emotion, confidence = \
                    self.predictor.predict(face)

                cv2.rectangle(
                    frame,
                    (x, y),
                    (x+w, y+h),
                    (0, 255, 0),
                    2
                )

                text = f"{emotion} ({confidence:.1f}%)"

                cv2.putText(
                    frame,
                    text,
                    (x, y-10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0,255,0),
                    2
                )

            writer.write(frame)

            frame_count += 1

            if frame_count % 100 == 0:

                print(
                    f"Processed {frame_count} frames..."
                )

        cap.release()

        writer.release()

        print("\nDone!")

        print(
            f"Saved as {output_video}"
        )