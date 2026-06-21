import cv2

from video.detector import FaceDetector

cap = cv2.VideoCapture("test_video.mp4")

detector = FaceDetector()

while True:

    ret, frame = cap.read()

    if not ret:
        break

    faces = detector.detect(frame)

    for (x, y, w, h) in faces:

        cv2.rectangle(
            frame,
            (x, y),
            (x + w, y + h),
            (0, 255, 0),
            2
        )

    cv2.imshow("Detector", frame)

    if cv2.waitKey(1) == ord("q"):
        break

cap.release()

cv2.destroyAllWindows()