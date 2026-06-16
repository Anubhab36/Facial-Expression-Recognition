import os
import cv2
import mediapipe as mp

INPUT_DIR = "pain_dataset/train"
OUTPUT_DIR = "processed_dataset/train"

# Initialize MediaPipe Face Detection
mp_face_detection = mp.solutions.face_detection
face_detector = mp_face_detection.FaceDetection(
    model_selection=1,
    min_detection_confidence=0.5
)

total_images = 0
saved_faces = 0
skipped_images = 0


def process_folder(class_name):
    global total_images
    global saved_faces
    global skipped_images

    input_folder = os.path.join(INPUT_DIR, class_name)
    output_folder = os.path.join(OUTPUT_DIR, class_name)

    os.makedirs(output_folder, exist_ok=True)

    print(f"\nProcessing {class_name}...")

    for filename in os.listdir(input_folder):

        input_path = os.path.join(input_folder, filename)

        if not os.path.isfile(input_path):
            continue

        image = cv2.imread(input_path)

        if image is None:
            skipped_images += 1
            continue

        total_images += 1

        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        results = face_detector.process(rgb)

        if results.detections is None:
            skipped_images += 1
            continue

        # Largest detected face
        detection = max(
            results.detections,
            key=lambda d: d.location_data.relative_bounding_box.width *
                          d.location_data.relative_bounding_box.height
        )

        bbox = detection.location_data.relative_bounding_box

        h, w, _ = image.shape

        x = int(bbox.xmin * w)
        y = int(bbox.ymin * h)
        bw = int(bbox.width * w)
        bh = int(bbox.height * h)

        # Add 20% padding
        pad_x = int(0.2 * bw)
        pad_y = int(0.2 * bh)

        x = max(0, x - pad_x)
        y = max(0, y - pad_y)

        x2 = min(w, x + bw + 2 * pad_x)
        y2 = min(h, y + bh + 2 * pad_y)

        face = image[y:y2, x:x2]

        if face.size == 0:
            skipped_images += 1
            continue

        output_path = os.path.join(output_folder, filename)

        cv2.imwrite(output_path, face)

        saved_faces += 1


for category in [
    "pain",
    "distress",
    "normal"
]:
    process_folder(category)

print("\n==============================")
print(f"Images Checked : {total_images}")
print(f"Faces Saved    : {saved_faces}")
print(f"Skipped        : {skipped_images}")