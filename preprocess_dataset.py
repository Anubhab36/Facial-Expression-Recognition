import os
import cv2

INPUT_DIR = "processed_dataset/train"
OUTPUT_DIR = "final_dataset/train"

IMAGE_SIZE = 48

classes = [
    "pain",
    "distress",
    "normal"
]

for class_name in classes:

    input_folder = os.path.join(
        INPUT_DIR,
        class_name
    )

    output_folder = os.path.join(
        OUTPUT_DIR,
        class_name
    )

    os.makedirs(
        output_folder,
        exist_ok=True
    )

    print(f"\nProcessing {class_name}...")

    processed = 0

    for filename in os.listdir(input_folder):

        input_path = os.path.join(
            input_folder,
            filename
        )

        image = cv2.imread(
            input_path
        )

        if image is None:
            continue

        resized = cv2.resize(
            image,
            (64, 64)
        )

        output_path = os.path.join(
            output_folder,
            filename
        )

        cv2.imwrite(
            output_path,
            resized
        )

        processed += 1

    print(
        f"Processed {processed} images"
    )

print("\n==========================")
print("Dataset preprocessing complete.")