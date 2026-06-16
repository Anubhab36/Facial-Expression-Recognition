import os
import random
import shutil

SOURCE_DIR = "final_dataset/train"
VALIDATION_DIR = "final_dataset/validation"

SPLIT_RATIO = 0.2

random.seed(42)

classes = [
    "pain",
    "distress",
    "normal"
]

for class_name in classes:

    source_folder = os.path.join(
        SOURCE_DIR,
        class_name
    )

    validation_folder = os.path.join(
        VALIDATION_DIR,
        class_name
    )

    os.makedirs(
        validation_folder,
        exist_ok=True
    )

    images = [

        img for img in os.listdir(source_folder)

        if os.path.isfile(
            os.path.join(source_folder, img)
        )

    ]

    random.shuffle(images)

    validation_count = int(
        len(images) * SPLIT_RATIO
    )

    validation_images = images[:validation_count]

    for image in validation_images:

        src = os.path.join(
            source_folder,
            image
        )

        dst = os.path.join(
            validation_folder,
            image
        )

        shutil.move(src, dst)

    print(
        f"{class_name}: "
        f"{len(images)-validation_count} train | "
        f"{validation_count} validation"
    )

print("\nDataset split completed.")