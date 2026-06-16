import os
from PIL import Image

# Dataset directory
DATASET_DIR = "pain_dataset/train"

# Minimum acceptable image size
MIN_WIDTH = 100
MIN_HEIGHT = 100

removed = 0
checked = 0


def clean_folder(folder):

    global removed
    global checked

    for filename in os.listdir(folder):

        filepath = os.path.join(folder, filename)

        if not os.path.isfile(filepath):
            continue

        checked += 1

        try:

            with Image.open(filepath) as img:

                img.verify()

            with Image.open(filepath) as img:

                width, height = img.size

                if width < MIN_WIDTH or height < MIN_HEIGHT:

                    os.remove(filepath)

                    removed += 1

                    print(
                        f"Removed tiny image: {filename}"
                    )

        except Exception:

            try:

                os.remove(filepath)

                removed += 1

                print(
                    f"Removed corrupt image: {filename}"
                )

            except:
                pass


for class_name in [

    "pain",

    "distress",

    "normal"

]:

    folder = os.path.join(

        DATASET_DIR,

        class_name

    )

    print(

        f"\nCleaning {class_name}..."

    )

    clean_folder(folder)

print("\n==========================")

print(

    f"Checked : {checked}"

)

print(

    f"Removed : {removed}"

)

print(

    f"Remaining : {checked - removed}"

)