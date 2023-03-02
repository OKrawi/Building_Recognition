import os
import sys
from pathlib import Path


# Get train and test image paths
def get_image_paths():
    # Step 1 - Get image paths and labels
    main_dir = os.path.dirname(sys.path[0])

    # location holding the directories with images
    train_images_location = os.path.join(main_dir, 'panoramas\\Train')
    test_images_location = os.path.join(main_dir, 'panoramas\\_Test')

    # Check the main directories and exist if either doesn't exist
    train_path = Path(train_images_location)
    test_path = Path(test_images_location)

    if not train_path.exists():
        sys.exit("Train image not available in path: {}".format(train_images_location))

    if not test_path.exists():
        sys.exit("Test image not available in path: {}".format(test_images_location))

    # Preparing the dataset
    train_images_paths = []
    for file in os.listdir(train_images_location):
        train_images_paths.append(os.path.join(train_images_location, file))

    # Preparing the dataset
    test_images_paths = []
    for file in os.listdir(test_images_location):
        test_images_paths.append(os.path.join(test_images_location, file))

    return train_images_paths, test_images_paths


# Get y label for each image
def get_y_label(image_path):
    y_label = image_path.split("panoramas\\")[1]
    y_label = y_label[6:].split("_")[0]
    y_label = y_label.split(".jpg")[0]
    lat = y_label.split(",")[0][:7]
    lng = y_label.split(",")[1][:7]
    y_label = lat + "," + lng
    return y_label


# Get and print progress
def get_progress(list_name, progress, n, len_images_paths):
    new_progress = int((n + 1) * 100 / len_images_paths)
    if new_progress > (progress + 4) or new_progress == 100:
        print("Work progress on {}: {}%".format(list_name, new_progress))
        return new_progress
    return progress
