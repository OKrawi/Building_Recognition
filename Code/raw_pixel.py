# import libraries
import numpy as np
import cv2
import time
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score, accuracy_score
import warnings
import ctypes
from functions import get_y_label, get_progress, get_image_paths
warnings.filterwarnings('ignore')


# Function using raw pixel approach
def raw_pixel(images_paths, x_image_list, y_label_list, list_name):
    # Track progress and get total number of images
    progress = 0
    len_images_paths = len(images_paths)
    # Loop over images
    for n, image_path in enumerate(images_paths):
        # Read and resize images
        image = cv2.imread(image_path)
        image = cv2.resize(image, (300, 300))
        # Add raw images to the X set
        x_image_list.append(image.flatten())
        # Add the labels to the Y set
        y_label_list.append(get_y_label(image_path))
        # Update and print progress
        progress = get_progress(list_name, progress, n, len_images_paths)
    print("\n")


print("Starting the Raw Pixel approach:\n")

# Track time
t0 = time.time()

[train_images_paths, test_images_paths] = get_image_paths()

# create empty lists to hold the images being read
X_train_image = []
X_test_image = []
y_train_label = []
y_test_label = []

# Call the function for both training and test images
raw_pixel(train_images_paths, X_train_image, y_train_label, "Train Data")
raw_pixel(test_images_paths, X_test_image, y_test_label, "Test Data")

# convert lists to numpy array and print their shapes
X_train = np.array(X_train_image)
X_test = np.array(X_test_image)
print("Shape of X_train is: {}".format(X_train.shape))
print("Shape of X_test is: {}".format(X_test.shape))

y_train = np.array(y_train_label)
y_test = np.array(y_test_label)
print("Shape of y_train is: {}".format(y_train.shape))
print("Shape of y_test is: {}".format(y_test.shape))

# Train the model
rfc_1 = RandomForestClassifier()
rfc_1.fit(X_train, y_train)
# Evaluate the model with test data
y_pred = rfc_1.predict(X_test)

# Get result metrics, and time and print them
acc = round(accuracy_score(y_test, y_pred) * 100, 2)
precision = round(precision_score(y_test, y_pred, average='weighted') * 100, 2)

t1 = time.time()
work_time = t1 - t0

print("Results for the Raw Pixel approach:\n")
print('\nAccuracy = {}%\nPrecision = {}%\n\nTime taken = {} sec\n'.format(acc, precision, work_time))

# Show message that work is finished
ctypes.windll.user32.MessageBoxW(0, "The code has finished.", "Finished!", 0)
