# import libraries
import numpy as np
import cv2
import time
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score, accuracy_score, f1_score
import mahotas
import warnings
import ctypes
from functions import get_y_label, get_progress, get_image_paths
warnings.filterwarnings('ignore')


# Function using global features
def global_features(images_paths, x_image_list, y_label_list, list_name):
    # Track progress and get total number of images
    progress = 0
    len_images_paths = len(images_paths)
    # Loop over images
    for n, image_path in enumerate(images_paths):
        # Read and resize images
        image = cv2.imread(image_path)
        image_resized = cv2.resize(image, (300, 300))
        # Get gray image
        image_gray = cv2.cvtColor(image_resized, cv2.COLOR_BGR2GRAY)
        # HUMoments for shape
        image_hu = cv2.HuMoments(cv2.moments(image_gray)).flatten()
        # Haralick for texture
        image_har = mahotas.features.haralick(image_gray).mean(axis=0)
        # Convert the image to HSV color-space
        image_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        # get normalized and flattened color histogram
        image_hist = cv2.calcHist([image_hsv], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
        cv2.normalize(image_hist, image_hist)
        image_hist_flat = image_hist.flatten()
        # combine the features extracted
        f_vector_concat = np.hstack([image_hist_flat, image_har, image_hu])
        # Add results to the X set
        x_image_list.append(f_vector_concat)
        # Add the labels to the Y set
        y_label_list.append(get_y_label(image_path))
        # Update and print progress
        progress = get_progress(list_name, progress, n, len_images_paths)
    print("\n")


print("Starting the Global Features approach:\n")

# Track time
t0 = time.time()

[train_images_paths, test_images_paths] = get_image_paths()

# create empty lists to hold the images being read
X_train_image = []
X_test_image = []
y_train_label = []
y_test_label = []

# Call the function for both training and test images
global_features(train_images_paths, X_train_image, y_train_label, "Train Data")
global_features(test_images_paths, X_test_image, y_test_label, "Test Data")

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
print("Training the model")
rfc_2 = RandomForestClassifier(n_estimators=478, max_depth=13)
rfc_2.fit(X_train, y_train)

# # The following code can be used to get the best parameters for the RFC
# param_dist = {'n_estimators': randint(50, 500),
#             'max_depth': randint(1, 20)}
# rand_search = RandomizedSearchCV(rfc,
#                                  param_distributions=param_dist,
#                                  n_iter=5,
#                                  cv=5)
# rand_search.fit(X_train, y_train)

# # Create a variable for the best model
# best_rfc = rand_search.best_estimator_

# # Print the best hyperparameters
# print('Best hyperparameters:',  rand_search.best_params_)

# Evaluate the model
# y_pred = best_rfc.predict(X_test)
y_pred = rfc_2.predict(X_test)

# Get result metrics, and time and print them
acc = round(accuracy_score(y_test, y_pred) * 100, 2)
precision = round(precision_score(y_test, y_pred, average='weighted', labels=np.unique(y_pred)) * 100, 2)
f1 = f1_score(y_test, y_pred, average='weighted', labels=np.unique(y_pred))

t1 = time.time()
work_time = t1 - t0

print("Results for the Global Features approach:\n")
print('\nAccuracy = {}%\nPrecision = {}%\nF1-Score = {}\n\nTime taken = {} sec\n'
      .format(acc, precision, f1, work_time))

# Show message that work is finished
ctypes.windll.user32.MessageBoxW(0, "The code has finished.", "Finished!", 0)
