# import libraries
import numpy as np
import cv2
import os
import sys
import time
# from scipy.stats import randint
from sklearn.ensemble import RandomForestClassifier
# from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import precision_score, accuracy_score, f1_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
from pathlib import Path
import mahotas
import warnings
import ctypes
warnings.filterwarnings('ignore')

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


# Get y label for each image
def get_y_label(image_path):
    y_label = image_path[78:].split("_")[0]
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


#################################
# (1) USING RAW PIXEL APPROACH
##################################

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

# create empty lists to hold the images being read
X_train_image_1 = []
X_test_image_1 = []
y_train_label_1 = []
y_test_label_1 = []

# Call the function for both training and test images
raw_pixel(train_images_paths, X_train_image_1, y_train_label_1, "Train Data")
raw_pixel(test_images_paths, X_test_image_1, y_test_label_1, "Test Data")

# convert lists to numpy array and print their shapes
X_train = np.array(X_train_image_1)
X_test = np.array(X_test_image_1)
print("Shape of X_train is: {}".format(X_train.shape))
print("Shape of X_test is: {}".format(X_test.shape))

y_train = np.array(y_train_label_1)
y_test = np.array(y_test_label_1)
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


# #######################################################
# # (2) USING GLOBAL FEATURES for Image Classification
# ########################################################


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

# create empty lists to hold the images being read
X_train_image_2 = []
X_test_image_2 = []
y_train_label_2 = []
y_test_label_2 = []

# Call the function for both training and test images
global_features(train_images_paths, X_train_image_2, y_train_label_2, "Train Data")
global_features(test_images_paths, X_test_image_2, y_test_label_2, "Test Data")

# convert lists to numpy array and print their shapes
X_train = np.array(X_train_image_2)
X_test = np.array(X_test_image_2)
print("Shape of X_train is: {}".format(X_train.shape))
print("Shape of X_test is: {}".format(X_test.shape))

y_train = np.array(y_train_label_2)
y_test = np.array(y_test_label_2)
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


# #############################################################################
# Using KEYPOINTS & DESCRIPTORS from SIFT/ORB and Bag of Visual Words using KMeans
# #############################################################################


# Function using global features
def get_descriptors(method, images_paths, x_image_list, y_label_list, list_name):
    # Track progress and get total number of images
    progress = 0
    len_images_paths = len(images_paths)
    # Loop over images
    for n, image_path in enumerate(images_paths):
        # Read and resize images
        image = cv2.imread(image_path)
        image = cv2.resize(image, (300, 300))
        # Get gray image
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # Get keypoint and descriptors using specified method
        keypoints, descriptors = method.detectAndCompute(image, None)
        # Add descriptors to the X set
        x_image_list.append(descriptors)
        # Add the labels to the Y set
        y_label_list.append(get_y_label(image_path))
        # Update and print progress
        progress = get_progress(list_name, progress, n, len_images_paths)
    print("\n")


# creating bag of visual words feature vectors for the images in the list using kmeans
def bag_of_words(kmeans, x_image_list):
    bow_vec = np.zeros([len(x_image_list), 10])
    for index, features in enumerate(x_image_list):
        for i in kmeans.predict(features):
            bow_vec[index, i] += 1
            continue
        continue
    return bow_vec


print("Starting the Descriptors and Bag of Words approach:\n")

# Track time
t0 = time.time()

# create empty lists to hold the images being read
X_train_image_3 = []
X_test_image_3 = []
y_train_label_3 = []
y_test_label_3 = []

# Specify required algorithm
# algorithm = cv2.ORB_create()
algorithm = cv2.SIFT_create()

# Call the function for both training and test images
get_descriptors(algorithm, train_images_paths, X_train_image_3, y_train_label_3, "Train Data")
get_descriptors(algorithm, test_images_paths, X_test_image_3, y_test_label_3, "Test Data")

# Stack the X arrays and apply a min-max scaler to them
X_train_image_features = np.vstack(np.array(X_train_image_3))
X_test_image_features = np.vstack(np.array(X_test_image_3))

scaler = MinMaxScaler(feature_range=(0, 1))
X_train_image_features = scaler.fit_transform(X_train_image_features)
X_test_image_features = scaler.fit_transform(X_test_image_features)

# Use KMeans to compute centroids to build bag of visual words
kmeans_train = KMeans(n_clusters=10, random_state=0).fit(X_train_image_features)
kmeans_test = KMeans(n_clusters=10, random_state=0).fit(X_test_image_features)

X_train = bag_of_words(kmeans_train, X_train_image_3)
X_test = bag_of_words(kmeans_test, X_test_image_3)
print("Shape of X_train is: {}".format(X_train.shape))
print("Shape of X_test is: {}".format(X_test.shape))

y_train = np.array(y_train_label_3)
y_test = np.array(y_test_label_3)
print("Shape of y_train is: {}".format(y_train.shape))
print("Shape of y_test is: {}".format(y_test.shape))

# Train the model
rfc_3 = RandomForestClassifier()
rfc_3.fit(X_train, y_train)

# Evaluate model
y_pred = rfc_3.predict(X_test)

# Get result metrics, and time and print them
acc = round(accuracy_score(y_test, y_pred) * 100, 2)
precision = round(precision_score(y_test, y_pred, average='weighted', labels=np.unique(y_pred)) * 100, 2)

t1 = time.time()
work_time = round(t1 - t0, 2)

print("Results for the SIFT/ORB and Bag of Visual Words using KMeans approach:\n")
print('\nAccuracy = {}%\nPrecision = {}%\n\nTime taken = {} sec\n'.format(acc, precision, work_time))

# Show message that work is finished
ctypes.windll.user32.MessageBoxW(0, "The code has finished.", "Finished!", 0)
