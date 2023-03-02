# import libraries
import numpy as np
import cv2
import time
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score, accuracy_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
import warnings
import ctypes
from functions import get_y_label, get_progress, get_image_paths
warnings.filterwarnings('ignore')


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

[train_images_paths, test_images_paths] = get_image_paths()

# create empty lists to hold the images being read
X_train_image = []
X_test_image = []
y_train_label = []
y_test_label = []

# Specify required algorithm
# algorithm = cv2.ORB_create()
algorithm = cv2.SIFT_create()

# Call the function for both training and test images
get_descriptors(algorithm, train_images_paths, X_train_image, y_train_label, "Train Data")
get_descriptors(algorithm, test_images_paths, X_test_image, y_test_label, "Test Data")

# Stack the X arrays and apply a min-max scaler to them
X_train_image_features = np.vstack(np.array(X_train_image))
X_test_image_features = np.vstack(np.array(X_test_image))

scaler = MinMaxScaler(feature_range=(0, 1))
X_train_image_features = scaler.fit_transform(X_train_image_features)
X_test_image_features = scaler.fit_transform(X_test_image_features)

# Use KMeans to compute centroids to build bag of visual words
kmeans_train = KMeans(n_clusters=10, random_state=0).fit(X_train_image_features)
kmeans_test = KMeans(n_clusters=10, random_state=0).fit(X_test_image_features)

X_train = bag_of_words(kmeans_train, X_train_image)
X_test = bag_of_words(kmeans_test, X_test_image)
print("Shape of X_train is: {}".format(X_train.shape))
print("Shape of X_test is: {}".format(X_test.shape))

y_train = np.array(y_train_label)
y_test = np.array(y_test_label)
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
