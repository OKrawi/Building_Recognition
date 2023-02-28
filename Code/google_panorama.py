# Import required modules
import google_streetview.api as sv
import os
import sys
import numpy as np
import json
import time
from pathlib import Path

# Read and load the data
file = open('../Data/panos.json')
data = json.load(file)
file.close()

# Check the train directory and create it if it doesn't exist
main_dir = os.path.join(os.path.dirname(sys.path[0]), 'panoramas\\Train')
main_path = Path(main_dir)

if not main_path.exists():
    os.makedirs(os.path.join(os.path.dirname(sys.path[0]), 'panoramas\\Train'))

# Check the test directory and create it if it doesn't exist
test_dir = os.path.join(os.path.dirname(sys.path[0]), 'panoramas\\_Test')
test_path = Path(test_dir)

if not test_path.exists():
    os.makedirs(os.path.join(os.path.dirname(sys.path[0]), 'panoramas\\_Test'))

# Initialize variables
progress = 0
downloaded = 0
current_num_of_files = len(os.listdir(main_dir))
expected_num_of_files = len(data) * 8

# Keep on trying to download pictures as long as we don't have pictures from 8 angles for each panorama ID
while current_num_of_files < expected_num_of_files:
    # Loop over the data
    for i, pano in enumerate(data):
        # Print progress
        new_progress = int((i + 1) * 100 / len(data))
        if new_progress > (progress + 4) or new_progress == 100:
            print("Progress {}%".format(new_progress))
            progress = new_progress

        # Get location of each panorama
        location = ("{},{}".format(pano["lat"], pano["lng"]))

        # Get panoramas from different angles
        for heading in np.arange(0, 360, 45):

            # Only proceed if the picture is new
            path_string = main_dir + '\\{}_{}.jpg'.format(location, heading)
            path = Path(path_string)

            if not path.is_file():
                # Set params for request
                params = [{
                    'size': '640x640',
                    'location': location,
                    'heading': heading,
                    'pitch': '30',
                    'key': 'AIzaSyCuNV6FLwCDj05u0V537hhUZd40f-TsSe8'
                }]

                # Wrap request in a try-except block to watch out for server-related errors
                try:
                    # Make the request and download and save the picture with the required name
                    results = sv.results(params)
                    results.download_links("../panoramas/Train")
                    os.rename(main_dir + "\\gsv_0.jpg", path_string)
                    # Increment tracking variables
                    downloaded = downloaded + 1
                    current_num_of_files = current_num_of_files + 1
                except Exception as ex:
                    # Print error and wait for 5 seconds before retrying
                    print(ex)
                    print("Issue is for location: {}, heading: {}".format(location, heading))
                    time.sleep(5)
                    continue

# Remove auto-downloaded metadata file
metadata_path = Path(main_dir + '\\metadata.json')
if metadata_path.is_file():
    os.remove(main_dir + '\\metadata.json')

# Print total number of downloaded pictures
print("Downloaded {} pictures".format(downloaded))
