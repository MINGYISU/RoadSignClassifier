import cv2
import numpy as np
import os
import sys
import pickle

from traffic import IMG_HEIGHT, IMG_WIDTH, NUM_CATEGORIES

def main():
    # Load data using your load_data function
    data_dir = "gtsrb" # !!!Remember to manually change this if the data directory name is changed
    images, labels = load_data(data_dir)

    # Save the data
    to_save = 'gtsrb_data.pkl'
    with open(to_save, 'wb') as f:
        pickle.dump((images, labels), f)



def load_data(data_dir):
    """
    Load image data from directory `data_dir`.

    Assume `data_dir` has one directory named after each category, numbered
    0 through NUM_CATEGORIES - 1. Inside each category directory will be some
    number of image files.

    Return tuple `(images, labels)`. `images` should be a list of all
    of the images in the data directory, where each image is formatted as a
    numpy ndarray with dimensions IMG_WIDTH x IMG_HEIGHT x 3. `labels` should
    be a list of integer labels, representing the categories for each of the
    corresponding `images`.
    """
    images = []
    labels = []

    for category in range(NUM_CATEGORIES):
        # Use os.path to achieve platform-independent
        category_path = os.path.join(data_dir, str(category))

        # Check if category_path is an existing directory
        if os.path.isdir(category_path):
            for filename in os.listdir(category_path):
                img_path = os.path.join(category_path, filename)

                # Load the image
                img = cv2.imread(img_path)

                # Resize the image to the desired dimension
                img = cv2.resize(img, (IMG_WIDTH, IMG_HEIGHT))

                # Append the image and label
                images.append(img)
                labels.append(category)
        else:
            sys.exit(f"Category directory {category_path} does not exist.")

    return images, labels

if __name__ == "__main__":
    main()
