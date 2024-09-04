import os
import pickle
import sys
import numpy as np
import tensorflow as tf

from sklearn.model_selection import train_test_split
from tensorflow.keras import layers

EPOCHS = 10
IMG_WIDTH = 30
IMG_HEIGHT = 30
NUM_CATEGORIES = 43
TEST_SIZE = 0.4


def main():
    # Get image arrays and labels for all image files
    # Load the saved data from existing file to reduce time each time training a new model
    to_load = "gtsrb_data.pkl" # !!!Remember to change this if you change the name of the file
    if not os.path.isfile(to_load):
        sys.exit(f"{to_load} doesn't exist! Make sure you run $python dataloading.py first!")
    with open(to_load, 'rb') as f:
        images, labels = pickle.load(f)
    if len(images) != 0 and len(labels) != 0:
        print("Data loaded successfully!")
    else:
        sys.exit("Data loading failed!")

    # Split data into training and testing sets
    labels = tf.keras.utils.to_categorical(labels)
    x_train, x_test, y_train, y_test = train_test_split(
        np.array(images), np.array(labels), test_size=TEST_SIZE
    )

    # Get a compiled neural network
    model = get_model()

    # Fit model on training data
    history = model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=EPOCHS, batch_size=32, verbose=2)

    # Save the history
    to_save = 'history.pkl'
    with open(to_save, 'wb') as f:
        pickle.dump(history, f)

    # Evaluate neural network performance
    model.evaluate(x_test,  y_test, verbose=2)

    
    # Save model to file
    filename = "model.keras"
    # if a preferred model name is provided, save to that name
    if len(sys.argv) == 2:
        filename = sys.argv[1]
    model.save(filename)
    print(f"Model saved to {filename}.")




def get_model():
    """
    Returns a compiled convolutional neural network model. Assume that the
    `input_shape` of the first layer is `(IMG_WIDTH, IMG_HEIGHT, 3)`.
    The output layer should have `NUM_CATEGORIES` units, one for each category.
    """
    model = tf.keras.models.Sequential([
        # Input layer
        layers.Input((IMG_WIDTH, IMG_HEIGHT, 3)), 

        # Data Augmentation
        layers.RandomFlip("horizontal_and_vertical"),
        layers.RandomContrast(0.5),
        layers.RandomZoom(height_factor=(0.2, 0.3), width_factor=(0.2, 0.3)),
        layers.RandomRotation(factor=0.20),
        layers.RandomTranslation(height_factor=0.1, width_factor=0.1),
        
        # Feature extraction

        # First convolutional block, with 32 filters, each with a 3*3 kernal
        layers.Conv2D(32, (3, 3), activation='relu', padding="same"),
        layers.MaxPooling2D((2, 2)),

        # Second convolutional block
        layers.Conv2D(64, (5, 5), activation='relu', padding="same"),
        layers.MaxPooling2D((3, 3)),

        # Second convolutional block
        layers.Conv2D(128, (7, 7), activation='relu', padding="same"),
        layers.MaxPooling2D((5, 5)),


        # Flatten the output of the convolutional layers
        layers.Flatten(),

        # Add a hidden layer, randomly dropout some nodes to prevent overfitting
        layers.Dense(64, activation="relu"), 
        layers.Dropout(0.5), 
        layers.BatchNormalization(), 

        # Add another hidden layer to improve accuracy
        layers.Dense(128, activation="relu"), 
        layers.Dropout(0.75), 
        layers.BatchNormalization(), 

        # Add another hidden layer to improve accuracy
        layers.Dense(256, activation="relu"), 
        layers.Dropout(0.5), 
        layers.BatchNormalization(), 

        # output layer
        layers.Dense(64, activation='relu'),
        layers.Dense(NUM_CATEGORIES, activation='softmax')
    ])

    # Train the neural network
    model.compile(
        optimizer="adam",
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )

    return model


if __name__ == "__main__":
    main()
