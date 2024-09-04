import cv2
import sys
import numpy as np
from tensorflow.keras.models import load_model

from traffic import IMG_HEIGHT, IMG_WIDTH

def main():
    if len(sys.argv) < 2:
        sys.exit("Usage: python traffic.py image_path [model_path]")

    image_path = sys.argv[1]
    img = cv2.imread(image_path)
    img = cv2.resize(img, (IMG_WIDTH, IMG_HEIGHT))

    # Add a batch dimension
    img = np.expand_dims(img, axis=0)

    model_path = "model.keras" # default model
    # if a model name is provided
    if len(sys.argv) == 3:
        model_path = sys.argv[2]
    model = load_model(model_path)
    prediction = model.predict(img)
    print("Probability list: \n" + str(prediction))
    predicted_class = np.argmax(prediction)
    print("Predicted class: " + str(predicted_class))

if __name__ == "__main__":
    main()
