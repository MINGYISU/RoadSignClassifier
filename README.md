# Road Sign Classifier
## Overview
This project implements a deep learning model to recognize traffic signs using the German Traffic Sign Recognition Benchmark (GTSRB) dataset. The model is built with TensorFlow/Keras and leverages convolutional neural networks (CNNs) for image classification.
### Data Preprocessing:
- Images are loaded and preprocessed using OpenCV and resized to 30x30 pixels.
- Preprocessed data is saved as a pickle file to avoid redundant loading.
- The dataset is divided into 43 categories, each representing a different type of road sign.
### Model Architecture:
- The CNN uses several convolutional and pooling layers for feature extraction.
- Data Augmentation is applied to enhance model robustness, including techniques like random flipping, zoom, rotation, and contrast adjustments.
- Batch Normalization and Dropout layers are included for better generalization and preventing overfitting.
- The final layer uses the softmax activation function to classify images into one of 43 categories.
### Training and Evaluation:
- The model is trained for 10 epochs using the Adam optimizer and categorical crossentropy loss function.
- The history of the training process is saved to a pickle file and can be used for visualization and evaluation.
- The trained model is saved for future use. 

## !Important: Training dataset
The training source, i.e., GTSRB - German Traffic Sign Recognition Benchmark, is not provided here due to the oversize amount of storage. Please go to Kaggle (https://www.kaggle.com/datasets/meowmeowmeowmeowmeow/gtsrb-german-traffic-sign) to download the dataset. 

## Getting Start
1. Go to https://www.kaggle.com/datasets/meowmeowmeowmeowmeow/gtsrb-german-traffic-sign and download the GTSRB dataset, make sure the folder is named *gtsrb* and is saved in the same folder as *dataloading.py*
2. Run $**pip3 install -r requirements.txt**
3. Run $**python3 dataloading.py** and a file called *gtsrb_data.pkl* should be generated
4. Make sure the file *gtsrb_data.pkl* exists and run $**python3 traffic.py [model_name]**
5. Run $**python3 test.py** to visualize the test result
6. Run $**python3 predict.py path_to_your_image** to call the model to classify your image

## Credits
- traffic.py: Portions of code from main() were adapted from [CS50’s Web Programming with Python and JavaScript](https://cdn.cs50.net/ai/2023/x/projects/5/trafc.zip).Harvard University.2024
- /gtsrb: The dataset was retrieved from German Traffic Sign Recognition Benchmark (http://benchmark.ini.rub.de/?section=gtsrb&subsection=news)  

This project is licensed under a Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International (CC BY-NC-SA 4.0) license. This is a human-readable summary of (and not a substitute for) the license. Official translations of this license are available in other languages.

You are free to:

- Share — copy and redistribute the material in any medium or format
- Adapt — remix, transform, and build upon the material

Under the following terms:

- Attribution — You must give appropriate credit, provide a link to the license, and indicate if changes were made. You may do so in any reasonable manner, but not in any way that suggests the licensor endorses you or your use.
- NonCommercial — You may not use the material for commercial purposes.
- ShareAlike — If you remix, transform, or build upon the material, you must distribute your contributions under the same license as the original.
- No additional restrictions — You may not apply legal terms or technological measures that legally restrict others from doing anything the license permits.
