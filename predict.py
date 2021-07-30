import argparse
import os, sys, argparse
import numpy as np

from tensorflow.python import training
from methods.methods import *
from methods.model import *
from tensorflow.keras.metrics import Recall, Precision, MeanIoU
from tqdm import tqdm
import matplotlib.pyplot as plt

"""PARSE ARGUMENTS"""
parser = argparse.ArgumentParser(description='Predicti with UNet with MobileNet')
parser.add_argument('-p', '--path', type=str, help='path to the image or directory for prediction')
parser.add_argument('-s', '--save_path', type=str, default='preds/', help='path where predictions are saved')
parser.add_argument('-he', '--image_height', type=int, default=256, help='image height')
parser.add_argument('-wi', '--image_width', type=int, default=256, help='image width')
parser.add_argument('-cl', '--classes', type=int, default=21, help='number of classes')
parser.add_argument('-wf', '--weights_file', type=str, default='test.hdf5', help='weights file')

args = parser.parse_args()

"""HYPERPARAMETERS"""
PATH = args.path
ISDIR = os.path.isdir(PATH)
NUM_OF_CLASSES = args.classes
WEIGHTS_FILE = args.weights_file
IMAGE_SIZE = args.image_height
SAVE_PATH = args.save_path

"""LOADING MODEL"""
model = model()
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=[MeanIoU(num_classes=NUM_OF_CLASSES), 
                                    tf.keras.metrics.Precision(), 
                                    tf.keras.metrics.Recall()])    # compile the model
print("Loading weights...")
model.load_weights(WEIGHTS_FILE)
print("Weights loaded!")

"""PREDICTING"""
if ISDIR:
    image_names = sorted(os.listdir(PATH))

    print(f"Predicting images from {PATH}...")
    for image_name in tqdm(image_names):
        # Reading stage
        image = read_image(PATH + image_name, decode=False)     # read image
        
        # Prediction stage
        image = image[None, :]
        prediction = model.predict(image)           # predict - output has shape (1, IMAGE_SIZE, IMAGE_SIZE, NUM_OF_CLASSES)
        prediction = np.squeeze(prediction)         # remove the first axis
        prediction = np.argmax(prediction, axis=2)  # argmax the predictions

        # Saving stage
        cv2.imwrite(filename=SAVE_PATH + image_name, img=prediction)
    print("Finished!")
else:
    print("Predicting single image...")
    image = read_image(PATH, decode=False)      # read the image
    image = image[None, :]                      # adjust the axis
    preds = model.predict(image)                # predict - output has shape (1, IMAGE_SIZE, IMAGE_SIZE, NUM_OF_CLASSES)
    preds = np.squeeze(preds)                   # remove the first axis
    preds = np.argmax(preds, axis=2)            # argmax the predictions
    
    plt.imshow(np.dot(preds, 255 / (NUM_OF_CLASSES - 1)))
    plt.show()