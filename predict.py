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
# parser.add_argument('-e', '--epochs', type=int, default=5, help='number of epochs')
# parser.add_argument('-lr', '--learning_rate', type=float, default=1e-3, help='learning rate')
parser.add_argument('-he', '--image_height', type=int, default=256, help='image height')
parser.add_argument('-wi', '--image_width', type=int, default=256, help='image width')
# parser.add_argument('-dp', '--data_path', type=str, default='data/', help='data path')
parser.add_argument('-cl', '--classes', type=int, default=21, help='number of classes')
# parser.add_argument('-lw', '--load_weights', type=str2bool, default=True, help='load weights from existing file')
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
    image_names = os.listdir(PATH)

    print(f"Reading images from {PATH}...")
    images = []
    for image_name in tqdm(image_names):
        image = read_image(PATH + image_name, decode=False)     # read image
        image = image.astype(np.uint8)                          # convert to uint8 for saving memory
        images.append(image)                                    # append to images list
    images = np.array(images).astype(np.uint8)                  # convert images list into numpy array

    print("Predicting loaded images...")
    predictions = []
    for image in tqdm(images):
        image = image[None, :]                      # adjust the axis
        prediction = model.predict(image)           # predict - output has shape (1, IMAGE_SIZE, IMAGE_SIZE, NUM_OF_CLASSES)
        prediction = np.squeeze(prediction)         # remove the first axis
        prediction = np.argmax(prediction, axis=2)  # argmax the predictions
        prediction = prediction.astype(np.uint8)    # convert into uint8 for saving memory
        predictions.append(prediction)              # append to predictions
    predictions = np.array(predictions).astype(np.uint8)

    print(f"Saving predictions to {SAVE_PATH}...")
    for i in tqdm(range(len(predictions))):
        cv2.imwrite(filename=SAVE_PATH + image_names[i], img=predictions[i])
    print(f"Finished!")
else:
    print("Predicting single image...")
    image = read_image(PATH, decode=False)      # read the image
    image = image[None, :]                      # adjust the axis
    preds = model.predict(image)                # predict - output has shape (1, IMAGE_SIZE, IMAGE_SIZE, NUM_OF_CLASSES)
    preds = np.squeeze(preds)                   # remove the first axis
    preds = np.argmax(preds, axis=2)            # argmax the predictions
    
    plt.imshow(np.dot(preds, 255 / (NUM_OF_CLASSES - 1)))
    plt.show()