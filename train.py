import os, sys, argparse
from pprint import pprint
from tensorflow.python.keras.utils.generic_utils import default
from tqdm import tqdm
import time
import random
import numpy as np
import cv2
from glob import glob
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.utils import class_weight

from methods.methods import *
from methods.model import *
from tensorflow.keras.layers import Conv2D, Activation, BatchNormalization
from tensorflow.keras.layers import UpSampling2D, Input, Concatenate
from tensorflow.keras.models import Model
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.metrics import Recall, Precision, MeanIoU
from tensorflow.keras import backend as K


"""SET THE RANDOM SEED"""
np.random.seed(42)
tf.random.set_seed(42)


"""PARSE ARGUMENTS"""
parser = argparse.ArgumentParser(description='UNet with MobileNet')
parser.add_argument('-b', '--batch_size', type=int, default=8, help='batch size')
parser.add_argument('-e', '--epochs', type=int, default=5, help='number of epochs')
parser.add_argument('-lr', '--learning_rate', type=float, default=1e-3, help='learning rate')
parser.add_argument('-he', '--image_height', type=int, default=256, help='image height')
parser.add_argument('-wi', '--image_width', type=int, default=256, help='image width')
parser.add_argument('-dp', '--data_path', type=str, default='data/', help='data path')
parser.add_argument('-cl', '--classes', type=int, default=21, help='number of classes')
parser.add_argument('-lw', '--load_weights', type=str2bool, default=True, help='load weights from existing file')
parser.add_argument('-wf', '--weights_file', type=str, default='test.hdf5', help='weights file')

args = parser.parse_args()

"""HYPERPARAMETERS"""
IMAGE_SIZE = args.image_height      # equal in both dimensions
EPOCHS = args.epochs                # number of epochs in training
BATCH = args.batch_size             # batch size
LR = args.learning_rate             # learning rate
PATH = args.data_path               # path to the file with the dataset
NUM_OF_CLASSES = args.classes       # number of classes to be detected
LOAD_WEIGHTS = args.load_weights    # True - load weights from the file, False - don't load the saved weights
WEIGHTS_FILE = args.weights_file    # file with the model weights

print(f"Launching the training file with the following parameters:\n\
        Batch size: {BATCH}\n\
        Epochs: {EPOCHS}\n\
        Image size: {IMAGE_SIZE}\n\
        Learning rate: {LR}\n\
        Path to the data: {PATH}\n\
        Number of classes: {NUM_OF_CLASSES}\n\
        Loading weights from a file: {LOAD_WEIGHTS}\n\
        Weights file: {WEIGHTS_FILE}")

"""TRAINING/LOADING WEIGHTS"""
(train_x, train_y), (valid_x, valid_y), (test_x, test_y) = load_data(PATH)

train_dataset = tf_dataset(train_x, train_y, batch=BATCH)
valid_dataset = tf_dataset(valid_x, valid_y, batch=BATCH)

train_steps = len(train_x)//BATCH
valid_steps = len(valid_x)//BATCH

if len(train_x) % BATCH != 0:
    train_steps += 1
if len(valid_x) % BATCH != 0:
    valid_steps += 1


"""INITIALIZE MODEL"""
model = model()
# model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])      # compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=[MeanIoU(num_classes=NUM_OF_CLASSES)])    # compile the model

# if LOAD_WEIGHTS:
if LOAD_WEIGHTS:
    print("Loading weights...")
    model.load_weights(WEIGHTS_FILE)
    print("Weights loaded!")


"""TRAINING PARAMETERS"""
callbacks = [
    ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=4),
    EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
    ModelCheckpoint(WEIGHTS_FILE, save_best_only=True, monitor='val_loss', mode='min')
]


"""TRAIN"""
print("Starting training...")
model.fit(
    train_dataset,
    validation_data=valid_dataset,
    epochs=EPOCHS,
    steps_per_epoch=train_steps,
    validation_steps=valid_steps,
    callbacks=callbacks,
    verbose=1
)
print("Training finished!")

print("Saving the model...")
model.save(WEIGHTS_FILE)
print("Model saved!")