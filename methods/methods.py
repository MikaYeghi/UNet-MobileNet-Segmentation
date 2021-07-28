import os, argparse
import cv2
import tensorflow as tf
import numpy as np
from glob import glob
from sklearn.model_selection import train_test_split

def load_data(path, split=0.1):
    images = sorted(glob(os.path.join(path, "images/*")))       # read original images
    masks = sorted(glob(os.path.join(path, "masks/*")))         # read masks
    image_mask_dict = {}

    for mask in masks:
        image_name = mask[:-9] + ".png"                         # remove "mask"
        image_name = image_name[11:]                            # remove "data/masks/"
        image_name = path + "images/" + image_name              # replace with "data/images/"
        image_mask_dict[image_name] = mask                      # generate the dictionary
    
    images = list(image_mask_dict.keys())                       # extract images
    masks = list(image_mask_dict.values())                      # extract masks

    total_size = len(images)                                    # total number of images
    valid_size = int(split * total_size)                        # number of validation images
    test_size = int(split * total_size)                         # number of test images

    train_x, valid_x = train_test_split(images, test_size=valid_size, random_state=42)      # split X into train/val
    train_y, valid_y = train_test_split(masks, test_size=valid_size, random_state=42)       # split y into train/val

    print(train_x[0], train_y[0])

    train_x, test_x = train_test_split(train_x, test_size=test_size, random_state=42)       # split X into train/test
    train_y, test_y = train_test_split(train_y, test_size=test_size, random_state=42)       # split y into train/test

    return (train_x, train_y), (valid_x, valid_y), (test_x, test_y)

def read_image(path, image_size=256, decode=True):                  # returns a (IMAGE_SIZE, IMAGE_SIZE, 3) tensor (RGB)
    if decode:
        path = path.decode()                        # decode the path
    x = cv2.imread(path, cv2.IMREAD_COLOR)          # read the image as BGR
    B, G, R = cv2.split(x)                          # split into 3 channels
    x = cv2.merge([R, G, B])                        # convert into RGB
    x = cv2.resize(x, (image_size, image_size))     # resize the image
    x = x/255.0                                     # normalize
    return x

def read_mask(path, image_size=256, num_of_classes=21, decode=True):                                                           # returns a (IMAGE_SIZE, IMAGE_SIZE, NUM_OF_CLASSES) tensor
    if decode:
        path = path.decode()                                                                # decode the path
    x = cv2.imread(path, cv2.IMREAD_GRAYSCALE)                                              # read the mask as gray scale
    x = cv2.resize(x, (image_size, image_size))                                             # resize the mask
    one_hot_x = np.zeros(shape=(image_size, image_size, num_of_classes)).astype(np.uint8)   # one-hot encoded tensor
    for i in range(len(one_hot_x)):
        for j in range(len(one_hot_x[i])):
            one_hot_x[i][j][x[i][j]] = 1                                                    # convert each element into a one-hot encoded array
    return one_hot_x

def tf_parse(x, y, image_size=256, num_of_classes=21):
    def _parse(x, y):
        x = read_image(x)
        y = read_mask(y)
        return x, y

    x, y = tf.numpy_function(_parse, [x, y], [tf.float64, tf.uint8])
    x.set_shape([image_size, image_size, 3])
    y.set_shape([image_size, image_size, num_of_classes])
    return x, y

def tf_dataset(x, y, batch=8):
    # x - path to images
    # y - path to masks
    dataset = tf.data.Dataset.from_tensor_slices((x, y))        # generates a dataset
    dataset = dataset.map(tf_parse)                             # parse images
    dataset = dataset.batch(batch)                              
    dataset = dataset.repeat()

    return dataset

def read_and_rgb(x):                            # read and convert the image into RGB
    # x - path to the image
    x = cv2.imread(x)                           # read the image
    x = cv2.cvtColor(x, cv2.COLOR_BGR2RGB)      # convert into RGB
    return x                                    # return the image

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')