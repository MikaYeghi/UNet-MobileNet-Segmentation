import os, argparse, sys
import cv2
import random
from sklearn.utils import class_weight
import tensorflow as tf
import numpy as np
from glob import glob
from tensorflow.python.ops.gen_array_ops import reshape
from tqdm import tqdm
from sklearn.model_selection import train_test_split

def meanIoU(masks, preds, num_of_classes=21):
    IoUs = dict(zip([i for i in range(num_of_classes)], [0 for i in range(num_of_classes)]))
    for class_idx in tqdm(range(num_of_classes)):
        union = np.logical_or(masks==class_idx, preds==class_idx).astype(np.uint8)
        intersection = np.logical_and(masks==class_idx, preds==class_idx).astype(np.uint8)
        IoUs[class_idx] = np.sum(intersection) / np.sum(union)
    mean_IoU = sum(IoUs.values()) / num_of_classes

    return mean_IoU

def load_data(path, split=0.1, use_percentage=0.1):
    # images = sorted(glob(os.path.join(path, "images/*")))       # read original images
    masks = sorted(glob(os.path.join(path, "masks/*")))         # read masks
    image_mask_dict = {}

    for mask in masks:
        image_name = mask[:-9] + ".png"                         # remove "mask"
        image_name = image_name[11:]                            # remove "data/masks/"
        image_name = path + "images/" + image_name              # replace with "data/images/"
        image_mask_dict[image_name] = mask                      # generate the dictionary
    del masks

    total_number = len(image_mask_dict)                         # total number of images
    images_to_use = random.sample(image_mask_dict.keys(), int(total_number * use_percentage))
    selected_images = {image: image_mask_dict[image] for image in images_to_use}
    image_mask_dict = selected_images
    del selected_images, images_to_use
    print(f"Using {len(image_mask_dict)} images from {total_number}...")

    images = list(image_mask_dict.keys())                       # extract images
    masks = list(image_mask_dict.values())                      # extract masks
    del image_mask_dict

    total_size = len(images)                                    # total number of images
    valid_size = int(split * total_size)                        # number of validation images
    test_size = int(split * total_size)                         # number of test images

    train_x, valid_x = train_test_split(images, test_size=valid_size)      # split X into train/val
    train_y, valid_y = train_test_split(masks, test_size=valid_size)       # split y into train/val

    train_x, test_x = train_test_split(train_x, test_size=test_size)       # split X into train/test
    train_y, test_y = train_test_split(train_y, test_size=test_size)       # split y into train/test

    return (train_x, train_y), (valid_x, valid_y), (test_x, test_y)

def make_dataset(Xs, ys, IMAGE_SIZE=256, skip_counter=1):
    masks = []
    print("Reading grayscale masks.")
    for mask_path in tqdm(ys[::skip_counter]):
        mask = read_mask(mask_path, image_size=IMAGE_SIZE, decode=False)
        mask = mask.astype(np.uint8)
        masks.append(mask)        
    masks = np.array(masks).astype(np.uint8)

    images = []
    print("Reading RGB images.")
    for img_path in tqdm(Xs[::skip_counter]):
        img = read_image(img_path, image_size=IMAGE_SIZE, decode=False)
        img = img.astype(np.uint8)
        images.append(img)
    images = np.array(images).astype(np.uint8)

    return images, masks

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
    x = cv2.resize(x, (image_size, image_size), interpolation = cv2.INTER_NEAREST)          # resize the mask
    one_hot_x = np.zeros(shape=(image_size, image_size, num_of_classes)).astype(np.uint8)   # one-hot encoded tensor
    for i in range(len(one_hot_x)):
        for j in range(len(one_hot_x[i])):
            one_hot_x[i][j][x[i][j]] = 1                                                    # convert each element into a one-hot encoded array
    return one_hot_x

def tf_parse(x, y, w, image_size=256, num_of_classes=21):
    def _parse(x, y):
        x = read_image(x)
        y = read_mask(y)
        return x, y
    x, y = tf.numpy_function(_parse, [x, y], [tf.float64, tf.uint8])
    x.set_shape([image_size, image_size, 3])
    y.set_shape([image_size, image_size, num_of_classes])
    return x, y

def tf_dataset(x, y, weights, batch=8):
    # x - path to images
    # y - path to masks
    dataset = tf.data.Dataset.from_tensor_slices((x, y, weights))       # generates a dataset
    dataset = dataset.map(tf_parse)                                     # parse images
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
    
def get_class_weights(path, skip_counter=1000):
    path += "masks/"                # folder containing masks is "masks"
    os.chdir(path)                  # enter masks

    mask_filenames = os.listdir()   # list of filenames in masks
    mask_labels = np.array([])      # initialize empty array with labels from masks

    print("Reading mask files for estimating class weights...")

    for mask_filename in tqdm(mask_filenames[::skip_counter]):
        mask = read_mask(mask_filename, decode=False)           # read one-hot encoded masks
        mask = np.argmax(mask, axis=2)                          # argmax them
        reshaped_mask = mask.reshape(-1, 1)                     # convert into 1d
        mask_labels = np.append(mask_labels, reshaped_mask)     # append to the labels array

    class_weights = class_weight.compute_class_weight('balanced', np.unique(mask_labels), mask_labels)      # compute labels
    num_of_classes = np.unique(mask_labels).shape[0]
    print("Number of classes is ", num_of_classes)
    class_weights = dict(zip(range(num_of_classes), list(class_weights)))
    os.chdir("../..")
    return class_weights