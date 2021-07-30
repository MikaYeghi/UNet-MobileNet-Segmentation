import os
from tqdm import tqdm
import cv2
import numpy as np
from keras.metrics import MeanIoU
from methods.methods import *


m = MeanIoU(num_classes=21)

mask_filenames = os.listdir("data/masks/")
pred_filenames = os.listdir("preds/")

print("Reading masks...")
masks = []
for mask_filename in tqdm(mask_filenames[::1]):
    mask = cv2.imread("data/masks/" + mask_filename, cv2.IMREAD_GRAYSCALE)
    mask = mask.astype(np.uint8)
    mask = cv2.resize(mask, (256, 256))
    masks.append(mask)
masks = np.array(masks)

print("Reading predictions...")
preds = []
for pred_filename in tqdm(pred_filenames[::1]):
    pred = cv2.imread("preds/" + pred_filename, cv2.IMREAD_GRAYSCALE)
    pred = pred.astype(np.uint8)
    preds.append(pred)
preds = np.array(preds)

masks = masks.astype(np.uint8)
preds = preds.astype(np.uint8)

# m.update_state(preds, masks)
# print(m.result().numpy())
mean_IoU = meanIoU(masks, preds, 21)
print(mean_IoU)