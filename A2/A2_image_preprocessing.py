import os
import cv2
import numpy as np
import pandas as pd
from keras.preprocessing.image import img_to_array
from os.path import dirname, abspath

# Root path
base_dir = dirname(dirname(abspath(__file__)))


# Load image arrays for hybrid CNN-SVM model
def A2_CNN_SVM_load_images(labels_dir, labels_filename, images_dir):
    dataframe = pd.read_csv(os.path.join(labels_dir, labels_filename), delimiter='\t')
    img_list = dataframe.values[:, 1]
    labels = dataframe.values[:, 3]  # Smiling labels
    figure_count = dataframe.values[:, 0]
    imgs = [os.path.join(images_dir, file) for file in img_list]

    norm_size = 64

    if os.path.isdir(images_dir):
        global_imgs = []
        global_labels = []
        i = 0
        # Read images, grayscale conversion, resizing
        for img_path in imgs:
            image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            image = cv2.resize(image, (norm_size, norm_size))
            image = img_to_array(image)

            label = labels[figure_count[i]]
            if label == -1:
                label = 0
            global_imgs.append(image)
            global_labels.append(label)
            i += 1

        global_imgs = np.array(global_imgs)
        global_labels = np.array(global_labels)

    return global_imgs, global_labels


# load image arrays with mouth localisation for hybrid CNN-SVM model
def A2_CNN_SVM_load_mouth_images(labels_dir, labels_filename, images_dir):
    dataframe = pd.read_csv(os.path.join(labels_dir, labels_filename), delimiter='\t')
    img_list = dataframe.values[:, 1]
    labels = dataframe.values[:, 3]  # Smiling labels
    figure_count = dataframe.values[:, 0]
    imgs = [os.path.join(images_dir, file) for file in img_list]

    norm_size = 64

    if os.path.isdir(images_dir):
        global_imgs = []
        global_labels = []
        i = 0
        for img_path in imgs:
            image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            # Determine coordinates and size of mouth region
            xmin = int(image.shape[0] * 0.25)
            ymin = int(image.shape[1] * 0.78)
            w = int(image.shape[0] * 0.3)
            h = int(image.shape[1] * 0.2)
            # Crop the mouth and convert to arrays
            imgCrop = image[ymin:ymin + h, xmin:xmin + w].copy()
            imgCrop = cv2.resize(imgCrop, (norm_size, norm_size))
            imgCrop = img_to_array(imgCrop)

            label = labels[figure_count[i]]
            if label == -1:
                label = 0
            global_imgs.append(imgCrop)
            global_labels.append(label)
            i += 1

        global_imgs = np.array(global_imgs)
        global_labels = np.array(global_labels)

    return global_imgs, global_labels


# Crop the target region
def __crop(img, position, size):
    ow, oh = img.size
    x1, y1 = position
    tw = size[0]
    th = size[1]
    if (ow > tw and oh > th):
        return img.crop((x1, y1, x1 + tw, y1 + th))
    return img
