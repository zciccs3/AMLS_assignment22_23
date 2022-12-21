import os
import cv2
import numpy as np
import pandas as pd
from keras.preprocessing.image import img_to_array
from tensorflow.python.keras.utils.np_utils import to_categorical


def A2_CNN_SVM_load_images(basedir, labels_filename, images_dir):
    dataframe = pd.read_csv(os.path.join(basedir, labels_filename), delimiter='\t')
    img_list = dataframe.values[:, 1]
    labels = dataframe.values[:, 3]  # Smiling labels
    figure_count = dataframe.values[:, 0]
    imgs = [os.path.join(images_dir, file) for file in img_list]

    norm_size = 64
    CLASS_NUM = 2

    print("Image feature extraction...")

    if os.path.isdir(images_dir):
        global_imgs = []
        global_labels = []
        i = 0
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
        print("Shape of global_imgs is ", global_imgs.shape)
        global_labels = np.array(global_labels)

        # global_labels = to_categorical(global_labels, num_classes=CLASS_NUM)
        # global_labels = np.array(global_labels)

    print("Finish image feature extraction......")

    return global_imgs, global_labels


def A2_CNN_SVM_load_mouth_images(basedir, labels_filename, images_dir):
    dataframe = pd.read_csv(os.path.join(basedir, labels_filename), delimiter='\t')
    img_list = dataframe.values[:, 1]
    labels = dataframe.values[:, 3]  # Smiling labels
    figure_count = dataframe.values[:, 0]
    imgs = [os.path.join(images_dir, file) for file in img_list]

    norm_size = 64
    CLASS_NUM = 2

    print("Image feature extraction...")

    if os.path.isdir(images_dir):
        global_imgs = []
        global_labels = []
        i = 0
        for img_path in imgs:
            image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

            xmin = int(image.shape[0] * 0.25)
            ymin = int(image.shape[1] * 0.78)
            w = int(image.shape[0] * 0.3)
            h = int(image.shape[1] * 0.2)
            imgCrop = image[ymin:ymin + h, xmin:xmin + w].copy()
            # img = cv2.resize(src=img, dsize=(img_size, img_size), interpolation=cv2.INTER_LANCZOS4)

            imgCrop = cv2.resize(imgCrop, (norm_size, norm_size))

            # cv2.imshow('imgCrop', imgCrop)
            # cv2.waitKey(0)

            imgCrop = img_to_array(imgCrop)

            label = labels[figure_count[i]]
            if label == -1:
                label = 0
            global_imgs.append(imgCrop)
            global_labels.append(label)
            i += 1

        global_imgs = np.array(global_imgs)
        print("Shape of global_imgs is ", global_imgs.shape)
        global_labels = np.array(global_labels)

        # global_labels = to_categorical(global_labels, num_classes=CLASS_NUM)
        # global_labels = np.array(global_labels)

    print("Finish image feature extraction......")

    return global_imgs, global_labels


def __crop(img, position, size):
    ow, oh = img.size
    x1, y1 = position
    tw = size[0]
    th = size[1]
    if (ow > tw and oh > th):
        return img.crop((x1, y1, x1 + tw, y1 + th))
    return img
