import numpy as np
import pandas as pd
import os
import cv2
import torchvision.transforms as transforms
from os.path import dirname, abspath

# Root path
base_dir = dirname(dirname(abspath(__file__)))


# Image pre-processing to crop double eyes and convert to arrays
def B2_image_preprocessing_double_eyes(labels_dir, labels_filename, images_dir):
    dataframe = pd.read_csv(os.path.join(labels_dir, labels_filename), delimiter='\t')
    img_list = dataframe.values[:, 3]
    labels = dataframe.values[:, 1]  # Eyes color labels
    figure_count = dataframe.values[:, 0]
    imgs = [os.path.join(images_dir, file) for file in img_list]

    if os.path.isdir(images_dir):
        global_imgs = []
        global_labels = []
        i = 0
        for img_path in imgs:
            # cv2
            img = cv2.imread(img_path)
            # Images resize
            xmin = int(img.shape[0] * 0.35)
            ymin = int(img.shape[1] * 0.49)
            w = int(img.shape[0] * 0.3)
            h = int(img.shape[1] * 0.07)
            imgCrop = img[ymin:ymin + h, xmin:xmin + w].copy()

            label = labels[figure_count[i]]
            global_imgs.append(imgCrop)
            global_labels.append(label)
            i += 1

        # Convert to arrays
        global_imgs = np.array(global_imgs).reshape(len(global_imgs), -1)
        global_labels = np.array(global_labels)

    return global_imgs, global_labels


# Image pre-processing to crop single eye and convert to arrays
def B2_image_preprocessing_single_eye(labels_dir, labels_filename, images_dir):
    dataframe = pd.read_csv(os.path.join(labels_dir, labels_filename), delimiter='\t')
    img_list = dataframe.values[:, 3]
    labels = dataframe.values[:, 1]  # Eyes color labels
    figure_count = dataframe.values[:, 0]
    imgs = [os.path.join(images_dir, file) for file in img_list]

    if os.path.isdir(images_dir):
        global_imgs = []
        global_labels = []
        i = 0
        for img_path in imgs:
            # cv2
            img = cv2.imread(img_path)
            # Images resize
            xmin = int(img.shape[0] * 0.373)
            ymin = int(img.shape[1] * 0.49)
            w = int(img.shape[0] * 0.085)
            h = int(img.shape[1] * 0.07)
            imgCrop = img[ymin:ymin + h, xmin:xmin + w].copy()

            label = labels[figure_count[i]]
            global_imgs.append(imgCrop)
            global_labels.append(label)
            i += 1

        # Convert to arrays
        global_imgs = np.array(global_imgs).reshape(len(global_imgs), -1)
        global_labels = np.array(global_labels)

    return global_imgs, global_labels


# Image pre-processing to crop single eye when images whose face is covered by sunglasses are removed
def B2_image_preprocessing_single_eye_glass_removal(labels_dir, labels_filename, images_dir):
    dataframe = pd.read_csv(os.path.join(labels_dir, labels_filename), delimiter='\t')
    img_list = dataframe.values[:, 3]
    labels = dataframe.values[:, 1]  # Eyes color labels
    figure_count = dataframe.values[:, 0]
    imgs = [os.path.join(images_dir, file) for file in img_list]

    if os.path.isdir(images_dir):
        global_imgs = []
        global_labels = []
        index_without_glass = []
        i = 0
        for img_path in imgs:
            # cv2
            img = cv2.imread(img_path)
            # Images resize
            xmin = int(img.shape[0] * 0.373)
            ymin = int(img.shape[1] * 0.49)
            w = int(img.shape[0] * 0.085)
            h = int(img.shape[1] * 0.07)
            imgCrop = img[ymin:ymin + h, xmin:xmin + w].copy()

            # Determine whether wearing sunglasses or not
            if np.array(imgCrop).mean() > 60:
                # Append images without sunglasses
                global_imgs.append(imgCrop)
                global_labels.append(labels[figure_count[i]])
                index_without_glass.append(i)

            i += 1

        # Convert to arrays
        global_imgs = np.array(global_imgs).reshape(len(global_imgs), -1)
        global_labels = np.array(global_labels)
        global_index = np.array(index_without_glass)

    return global_imgs, global_labels, global_index


# crop the image for CNN method
def __crop(img, position, size):
    ow, oh = img.size
    x1, y1 = position
    tw = size[0]
    th = size[1]
    if (ow > tw and oh > th):
        return img.crop((x1, y1, x1 + tw, y1 + th))
    return img


# CNN image transformation
def image_transformation_CNN():
    transform = transforms.Compose([transforms.Lambda(lambda img: __crop(img, (int(img.size[0]*0.35), int(img.size[1]*0.493)), (int(img.size[0]*0.3), int(img.size[1]*0.07)))),
                                    transforms.Resize((48, 48)),
                                    transforms.ToTensor(),
                                    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
                                    ])
    return transform

