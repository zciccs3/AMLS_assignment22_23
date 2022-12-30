import os
import numpy as np
import cv2
import dlib
import pandas as pd
from os.path import dirname, abspath

# Root path
base_dir = dirname(dirname(abspath(__file__)))

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(base_dir + '/Good models and results/shape_predictor_68_face_landmarks.dat')


# Determine the size and coordinates of the provided cropped area
def rect_to_bb(rect):
    x = rect.left()
    y = rect.top()
    w = rect.right() - x
    h = rect.bottom() - y
    return (x, y, w, h)


# Face detection and crop facial area; convert to arrays as features
def face_crop(labels_dir, labels_filename, images_dir):
    detector = dlib.get_frontal_face_detector()
    dataframe = pd.read_csv(os.path.join(labels_dir, labels_filename), delimiter='\t')
    img_list = dataframe.values[:, 1]  # Image file names
    labels = dataframe.values[:, 2]  # Gender labels
    figure_count = dataframe.values[:, 0]
    imgs = [os.path.join(images_dir, file) for file in img_list]

    if os.path.isdir(images_dir):
        global_imgs = []
        global_labels = []
        i = 0
        target_size = None
        for img_path in imgs:
            gray = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            rects = detector(gray, 1)
            for _, rect in enumerate(rects):
                (x, y, w, h) = rect_to_bb(rect)
                print(x, y, w, h)
            faceCrop = gray[y:y + h, x:x + w].copy()

            label = labels[figure_count[i]]
            global_imgs.append(faceCrop)
            global_labels.append(label)

            i += 1

        global_imgs = np.array(global_imgs).reshape(len(global_imgs), -1)
        global_labels = np.array(global_labels)

    return global_imgs, global_labels


# Extract features from the original image after grayscale conversion, resizing and array conversion
def Original_image_feature_extraction(labels_dir, labels_filename, images_dir, img_size):
    dataframe = pd.read_csv(os.path.join(labels_dir, labels_filename), delimiter='\t')
    img_list = dataframe.values[:, 1]  # Image file names
    labels = dataframe.values[:, 2]  # Gender labels
    figure_count = dataframe.values[:, 0]
    imgs = [os.path.join(images_dir, file) for file in img_list]

    if os.path.isdir(images_dir):
        global_imgs = []
        global_labels = []
        i = 0
        for img_path in imgs:
            # Gray images transformation
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            # Images resize
            img = cv2.resize(src=img, dsize=(img_size, img_size), interpolation=cv2.INTER_LANCZOS4)
            label = labels[figure_count[i]]
            global_imgs.append(img)
            global_labels.append(label)
            i += 1

        # Convert to arrays as features and reshape the array
        global_imgs = np.array(global_imgs).reshape(len(global_imgs), -1)
        global_labels = np.array(global_labels)

    return global_imgs, global_labels