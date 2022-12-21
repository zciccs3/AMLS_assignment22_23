import os
import face_recognition
import numpy as np
from keras.preprocessing import image
import cv2
import dlib
import pandas as pd


detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('D:/Nottingham/FYP_Chenhao Shi/AMLS_Assignment2022_23_SN22073551/shape_predictor_68_face_landmarks.dat')


def rect_to_bb(rect):
    x = rect.left()
    y = rect.top()
    w = rect.right() - x
    h = rect.bottom() - y
    return (x, y, w, h)


def face_crop(basedir, labels_filename, images_dir):
    detector = dlib.get_frontal_face_detector()
    dataframe = pd.read_csv(os.path.join(basedir, labels_filename), delimiter='\t')
    img_list = dataframe.values[:, 1]  # Image file names
    labels = dataframe.values[:, 2]  # Gender labels
    figure_count = dataframe.values[:, 0]
    imgs = [os.path.join(images_dir, file) for file in img_list]

    print("Face recognition...")

    if os.path.isdir(images_dir):
        global_imgs = []
        global_labels = []
        i = 0
        target_size = None
        for img_path in imgs:
            # img = image.img_to_array(image.load_img(img_path,
            #                     target_size=target_size,
            #                     interpolation='bicubic'))
            #
            # resized_image = img.astype('uint8')
            # gray = cv2.cvtColor(resized_image, cv2.COLOR_BGR2GRAY)
            # gray = gray.astype('uint8')
            gray = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            # gray = cv2.imread(img_path)

            rects = detector(gray, 1)
            # num_faces = len(rects)
            # if num_faces == 0:
            #     return None

            for _, rect in enumerate(rects):
                (x, y, w, h) = rect_to_bb(rect)
                print(x, y, w, h)

            faceCrop = gray[y:y + h, x:x + w].copy()

            # cv2.imshow('faceCrop', faceCrop)
            # cv2.waitKey(0)

            label = labels[figure_count[i]]
            global_imgs.append(faceCrop)
            global_labels.append(label)

            i += 1

            if i % 1000 == 0:
                print("Already process images ", i)

        global_imgs = np.array(global_imgs).reshape(len(global_imgs), -1)
        global_labels = np.array(global_labels)

    print("Finish face recognition......")

    return global_imgs, global_labels


def Original_image_feature_extraction(basedir, labels_filename, images_dir, img_size):
    dataframe = pd.read_csv(os.path.join(basedir, labels_filename), delimiter='\t')
    img_list = dataframe.values[:, 1]  # Image file names
    labels = dataframe.values[:, 2]  # Gender labels
    figure_count = dataframe.values[:, 0]
    imgs = [os.path.join(images_dir, file) for file in img_list]

    print("Image feature extraction......")

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
            if i % 1000 == 0:
                print("Already process images ", i)

        global_imgs = np.array(global_imgs).reshape(len(global_imgs), -1)
        global_labels = np.array(global_labels)

    print("Finish image feature extraction......")

    return global_imgs, global_labels