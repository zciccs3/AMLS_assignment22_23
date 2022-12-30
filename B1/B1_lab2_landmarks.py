import os
import numpy as np
from keras.preprocessing import image
import cv2
import dlib
import pandas as pd
from os.path import dirname, abspath

# Root path
base_dir = dirname(dirname(abspath(__file__)))

# Paths to images
# global labels_dir, image_paths, target_size
labels_dir = base_dir + '/Label_files'
images_dir = base_dir + '/Datasets/cartoon_set'
labels_filename = 'cartoon_train_labels.csv'

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(base_dir + '/Good models and results/shape_predictor_68_face_landmarks.dat')


# Convert the facial landmarks to 2-tuple coordinates
def shape_to_np(shape, dtype="int"):
    # initialize the list of (x, y)-coordinates
    coords = np.zeros((shape.num_parts, 2), dtype=dtype)

    for i in range(0, shape.num_parts):
        coords[i] = (shape.part(i).x, shape.part(i).y)

    # return the list of (x, y)-coordinates
    return coords


# Convert the bounding of faces to coordinates
def rect_to_bb(rect):
    # take a bounding predicted by dlib and convert it
    # to the format (x, y, w, h) as we would normally do
    # with OpenCV
    x = rect.left()
    y = rect.top()
    w = rect.right() - x
    h = rect.bottom() - y

    return (x, y, w, h)


# Load images, detect facial landmarks and return them
def run_dlib_shape(image):
    # load the input image, resize it, and convert it to grayscale
    resized_image = image.astype('uint8')
    gray = cv2.cvtColor(resized_image, cv2.COLOR_BGR2GRAY)
    gray = gray.astype('uint8')

    # detect faces in the grayscale image
    rects = detector(gray, 1)
    num_faces = len(rects)

    if num_faces == 0:
        return None, resized_image

    face_areas = np.zeros((1, num_faces))
    face_shapes = np.zeros((136, num_faces), dtype=np.int64)

    # loop over the face detections
    for (i, rect) in enumerate(rects):
        # Determine the facial landmarks for the face region, convert landmark coordinates to a numPy array
        temp_shape = predictor(gray, rect)
        temp_shape = shape_to_np(temp_shape)

        # Convert dlib rectangle to a OpenCV-style bounding box
        (x, y, w, h) = rect_to_bb(rect)
        face_shapes[:, i] = np.reshape(temp_shape, [136])
        face_areas[0, i] = w * h
    # find largest face and keep
    dlibout = np.reshape(np.transpose(face_shapes[:, np.argmax(face_areas)]), [68, 2])

    return dlibout, resized_image


# Extract facial landmarks as features and return them
def extract_features_labels():

    dataframe = pd.read_csv(os.path.join(labels_dir, labels_filename), delimiter='\t')
    train_list = dataframe.values[:, 3]
    train_labels = dataframe.values[:, 2]  # Face shape labels
    train_figure_count = dataframe.values[:, 0]
    imgs = [os.path.join(images_dir, file) for file in train_list]

    target_size = None

    if os.path.isdir(images_dir):
        all_features = []
        all_labels = []
        i = 0
        for img_path in imgs:
            # load image
            img = image.img_to_array(
                image.load_img(img_path,
                               target_size=target_size,
                               interpolation='bicubic'))
            features, _ = run_dlib_shape(img)
            if features is not None:
                all_features.append(features)
                all_labels.append(train_labels[train_figure_count[i]])
            i += 1

    # Convert landmark features into arrays
    landmark_features = np.array(all_features)
    face_shape_labels = np.array(all_labels)

    return landmark_features, face_shape_labels