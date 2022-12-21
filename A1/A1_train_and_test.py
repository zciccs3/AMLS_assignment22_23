from sklearn.metrics import accuracy_score, confusion_matrix
from A1_image_preprocessing import Original_image_feature_extraction
from sklearn.linear_model import LogisticRegression
from A1_image_preprocessing import face_crop
import numpy as np


def A1_logisticRegression_full_image(train_basedir, train_labels_filename, train_images_dir,
                                     test_basedir, test_labels_filename, test_images_dir, img_size):
    X_train, y_train = Original_image_feature_extraction(train_basedir, train_labels_filename, train_images_dir, img_size)
    logistic_model = LogisticRegression()
    logistic_model.fit(X_train, y_train)
    print("Finish training......")
    X_test, y_test = Original_image_feature_extraction(test_basedir, test_labels_filename, test_images_dir, img_size)
    y_prediction = logistic_model.predict(X_test)

    Accuracy = accuracy_score(y_test, y_prediction)
    print("Accuracy rate = ", Accuracy)
    Confusion_matrix = confusion_matrix(y_test, y_prediction)
    print(Confusion_matrix)

    return Accuracy, Confusion_matrix


def A1_logisticRegression_face_recognition(train_basedir, train_labels_filename, train_images_dir,
                                           test_basedir, test_labels_filename, test_images_dir):
    # Train images and labels pre-processing
    X_train, y_train = face_crop(train_basedir, train_labels_filename, train_images_dir)

    # print(X_train)
    x_train = []
    x_length = []
    x_train_final = []

    for i in range(len(X_train)):
        x_train.append(X_train[i][0].tolist())
        x_length.append(len(X_train[i][0].tolist()))

    min_length = min(x_length)
    min_index = x_length.index(min_length)
    for i in range(len(x_train)):
        x_train_final.append(x_train[i][min_length])

    x_train_input = np.concatenate(x_train_final, axis=0)

    logistic_model = LogisticRegression()
    logistic_model.fit(x_train_input, y_train)

    print("Finish training......")

    # Test images and label pre-processing
    X_test, y_test = face_crop(test_basedir, test_labels_filename, test_images_dir)
    x_test = []
    for i in range(len(X_test)):
        x_test.append(X_test[i][0].tolist())
    x_test = np.concatenate(x_test, axis=0)
    y_prediction = logistic_model.predict(x_test)
    Accuracy = accuracy_score(y_test, y_prediction)
    print("Accuracy rate = ", Accuracy)
    Confusion_matrix = confusion_matrix(y_test, y_prediction)
    print(Confusion_matrix)
    return Accuracy, Confusion_matrix
