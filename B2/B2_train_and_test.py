import os
import numpy as np
from matplotlib import pyplot as plt
from B2_image_preprocessing import B2_image_preprocessing_double_eyes, \
    B2_image_preprocessing_single_eye, B2_image_preprocessing_single_eye_glass_removal, image_transformation_CNN
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
import torch.utils.data as Data
import B2_Eyes_color_recognition_CNN
from os.path import dirname, abspath

# Root path
base_dir = dirname(dirname(abspath(__file__)))


# Training and testing of Random Forest model with double-eye localisation
def B2_random_forest_double_eyes_train_and_test(labels_dir, train_labels_filename, train_images_dir
                                                , test_labels_filename, test_images_dir):

    # Train images and labels pre-processing
    X_train, y_train = B2_image_preprocessing_double_eyes(labels_dir, train_labels_filename, train_images_dir)
    RF_model = RandomForestClassifier(n_estimators=50, n_jobs=-1)
    RF_model.fit(X_train, y_train)

    # Test images and label pre-processing
    X_test, y_test = B2_image_preprocessing_double_eyes(labels_dir, test_labels_filename, test_images_dir)
    y_prediction = RF_model.predict(X_test)

    Accuracy = accuracy_score(y_test, y_prediction)

    return Accuracy


# Training and testing of Random Forest model with single-eye localisation
def B2_random_forest_single_eye_train_and_test(labels_dir, train_labels_filename, train_images_dir,
                                                test_labels_filename, test_images_dir):

    # Train images and labels pre-processing
    X_train, y_train = B2_image_preprocessing_single_eye(labels_dir, train_labels_filename, train_images_dir)
    RF_model = RandomForestClassifier(n_estimators=50, n_jobs=-1)
    RF_model.fit(X_train, y_train)

    # Test images and label pre-processing
    X_test, y_test = B2_image_preprocessing_single_eye(labels_dir, test_labels_filename, test_images_dir)
    y_prediction = RF_model.predict(X_test)

    Accuracy = accuracy_score(y_test, y_prediction)

    return Accuracy


# Training and testing of Random Forest model with single-eye localisation and sunglasses removal
def B2_random_forest_single_eye_glass_removal_train_and_test(labels_dir, train_labels_filename, train_images_dir,
                                                test_labels_filename, test_images_dir):

    # Train images and labels pre-processing
    X_train, y_train, index = B2_image_preprocessing_single_eye_glass_removal(labels_dir, train_labels_filename, train_images_dir)
    RF_model = RandomForestClassifier(n_estimators=50, n_jobs=-1)
    RF_model.fit(X_train, y_train)

    # Test images and label pre-processing
    X_test, y_test, index = B2_image_preprocessing_single_eye_glass_removal(labels_dir, test_labels_filename, test_images_dir)
    y_prediction = RF_model.predict(X_test)

    Accuracy = accuracy_score(y_test, y_prediction)

    Confusion_matrix = confusion_matrix(y_test, y_prediction)
    Confusion_matrix = Confusion_matrix / Confusion_matrix.astype(np.float).sum(axis=1)
    sns.set()
    f, ax = plt.subplots()
    sns.heatmap(Confusion_matrix, annot=True, ax=ax)
    ax.set_xlabel('Predictions')
    ax.set_ylabel('True labels')
    plt.show()

    print(classification_report(y_test, y_prediction))

    return Accuracy


# CNN modol dataloader
def B2_CNN_dataloader():

    # Sources
    train_images_dir = base_dir + '/Datasets/cartoon_set'
    test_images_dir = base_dir + '/Datasets/cartoon_set_test'
    labels_dir = base_dir + '/Label_files'
    train_labels_filename = 'cartoon_train_labels.csv'
    test_labels_filename = 'cartoon_test_labels.csv'

    img_root_train = train_images_dir
    img_root_test = test_images_dir
    train_csv = os.path.join(labels_dir, train_labels_filename)
    test_csv = os.path.join(labels_dir, test_labels_filename)

    train_batch_size = 16
    test_batch_size = 4

    transform = image_transformation_CNN()

    # Load training and testing data batches
    train_dataset = B2_Eyes_color_recognition_CNN.Dataset(img_dir=img_root_train, train_csv=train_csv, transform=transform)
    train_dataloader = Data.DataLoader(train_dataset, batch_size=train_batch_size, shuffle=True)
    test_dataset = B2_Eyes_color_recognition_CNN.Dataset(img_dir=img_root_test, train_csv=test_csv, transform=transform)
    test_dataloader = Data.DataLoader(test_dataset, batch_size=test_batch_size, shuffle=True)

    return train_dataloader, test_dataloader