import os
import torch.utils.data as Data
from sklearn.metrics import accuracy_score, confusion_matrix
from torchvision.transforms import transforms
from A1_image_preprocessing import Original_image_feature_extraction, face_crop
from sklearn.linear_model import LogisticRegression
import numpy as np
import A1_CNN
from os.path import dirname, abspath

# Root path
base_dir = dirname(dirname(abspath(__file__)))


# Training and testing of logistic regression model with full image array as features
def A1_logisticRegression_full_image(labels_dir, train_labels_filename, train_images_dir
                                     , test_labels_filename, test_images_dir, img_size):
    # Extract training features and labels
    X_train, y_train = Original_image_feature_extraction(labels_dir, train_labels_filename, train_images_dir, img_size)
    # Build the model
    logistic_model = LogisticRegression()
    # Fit the model
    logistic_model.fit(X_train, y_train)
    # Extract testing features and true labels
    X_test, y_test = Original_image_feature_extraction(labels_dir, test_labels_filename, test_images_dir, img_size)
    y_prediction = logistic_model.predict(X_test)
    # Calculate classification accuracy
    Accuracy = accuracy_score(y_test, y_prediction)
    print("Accuracy rate = ", Accuracy)
    Confusion_matrix = confusion_matrix(y_test, y_prediction)

    return Accuracy, Confusion_matrix


# Logistic regression with face detection as features
def A1_logisticRegression_face_recognition(labels_dir, train_labels_filename, train_images_dir
                                           , test_labels_filename, test_images_dir):
    # Train images and labels pre-processing
    X_train, y_train = face_crop(labels_dir, train_labels_filename, train_images_dir)

    # print(X_train)
    x_train = []
    x_length = []
    x_train_final = []

    for i in range(len(X_train)):
        x_train.append(X_train[i][0].tolist())
        x_length.append(len(X_train[i][0].tolist()))

    min_length = min(x_length)
    for i in range(len(x_train)):
        x_train_final.append(x_train[i][min_length])
    # Concatenate all input arrays
    x_train_input = np.concatenate(x_train_final, axis=0)
    # build the model
    logistic_model = LogisticRegression()
    # Fit the model
    logistic_model.fit(x_train_input, y_train)

    # Test images and label pre-processing
    X_test, y_test = face_crop(labels_dir, test_labels_filename, test_images_dir)
    x_test = []
    for i in range(len(X_test)):
        x_test.append(X_test[i][0].tolist())
    x_test = np.concatenate(x_test, axis=0)
    y_prediction = logistic_model.predict(x_test)
    Accuracy = accuracy_score(y_test, y_prediction)
    print("Accuracy rate = ", Accuracy)
    Confusion_matrix = confusion_matrix(y_test, y_prediction)

    return Accuracy, Confusion_matrix


# CNN model dataloader for task A1
def A1_CNN_dataloader():

    # sourcess
    train_images_dir = os.path.join(base_dir + '/Datasets/celeba')
    test_images_dir = os.path.join(base_dir + '/Datasets/celeba_test')
    labels_dir = base_dir + '/Label_files'
    train_labels_filename = 'celeba_train_labels.csv'
    test_labels_filename = 'celeba_test_labels.csv'

    img_root_train = train_images_dir
    img_root_test = test_images_dir
    train_csv = os.path.join(labels_dir, train_labels_filename)
    test_csv = os.path.join(labels_dir, test_labels_filename)

    train_batch_size = 64
    test_batch_size = 8
    # Image format transformation
    transform = transforms.Compose([transforms.Resize((64, 64)),
                                    transforms.CenterCrop(64),
                                    transforms.RandomHorizontalFlip(),
                                    transforms.ToTensor(),
                                    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
                                    ])
    # load the training and testing data in batches
    train_dataset = A1_CNN.Dataset(img_dir=img_root_train, train_csv=train_csv, transform=transform)
    train_dataloader = Data.DataLoader(train_dataset, batch_size=train_batch_size, shuffle=True)
    test_dataset = A1_CNN.Dataset(img_dir=img_root_test, train_csv=test_csv, transform=transform)
    test_dataloader = Data.DataLoader(test_dataset, batch_size=test_batch_size, shuffle=True)

    return train_dataloader, test_dataloader
