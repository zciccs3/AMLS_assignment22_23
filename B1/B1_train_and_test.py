import numpy as np
from matplotlib import pyplot as plt
from torchvision import transforms
import torch.utils.data as Data
from B1_image_preprocessing import B1_image_preprocessing
from sklearn.ensemble import RandomForestClassifier
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import B1_lab2_landmarks as l2
import B1_lab2_landmarks_test as l2_test
import B1_CNN
from os.path import dirname, abspath

base_dir = dirname(dirname(abspath(__file__)))


def RF_imageArray_train_and_test():

    # label source
    labels_dir = base_dir + '/Label_files'
    # Train sources
    train_labels_filename = 'cartoon_train_labels.csv'
    train_images_dir = base_dir + '/Datasets/cartoon_set'
    # Test sources
    test_labels_filename = 'cartoon_test_labels.csv'
    test_images_dir = base_dir + '/Datasets/cartoon_set_test'
    # Images resize
    img_size = 64

    # Train images and labels pre-processing
    X_train, y_train = B1_image_preprocessing(labels_dir, train_labels_filename, train_images_dir, img_size)
    # build the Random Forest model
    RF_model = RandomForestClassifier(n_estimators=50, n_jobs=-1)
    # Fit the Random Forest model
    RF_model.fit(X_train, y_train)

    # Test images and label pre-processing
    X_test, y_test = B1_image_preprocessing(labels_dir, test_labels_filename, test_images_dir, img_size)
    # Predict the output in testing stage
    y_prediction = RF_model.predict(X_test)

    # Calculate the classification accuracy
    Accuracy = accuracy_score(y_test, y_prediction)
    print("Accuracy rate of RF_imageArray method = ", Accuracy)

    # Generate the confusion matrix
    Confusion_matrix = confusion_matrix(y_test, y_prediction)
    Confusion_matrix = Confusion_matrix / Confusion_matrix.astype(np.float).sum(axis=1)
    # Plot the confusion matrix
    sns.set()
    f, ax = plt.subplots()
    sns.heatmap(Confusion_matrix, annot=True, ax=ax)
    # ax.set_title('Confusion matrix')
    ax.set_xlabel('Predictions')
    ax.set_ylabel('True labels')
    plt.show()
    # Generate the classification report
    # print(classification_report(y_test, y_prediction))

    return Accuracy, Confusion_matrix


def RF_landmarks_train_and_test():

    # Feature extraction
    X_train, Labels_train = l2.extract_features_labels()
    X_train = X_train.reshape(X_train.shape[0], X_train.shape[1] * X_train.shape[2])

    # create the model - Random Forest
    clf = RandomForestClassifier(n_estimators=100)

    # Training - fit the training data to the model
    clf.fit(X_train, Labels_train)

    # Testing
    X_test, Labels_test = l2_test.extract_features_labels()
    X_test = X_test.reshape(X_test.shape[0], X_test.shape[1] * X_test.shape[2])
    # predict label of test images
    prediction = clf.predict(X_test)
    print("Accuracy rate of RF_landmarks method = ", accuracy_score(Labels_test, prediction))

    return prediction


def B1_CNN_dataloader():

    # Training and testing image sources
    img_root_train = base_dir + '/Datasets/cartoon_set'
    img_root_test = base_dir + '/Datasets/cartoon_set_test'
    # Training and testing labels
    train_csv = base_dir + '/Label_files/cartoon_train_labels.csv'
    test_csv = base_dir + '/Label_files/cartoon_test_labels.csv'

    train_batch_size = 16
    test_batch_size = 4

    # Transformation of images
    transform = transforms.Compose([transforms.Resize((64, 64)),
                                    transforms.CenterCrop(48),
                                    transforms.Grayscale(num_output_channels=1),
                                    transforms.RandomHorizontalFlip(),
                                    transforms.ToTensor(),
                                    transforms.Normalize(mean=0.5, std=0.5)
                                    ])

    # Build the training and testing dataloader
    train_dataset = B1_CNN.Dataset(img_dir=img_root_train, train_csv=train_csv, transform=transform)
    train_dataloader = Data.DataLoader(train_dataset, batch_size=train_batch_size, shuffle=True)
    test_dataset = B1_CNN.Dataset(img_dir=img_root_test, train_csv=test_csv, transform=transform)
    test_dataloader = Data.DataLoader(test_dataset, batch_size=test_batch_size, shuffle=True)

    return train_dataloader, test_dataloader

