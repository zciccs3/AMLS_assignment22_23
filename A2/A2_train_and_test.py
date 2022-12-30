import torch.utils.data as Data
from torchvision import transforms
import A2_CNN_face_recognition
from torchvision.transforms import transforms
import A2_CNN_mouth
import A2_image_preprocessing
import os
from tensorflow.python.keras.utils.np_utils import to_categorical
from A2_SVM_model_wrapper import FCtoSVM
from A2_build_CNN_model import CNN_model
from A2_image_preprocessing import A2_CNN_SVM_load_images, A2_CNN_SVM_load_mouth_images
import matplotlib.pyplot as plt
from os.path import dirname, abspath

# Root path
base_dir = dirname(dirname(abspath(__file__)))


# Dataloader for CNN model with face detection on images
def A2_CNN_face_recognition_dataloader():

    # sources
    img_root_train = base_dir + '/Datasets/celeba'
    img_root_test = base_dir + '/Datasets/celeba_test'
    train_csv = base_dir + '/Label_files/celeba_train_labels.csv'
    test_csv = base_dir + '/Label_files/celeba_test_labels.csv'

    train_batch_size = 16
    test_batch_size = 8
    # Image transformation
    transform = transforms.Compose([transforms.Grayscale(num_output_channels=1),
                                    transforms.Resize((64, 64)),
                                    transforms.ToTensor(),
                                    transforms.Normalize(mean=0.5, std=0.5)
                                    ])
    # Training and testing dataloader
    train_dataset = A2_CNN_face_recognition.Dataset(img_dir=img_root_train, train_csv=train_csv, transform=transform)
    train_dataloader = Data.DataLoader(train_dataset, batch_size=train_batch_size, shuffle=True)
    test_dataset = A2_CNN_face_recognition.Dataset(img_dir=img_root_test, train_csv=test_csv, transform=transform)
    test_dataloader = Data.DataLoader(test_dataset, batch_size=test_batch_size, shuffle=True)

    return train_dataloader, test_dataloader


# Dataloader for CNN model with mouth localisation
def A2_CNN_mouth_localisation_dataloader():

    img_root_train = base_dir + '/Datasets/celeba'
    img_root_test = base_dir + '/Datasets/celeba_test'
    train_csv = base_dir + '/Label_files/celeba_train_labels.csv'
    test_csv = base_dir + '/Label_files/celeba_test_labels.csv'

    train_batch_size = 16
    test_batch_size = 8
    # Image transformation
    transform = transforms.Compose([transforms.Grayscale(num_output_channels=1),
                                    transforms.Lambda(lambda img: A2_image_preprocessing.__crop(img, (img.size[0] * 0.25, img.size[1] * 0.6),
                                                                         (img.size[0] * 0.5, img.size[1] * 0.25))),
                                    transforms.Resize((64, 64)),
                                    transforms.RandomHorizontalFlip(),
                                    transforms.ToTensor(),
                                    transforms.Normalize(mean=0.5, std=0.5)
                                    ])
    # Training and testing dataloader
    train_dataset = A2_CNN_mouth.Dataset(img_dir=img_root_train, train_csv=train_csv, transform=transform)
    train_dataloader = Data.DataLoader(train_dataset, batch_size=train_batch_size, shuffle=True)
    test_dataset = A2_CNN_mouth.Dataset(img_dir=img_root_test, train_csv=test_csv, transform=transform)
    test_dataloader = Data.DataLoader(test_dataset, batch_size=test_batch_size, shuffle=True)

    return train_dataloader, test_dataloader


# Training and testing of the hybrid CNN-SVM model
def CNN_SVM_full_image_train_and_test():
    # Labels
    labels_dir = base_dir + '/Label_files'
    # Train sources
    train_labels_filename = 'celeba_train_labels.csv'
    train_images_dir = os.path.join(base_dir + '/Datasets', 'celeba')
    # Test sources
    test_labels_filename = 'celeba_test_labels.csv'
    test_images_dir = os.path.join(base_dir + '/Datasets', 'celeba_test')

    # Image pre-processing and obtain the training features and labels
    X_train, y_train = A2_CNN_SVM_load_images(labels_dir, train_labels_filename, train_images_dir)
    # Image pre-processing and obtain the testing features and labels
    X_test, y_test = A2_CNN_SVM_load_images(labels_dir, test_labels_filename, test_images_dir)
    # Build the hybrid CNN-SVM
    wrapper = FCtoSVM(CNN_model())
    # Summarise model structure
    wrapper.model.summary()
    # Arrange extracted images to training and testing variables
    train_images = X_train
    train_labels = y_train
    test_images = X_test
    test_labels = y_test

    # Call the hybrid CNN-SVM model
    wrapper = FCtoSVM(CNN_model())

    epochs = 15
    performance = {
        "CNN + SVM": [],
        "CNN_softmax": []
    }

    # Training process to compare the performance between CNN-SVM and CNN-Softmax
    for i in range(epochs):
        print('Starting epoch: {}'.format(i + 1))
        wrapper.fit(train_images, train_labels, epochs=1, batch_size=16)
        performance["CNN + SVM"].append(wrapper.evaluate(test_images, test_labels))
        print("CNN+SVM: ", wrapper.evaluate(test_images, test_labels))
        performance["CNN_softmax"].append(wrapper.model.evaluate(test_images, to_categorical(test_labels))[1])
        print("CNN_softmax: ", wrapper.model.evaluate(test_images, to_categorical(test_labels))[1])

    # Plot the figure for accuracy comparison
    x = range(epochs)
    y1 = performance["CNN + SVM"]
    y2 = performance["CNN_softmax"]
    plt.plot(x, y1, color='red', linestyle='-', label='CNN + SVM')
    plt.plot(x, y2, color='blue', linestyle='-', label='CNN_softmax')
    plt.legend()
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.show()


# Training and testing of the hybrid CNN-SVM model with mouth localisation
def CNN_SVM_mouth_localisation_train_and_test():

    # Labels
    labels_dir = base_dir + '/Label_files'
    # Train sources
    train_labels_filename = 'celeba_train_labels.csv'
    train_images_dir = os.path.join(base_dir + '/Datasets', 'celeba')
    # Test sources
    test_labels_filename = 'celeba_test_labels.csv'
    test_images_dir = os.path.join(base_dir + '/Datasets', 'celeba_test')

    # Image pre-processing and obtain the training features and labels
    X_train, y_train = A2_CNN_SVM_load_mouth_images(labels_dir, train_labels_filename, train_images_dir)
    # Image pre-processing and obtain the testing features and labels
    X_test, y_test = A2_CNN_SVM_load_mouth_images(labels_dir, test_labels_filename, test_images_dir)
    # Build the hybrid CNN-SVM
    wrapper = FCtoSVM(CNN_model())
    # Summarise model structure
    wrapper.model.summary()
    # Arrange extracted images to training and testing variables
    train_images = X_train
    train_labels = y_train
    test_images = X_test
    test_labels = y_test

    # Call the hybrid CNN-SVM model
    wrapper = FCtoSVM(CNN_model())

    epochs = 15
    performance = {
        "CNN + SVM": [],
        "CNN_softmax": []
    }

    # Training process to compare the performance between CNN-SVM and CNN-Softmax
    for i in range(epochs):
        print('Starting epoch: {}'.format(i + 1))
        wrapper.fit(train_images, train_labels, epochs=1, batch_size=16)
        print(wrapper)
        performance["CNN + SVM"].append(wrapper.evaluate(test_images, test_labels))
        print("CNN+SVM: ", wrapper.evaluate(test_images, test_labels))
        performance["CNN_softmax"].append(wrapper.model.evaluate(test_images, to_categorical(test_labels))[1])
        print("CNN_softmax: ", wrapper.model.evaluate(test_images, to_categorical(test_labels))[1])

    # Plot the figure for accuracy comparison
    x = range(epochs)
    y1 = performance["CNN + SVM"]
    y2 = performance["CNN_softmax"]
    plt.plot(x, y1, color='red', marker='o', linestyle='-', label='CNN + SVM')
    plt.plot(x, y2, color='blue', marker='o', linestyle='-', label='CNN_softmax')
    plt.legend()
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.show()
