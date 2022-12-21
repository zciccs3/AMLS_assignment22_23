from B2_train_and_test import B2_random_forest_double_eyes_train_and_test, \
    B2_random_forest_single_eye_train_and_test, B2_random_forest_single_eye_glass_removal_train_and_test
from B2_image_preprocessing import B2_image_preprocessing_double_eyes, image_transformation_CNN
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from B2_Eyes_color_recognition_CNN import Dataset, train, test_image_examples, test_accuracy_rate
import torch.utils.data as Data


# Train sources
train_basedir = 'D:/Nottingham/FYP_Chenhao Shi/AMLS_Assignment2022_23_SN22073551/dataset_AMLS_22-23/cartoon_set'
train_labels_filename = 'labels.csv'
train_images_dir = os.path.join(train_basedir, 'img')
# Test sources
test_basedir = 'D:/Nottingham/FYP_Chenhao Shi/AMLS_Assignment2022_23_SN22073551/dataset_AMLS_22-23_test/cartoon_set_test'
test_labels_filename = 'labels.csv'
test_images_dir = os.path.join(test_basedir, 'img')


# Method 1
# B2 double eyes image training and testing - Accuracy rate = 85.12%
Accuracy_double_eyes = B2_random_forest_double_eyes_train_and_test(train_basedir, train_labels_filename,
                                     train_images_dir, test_basedir, test_labels_filename, test_images_dir)
print("Accuracy rate for B2 with double eyes images = ", round(Accuracy_double_eyes, 4))


# Method 2
# B2 single eye image training and testing - Accuracy rate = 85.32%
Accuracy_single_eye = B2_random_forest_single_eye_train_and_test(train_basedir, train_labels_filename,
                                     train_images_dir, test_basedir, test_labels_filename, test_images_dir)
print("Accuracy rate for B2 with single eye images = ", round(Accuracy_single_eye, 4))


# Method 3
# B2 single eye image glass removal training and testing - Accuracy rate = 100%
Accuracy_glass_removal = B2_random_forest_single_eye_glass_removal_train_and_test(train_basedir, train_labels_filename,
                                     train_images_dir, test_basedir, test_labels_filename, test_images_dir)
print("Accuracy rate for B2 with single eye images = ", round(Accuracy_glass_removal, 4))


# Method 4
# B2 double eyes image with CNN without glass removal - Accuracy rate = 85.44%
classes = (0, 1, 2, 3, 4)
img_root_train = train_images_dir
img_root_test = test_images_dir
train_csv = os.path.join(train_basedir, train_labels_filename)
test_csv = os.path.join(test_basedir, test_labels_filename)

train_batch_size = 16
test_batch_size = 4

transform = image_transformation_CNN()

train_dataset = Dataset(img_dir=img_root_train, train_csv=train_csv, transform=transform)
train_dataloader = Data.DataLoader(train_dataset, batch_size=train_batch_size, shuffle=True)
test_dataset = Dataset(img_dir=img_root_test, train_csv=test_csv, transform=transform)
test_dataloader = Data.DataLoader(test_dataset, batch_size=test_batch_size, shuffle=True)

train(train_dataloader)
test_image_examples(test_dataloader, classes, test_batch_size)
test_accuracy_rate(test_dataloader, test_batch_size)
