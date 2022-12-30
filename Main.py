import sys
sys.path.append('A1')
sys.path.append('A2')
sys.path.append('B1')
sys.path.append('B2')
import os
import A1.A1_CNN
import A2.A2_CNN_mouth
import A2.A2_CNN_face_recognition
import B1.B1_CNN
import B2.B2_Eyes_color_recognition_CNN
from A1.A1_train_and_test import A1_logisticRegression_full_image, A1_CNN_dataloader
from A2.A2_train_and_test import A2_CNN_face_recognition_dataloader, A2_CNN_mouth_localisation_dataloader, \
    CNN_SVM_full_image_train_and_test, CNN_SVM_mouth_localisation_train_and_test
from B1.B1_train_and_test import RF_imageArray_train_and_test, RF_landmarks_train_and_test, B1_CNN_dataloader
from B2.B2_train_and_test import B2_random_forest_double_eyes_train_and_test, \
    B2_random_forest_single_eye_train_and_test, B2_random_forest_single_eye_glass_removal_train_and_test, B2_CNN_dataloader


# Labels
labels_dir = 'Label_files'
# Train sources
celeba_train_labels_filename = 'celeba_train_labels.csv'
celeba_train_images_dir = os.path.join('Datasets', 'celeba')
cartoon_train_labels_filename = 'cartoon_train_labels.csv'
cartoon_train_images_dir = 'Datasets/cartoon_set'
# Test sources
celeba_test_labels_filename = 'celeba_test_labels.csv'
celeba_test_images_dir = os.path.join('Datasets', 'celeba_test')
cartoon_test_labels_filename = 'cartoon_test_labels.csv'
cartoon_test_images_dir = 'Datasets/cartoon_set_test'
# Classes
A1_classes = ('female', 'male')
A2_classes = ('Not', 'smiling')
B1_classes = (0, 1, 2, 3, 4)
B2_classes = (0, 1, 2, 3, 4)
# Testing batch size
test_batch_size_A1 = 8
test_batch_size_A2 = 8
test_batch_size_B1 = 4
test_batch_size_B2 = 4


# A1 ==================================================================================================================
# Method 1 - CNN with full image
print('Starting A1-method 2: CNN with full image...')
train_dataloader_A1, test_dataloader_A1 = A1_CNN_dataloader()
# Train CNN model
A1.A1_CNN.train(train_dataloader_A1)
# Show examples of processed images and classification results
A1.A1_CNN.test_image_examples(test_dataloader_A1, A1_classes, test_batch_size_A1)
# Test the classification accuracy
A1.A1_CNN.test_accuracy_rate(test_dataloader_A1, test_batch_size_A1)
# ------------------------------------------------------------------------------------------------
# Method 2 - Logistic Regression with full image
print('Starting A1-method 1: Logistic Regression with full image...')
img_size = 128
Accuracy, _ = A1_logisticRegression_full_image(labels_dir, celeba_train_labels_filename, celeba_train_images_dir,
                                      celeba_test_labels_filename, celeba_test_images_dir, img_size)


# # A2 =================================================================================================================
# method 1 - CNN with face recognition
print('Starting A2-method 1: CNN with face recognition...')
train_dataloader1, test_dataloader1 = A2_CNN_face_recognition_dataloader()
# Train CNN model
A2.A2_CNN_face_recognition.train(train_dataloader1)
# Show examples of processed images and classification results
A2.A2_CNN_face_recognition.test_image_examples(test_dataloader1, A2_classes, test_batch_size_A2)
# Test the classification accuracy
A2.A2_CNN_face_recognition.test_accuracy_rate(test_dataloader1, test_batch_size_A2)
# ------------------------------------------------------------------------------------------------
# method 2 - CNN with mouth localisation
print('Starting A2-method 2: CNN with mouth localisation...')
train_dataloader2, test_dataloader2 = A2_CNN_mouth_localisation_dataloader()
# Train CNN model
A2.A2_CNN_mouth.train(train_dataloader2)
# Show examples of processed images and classification results
A2.A2_CNN_mouth.test_image_examples(test_dataloader2, A2_classes, test_batch_size_A2)
# Test the classification accuracy
A2.A2_CNN_mouth.test_accuracy_rate(test_dataloader2, test_batch_size_A2)
# ------------------------------------------------------------------------------------------------
# method 3 - hybrid CNN-SVM with full image
print('Starting A2-method 3: hybrid CNN-SVM with full image...')
CNN_SVM_full_image_train_and_test()
# ------------------------------------------------------------------------------------------------
# method 4 - hybrid CNN-SVM with mouth localisation
print('Starting A2-method 4: hybrid CNN-SVM with mouth localisation...')
CNN_SVM_mouth_localisation_train_and_test()


# B1 ===============================================================================================================
# Method 1 - CNN
print('Starting B1-method 1: CNN...')
train_dataloader_B1, test_dataloader_B1 = B1_CNN_dataloader()
# Train the CNN model
B1.B1_CNN.train(train_dataloader_B1)
# Show examples of processed images and classification results
B1.B1_CNN.test_image_examples(test_dataloader_B1, B1_classes, test_batch_size_B1)
# Test the classification accuracy
B1.B1_CNN.test_accuracy_rate(test_dataloader_B1, test_batch_size_B1)
# ------------------------------------------------------------------------------------------------
# Method 2 - Random Forest_imageArray method
print('Starting B1-method 2: Random Forest_imageArray method...')
Accuracy_RF_imageArray, Confusion_matrix = RF_imageArray_train_and_test()
# ------------------------------------------------------------------------------------------------
# Method 3 - Random Forest_landmarks method
print('Starting B1-method 3: Random Forest_landmarks method...')
Prediction = RF_landmarks_train_and_test()


# B2 ===============================================================================================================
# Method 1 - RF with double eyes image
print('Starting B2-method 1: RF with double eyes image...')
Accuracy_double_eyes = B2_random_forest_double_eyes_train_and_test(labels_dir, cartoon_train_labels_filename,
                                     cartoon_train_images_dir, cartoon_test_labels_filename, cartoon_test_images_dir)
print("Accuracy = ", round(Accuracy_double_eyes, 4))
# ------------------------------------------------------------------------------------------------
# Method 2 - RF with single eye image
print('Starting B2-method 2: RF with single eye image...')
Accuracy_single_eye = B2_random_forest_single_eye_train_and_test(labels_dir, cartoon_train_labels_filename,
                                     cartoon_train_images_dir, cartoon_test_labels_filename, cartoon_test_images_dir)
print("Accuracy rate for B2 with RF model single eye images = ", round(Accuracy_single_eye, 4))
# ------------------------------------------------------------------------------------------------
# Method 3 - RF with single eye image glass removal
print('Starting B2-method 3: RF with single eye image glass removal...')
Accuracy_glass_removal = B2_random_forest_single_eye_glass_removal_train_and_test(labels_dir, cartoon_train_labels_filename,
                                     cartoon_train_images_dir, cartoon_test_labels_filename, cartoon_test_images_dir)
print("Accuracy rate for B2 with RF model single eye images and sunglasses image removal= ", round(Accuracy_glass_removal, 4))
# ------------------------------------------------------------------------------------------------
# Method 4 - CNN with double eyes image without glass removal
print('Starting B2-method 4: CNN with double eyes image without glass removal...')
train_dataloader_B2, test_dataloader_B2 = B2_CNN_dataloader()
# Train CNN model
B2.B2_Eyes_color_recognition_CNN.train(train_dataloader_B2)
# Show examples of processed images and classification results
B2.B2_Eyes_color_recognition_CNN.test_image_examples(test_dataloader_B2, B2_classes, test_batch_size_B2)
# Test the classification accuracy
B2.B2_Eyes_color_recognition_CNN.test_accuracy_rate(test_dataloader_B2, test_batch_size_B2)