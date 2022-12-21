import os
import torch.utils.data as Data
from torchvision.transforms import transforms
from A1_CNN import Dataset, train, test_image_examples, test_accuracy_rate
from A1_train_and_test import A1_logisticRegression_full_image, A1_logisticRegression_face_recognition

train_basedir = 'D:/Nottingham/FYP_Chenhao Shi/AMLS_Assignment2022_23_SN22073551/dataset_AMLS_22-23/celeba'
train_labels_filename = 'labels.csv'
train_images_dir = os.path.join(train_basedir, 'img')
# Test sources
test_basedir = 'D:/Nottingham/FYP_Chenhao Shi/AMLS_Assignment2022_23_SN22073551/dataset_AMLS_22-23_test/celeba_test'
test_labels_filename = 'labels.csv'
test_images_dir = os.path.join(test_basedir, 'img')

img_size = 128

# Method 1
# Logistic Regression with full image: Accuracy rate = 88.1% but not convergent
# Accuracy, _ = A1_logisticRegression_full_image(train_basedir, train_labels_filename, train_images_dir,
#                                      test_basedir, test_labels_filename, test_images_dir, img_size)

# Method 2
# Logistic Regression with face recognition
# Accuracy2, _ = A1_logisticRegression_face_recognition(train_basedir, train_labels_filename, train_images_dir,
#                                            test_basedir, test_labels_filename, test_images_dir)

# Method 3
# CNN with full image: Accuracy rate = 95.10%
classes = ('female', 'male')
img_root_train = train_images_dir
img_root_test = test_images_dir
train_csv = os.path.join(train_basedir, train_labels_filename)
test_csv = os.path.join(test_basedir, test_labels_filename)

train_batch_size = 64
test_batch_size = 8

transform = transforms.Compose([transforms.Resize((128, 128)),
                               transforms.CenterCrop(128),
                               transforms.RandomHorizontalFlip(),
                               transforms.ToTensor(),
                               transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
                               ])

train_dataset = Dataset(img_dir=img_root_train, train_csv=train_csv, transform=transform)
train_dataloader = Data.DataLoader(train_dataset, batch_size=train_batch_size, shuffle=True)
test_dataset = Dataset(img_dir=img_root_test, train_csv=test_csv, transform=transform)
test_dataloader = Data.DataLoader(test_dataset, batch_size=test_batch_size, shuffle=True)

# train(train_dataloader)
# test_image_examples(test_dataloader, classes, test_batch_size)
test_accuracy_rate(test_dataloader, test_batch_size)
