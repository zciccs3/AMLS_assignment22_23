# Convert image information into arrays + Random forest
# Accuracy rate = 99.96%

from B1_image_preprocessing import B1_image_preprocessing
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Train sources
train_basedir = 'D:/Nottingham/FYP_Chenhao Shi/AMLS_Assignment2022_23_SN22073551/dataset_AMLS_22-23/cartoon_set'
train_labels_filename = 'labels.csv'
train_images_dir = os.path.join(train_basedir, 'img')
# Test sources
test_basedir = 'D:/Nottingham/FYP_Chenhao Shi/AMLS_Assignment2022_23_SN22073551/dataset_AMLS_22-23_test/cartoon_set_test'
test_labels_filename = 'labels.csv'
test_images_dir = os.path.join(test_basedir, 'img')
# Images resize
img_size = 64

# Train images and labels pre-processing
X_train, y_train = B1_image_preprocessing(train_basedir, train_labels_filename, train_images_dir, img_size)
RF_model = RandomForestClassifier(n_estimators=50, n_jobs=-1)
RF_model.fit(X_train, y_train)

print("Finish training......")

# Test images and label pre-processing
X_test, y_test = B1_image_preprocessing(test_basedir, test_labels_filename, test_images_dir, img_size)
y_prediction = RF_model.predict(X_test)

Accuracy = accuracy_score(y_test, y_prediction)
print("Accuracy rate = ", Accuracy)
