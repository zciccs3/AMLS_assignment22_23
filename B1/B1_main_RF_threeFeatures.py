from matplotlib import pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import mahotas
import cv2
import os
import h5py
import pandas as pd
from sklearn.ensemble import RandomForestClassifier


fixed_size = tuple((500, 500))

img_root_train = 'D:/Nottingham/FYP_Chenhao Shi/AMLS_Assignment2022_23_SN22073551/dataset_AMLS_22-23/cartoon_set/img'
img_root_test = 'D:/Nottingham/FYP_Chenhao Shi/AMLS_Assignment2022_23_SN22073551/dataset_AMLS_22-23_test/cartoon_set_test/img'
train_csv = 'D:/Nottingham/FYP_Chenhao Shi/AMLS_Assignment2022_23_SN22073551/dataset_AMLS_22-23/cartoon_set/labels.csv'
test_csv = 'D:/Nottingham/FYP_Chenhao Shi/AMLS_Assignment2022_23_SN22073551/dataset_AMLS_22-23_test/cartoon_set_test/labels.csv'

# number of trees for Random Forests
num_tree = 100
# bins for histograms
bins = 8
# seed for reproducing the same result
seed = 9


# features description 1 Hu Moments
def fd_hu_moments(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    feature = cv2.HuMoments(cv2.moments(image)).flatten()
    return feature


# feature-descriptor 2 Haralick Texture
def fd_haralick(image):
    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Compute the haralick texture feature vector
    haralic = mahotas.features.haralick(gray).mean(axis=0)
    return haralic


# feature-description 3 Color Histogram
def fd_histogram(image, mask=None):
    # Convert the image to HSV colors-space
    image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    # Compute the color histogram
    hist = cv2.calcHist([image], [0, 1, 2], None, [bins, bins, bins], [0, 256, 0, 256, 0, 256])
    # normalize the histogram
    cv2.normalize(hist, hist)
    # return the histogram
    return hist.flatten()


dataframe = pd.read_csv(train_csv, delimiter='\t')
train_list = dataframe.values[:, 3]
train_labels = dataframe.values[:, 2]  # Face shape labels
train_figure_count = dataframe.values[:, 0]
imgs = [os.path.join(img_root_train, file) for file in train_list]

labels = []
global_features = []
i = 0
for index in imgs:
    image = cv2.imread(index)
    if image is not None:
        image = cv2.resize(image, fixed_size)
        fv_hu_moments = fd_hu_moments(image)
        fv_haralick = fd_haralick(image)
        fv_histogram = fd_histogram(image)

        global_feature = np.hstack([fv_histogram, fv_haralick, fv_hu_moments])
        global_features.append(global_feature)
        labels.append(train_labels[train_figure_count[i]])
    i += 1

print("Global feature extraction ia completed......")
# get the overall feature vector size
print("Feature vector size {}".format(np.array(global_features).shape))
# get the overall training label size
print("Training Labels {}".format(np.array(labels).shape))

scaler = MinMaxScaler(feature_range=(0, 1))
rescaled_features = scaler.fit_transform(global_features)
print("Feature vector normalised......")

print("Target labels: {}".format(labels))
print("Target labels shape: {}".format(np.array(labels).shape))

# save the feature vector using HDF5
h5f_train_data = h5py.File('Train_data.h5', 'w')
h5f_train_data.create_dataset('dataset_1', data=np.array(rescaled_features))

h5f_train_label = h5py.File('Train_labels.h5', 'w')
h5f_train_label.create_dataset('dataset_1', data=np.array(labels))

h5f_train_data.close()
h5f_train_label.close()

print("End of training......")

# Import feature vectors and labels
h5f_train_data = h5py.File('Train_data.h5', 'r')
h5f_train_label = h5py.File('Train_labels.h5', 'r')
global_features_string = h5f_train_data['dataset_1']
global_labels_string = h5f_train_label['dataset_1']
global_train_features = np.array(global_features_string)
global_train_labels = np.array(global_labels_string)

# create the model - Random Forests
clf = RandomForestClassifier(n_estimators=100)
# fit the training data to the model
clf.fit(global_train_features, global_train_labels)
clf_pred = clf.predict(global_train_features)

print(classification_report(global_train_labels, clf_pred))
print(confusion_matrix(global_train_labels, clf_pred))


# Test
dataframe_test = pd.read_csv(test_csv, delimiter='\t')
test_list = dataframe_test.values[:, 3]
test_labels = dataframe_test.values[:, 2]  # Face shape labels
test_figure_count = dataframe.values[:, 0]
test_imgs = [os.path.join(img_root_test, file) for file in test_list]

test_global_features = []
j = 0
counter = 0
for index in test_imgs:
    test_image = cv2.imread(index)
    if test_image is not None:
        test_image = cv2.resize(test_image, fixed_size)
        fv_hu_moments_test = fd_hu_moments(test_image)
        fv_haralick_test = fd_haralick(test_image)
        fv_histogram_test = fd_histogram(test_image)

        test_feature = np.hstack([fv_histogram_test, fv_haralick_test, fv_hu_moments_test])
        test_global_features.append(test_feature)

        # show predicted label on image
        # cv2.putText(test_image, prediction, (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 3)
        # # display the output image
        # plt.imshow(cv2.cvtColor(test_image, cv2.COLOR_BGR2RGB))
        # plt.show()

    j += 1

test_scaler = MinMaxScaler(feature_range=(0, 1))
test_rescaled_features = test_scaler.fit_transform(test_global_features)

# save the test feature vector using HDF5
h5f_test_data = h5py.File('Test_data.h5', 'w')
h5f_test_data.create_dataset('dataset_2', data=np.array(test_rescaled_features))

h5f_test_data.close()

print("End of testing......")

# Import feature vectors and labels
h5f_test_data = h5py.File('Test_data.h5', 'r')
global_test_features_string = h5f_test_data['dataset_2']
global_test_features = np.array(global_test_features_string)

# predict label of test image
# prediction = clf.predict(test_global_feature.reshape(1, -1))[0]
# if prediction == test_labels[j]:
#     counter += 1

prediction = clf.predict(global_test_features)

accuracy = accuracy_score(prediction, test_labels)
print("Accuracy = ", accuracy)


