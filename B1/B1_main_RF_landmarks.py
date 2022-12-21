# Accuracy rate = 73%
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import B1_lab2_landmarks as l2
import B1_lab2_landmarks_test as l2_test


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

# Feature extraction
X_train, Labels_train = l2.extract_features_labels()
X_train = X_train.reshape(X_train.shape[0], X_train.shape[1] * X_train.shape[2])

# create the model - Random Forests
clf = RandomForestClassifier(n_estimators=100)

# Training - fit the training data to the model
clf.fit(X_train, Labels_train)
clf_pred = clf.predict(X_train)

print(classification_report(Labels_train, clf_pred))
print(confusion_matrix(Labels_train, clf_pred))
print("True labels: ", Labels_train)
print("Predicted labels: ", clf_pred)

# Testing
X_test, Labels_test = l2_test.extract_features_labels()
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1] * X_test.shape[2])
# predict label of test images
prediction = clf.predict(X_test)

print("True labels: ", Labels_test)
print("Predicted labels: ", prediction)
print("Accuracy rate = ", accuracy_score(Labels_test, prediction))











