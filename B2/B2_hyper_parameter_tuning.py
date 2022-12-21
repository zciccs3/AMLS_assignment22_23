from matplotlib import pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, GridSearchCV
from B2_image_preprocessing import B2_image_preprocessing_single_eye_glass_removal
import os
import numpy as np

# Train sources
train_basedir = 'D:/Nottingham/FYP_Chenhao Shi/AMLS/dataset_AMLS_22-23/cartoon_set'
train_labels_filename = 'labels.csv'
train_images_dir = os.path.join(train_basedir, 'img')
# Test sources
test_basedir = 'D:/Nottingham/FYP_Chenhao Shi/AMLS/dataset_AMLS_22-23_test/cartoon_set_test'
test_labels_filename = 'labels.csv'
test_images_dir = os.path.join(test_basedir, 'img')

X_train, y_train, index = B2_image_preprocessing_single_eye_glass_removal(
    train_basedir, train_labels_filename, train_images_dir)

# Estimator optimisation: estimator > 20
scorel = []
for i in range(0, 200, 10):
    rfc = RandomForestClassifier(n_estimators=i+1)
    score = cross_val_score(rfc, X_train, y_train, cv=10).mean()
    scorel.append(score)
print(max(scorel), (scorel.index(max(scorel))*10)+1)
plt.figure(figsize=[20, 5])
plt.plot(range(1, 201, 10), scorel)
plt.show()

# max_depth optimisation: max_depth = 11
param_grid = {'max_depth': np.arange(1, 100, 10)}
rfc = RandomForestClassifier(n_estimators=50)
GS = GridSearchCV(rfc, param_grid, cv=10)
GS.fit(X_train, y_train)
index_best = GS.best_params_
score_best = GS.best_score_
print("The best max_depth is ", index_best)
print("The largest score is ", score_best)

# max_features optimisation: max_features = 6
param_grid = {'max_features': np.arange(1, 30, 5)}
rfc = RandomForestClassifier(n_estimators=50)
GS = GridSearchCV(rfc, param_grid, cv=10)
GS.fit(X_train, y_train)
index_best = GS.best_params_
score_best = GS.best_score_
print("The best max_features is ", index_best)
print("The largest score is ", score_best)
