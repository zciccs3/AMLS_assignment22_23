from matplotlib import pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, GridSearchCV
from B2_image_preprocessing import B2_image_preprocessing_single_eye_glass_removal
import numpy as np
from os.path import dirname, abspath

# Root path
base_dir = dirname(dirname(abspath(__file__)))

# label source
labels_dir = base_dir + '/Label_files'
# Train sources
train_labels_filename = 'cartoon_train_labels.csv'
train_images_dir = base_dir + '/Datasets/cartoon_set'
# Test sources
test_labels_filename = 'cartoon_test_labels.csv'
test_images_dir = base_dir + '/Datasets/cartoon_set_test'

# Obtain the training image arrays and labels with single eye localisation and sunglasses removal
X_train, y_train, index = B2_image_preprocessing_single_eye_glass_removal(
    labels_dir, train_labels_filename, train_images_dir)

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
