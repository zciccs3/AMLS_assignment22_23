from B2_image_preprocessing import B2_image_preprocessing_double_eyes, \
    B2_image_preprocessing_single_eye, B2_image_preprocessing_single_eye_glass_removal
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score


def B2_random_forest_double_eyes_train_and_test(train_basedir, train_labels_filename, train_images_dir,
                                                test_basedir, test_labels_filename, test_images_dir):

    # Train images and labels pre-processing
    X_train, y_train = B2_image_preprocessing_double_eyes(train_basedir, train_labels_filename, train_images_dir)
    RF_model = RandomForestClassifier(n_estimators=50, n_jobs=-1)
    RF_model.fit(X_train, y_train)
    print("Finish training of B2 random forest double eyes method......")

    # Test images and label pre-processing
    X_test, y_test = B2_image_preprocessing_double_eyes(test_basedir, test_labels_filename, test_images_dir)
    y_prediction = RF_model.predict(X_test)
    print("Finish testing of B2 random forest double eyes method......")

    Accuracy = accuracy_score(y_test, y_prediction)

    return Accuracy


def B2_random_forest_single_eye_train_and_test(train_basedir, train_labels_filename, train_images_dir,
                                                test_basedir, test_labels_filename, test_images_dir):

    # Train images and labels pre-processing
    X_train, y_train = B2_image_preprocessing_single_eye(train_basedir, train_labels_filename, train_images_dir)
    RF_model = RandomForestClassifier(n_estimators=50, n_jobs=-1)
    RF_model.fit(X_train, y_train)
    print("Finish training of B2 random forest double eyes method......")

    # Test images and label pre-processing
    X_test, y_test = B2_image_preprocessing_single_eye(test_basedir, test_labels_filename, test_images_dir)
    y_prediction = RF_model.predict(X_test)
    print("Finish testing of B2 random forest double eyes method......")

    Accuracy = accuracy_score(y_test, y_prediction)

    return Accuracy


def B2_random_forest_single_eye_glass_removal_train_and_test(train_basedir, train_labels_filename, train_images_dir,
                                                test_basedir, test_labels_filename, test_images_dir):

    # Train images and labels pre-processing
    X_train, y_train, index = B2_image_preprocessing_single_eye_glass_removal(train_basedir, train_labels_filename, train_images_dir)
    RF_model = RandomForestClassifier(n_estimators=50, n_jobs=-1)
    RF_model.fit(X_train, y_train)
    print("Finish training of B2 random forest double eyes method......")

    # Test images and label pre-processing
    X_test, y_test, index = B2_image_preprocessing_single_eye_glass_removal(test_basedir, test_labels_filename, test_images_dir)
    y_prediction = RF_model.predict(X_test)
    print("Finish testing of B2 random forest double eyes method......")

    Accuracy = accuracy_score(y_test, y_prediction)

    return Accuracy