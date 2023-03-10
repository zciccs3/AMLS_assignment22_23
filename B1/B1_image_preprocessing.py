import numpy as np
import pandas as pd
import os
import cv2


# Image pre-processing for face shape recognition Random Forest model
def B1_image_preprocessing(labels_dir, labels_filename, images_dir, img_size):
    dataframe = pd.read_csv(os.path.join(labels_dir, labels_filename), delimiter='\t')
    img_list = dataframe.values[:, 3]
    labels = dataframe.values[:, 2]  # Face shape labels
    figure_count = dataframe.values[:, 0]
    imgs = [os.path.join(images_dir, file) for file in img_list]

    # Determine whether the folder is empty
    if os.path.isdir(images_dir):
        global_imgs = []
        global_labels = []
        i = 0
        for img_path in imgs:
            # Gray images transformation
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            # Images resize
            img = cv2.resize(src=img, dsize=(img_size, img_size), interpolation=cv2.INTER_LANCZOS4)
            label = labels[figure_count[i]]
            global_imgs.append(img)
            global_labels.append(label)
            i += 1

        # Convert to image arrays and reshape its size
        global_imgs = np.array(global_imgs).reshape(len(global_imgs), -1)
        global_labels = np.array(global_labels)

    return global_imgs, global_labels