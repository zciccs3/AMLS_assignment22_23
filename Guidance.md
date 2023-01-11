# Guidance

This file provides the main guidance for running this project successfully. Please copy the file [shape_predictor_68_face_landmarks.dat](https://www.kaggle.com/datasets/sergiovirahonda/shape-predictor-68-face-landmarksdat?resource=download) inside the [Good models and results](https://github.com/zciccs3/-zciccs3-AMLS_assignment22_23/tree/main/Good%20models%20and%20results) folder for the extraction of facial landmarks in some proposed models. 

## Setups

#### Python IDE installation
Install Integrated Development Environment (IDE) for Python.
#### Package installation
Install all packages required to run this project: torch, numpy, pandas, os, matplotlib, face_recognition, Keras, dlib, seaborn and PIL. Specifically, it is necessary to install dlib and cmake before installing the face_recognition package. The detailed introduction and instructions for installing face_recognition step by step can be found [here](https://github.com/ageitgey/face_recognition).
#### Code download
Download the programming repository [here](https://github.com/zciccs3/-zciccs3-AMLS_assignment22_23) in Github.

## Datasets preparation

#### Datasets download
This dataset is used in task A for binary classification tasks including gender detection and smile detection. It contains 5000 images for training and 1000 images for testing with faces of celebrities in real-world contexts. Each image is with the resolution of 178×218 in jpg format and annotated with attributes of 2 classes of gender (female and male) and 2 classes of emotion (smile or not) as labels. <br/>
Download [Celeba training and testing datasets](https://bit.ly/dataset_AMLS_22-23). <br/><br/>
Cartoon dataset is used in task B for multiclass classification tasks including face shape recognition and eye colour recognition. It contains 10000 images for training and 2500 images for testing with cartoon faces in blank context. Each image is with the resolution of 500×500 in png format and annotated with attributes of 5 classes of face shape and 5 classes of eye colour labels. <br/>
Download [Cartoon_set training and testing datasets](https://google.github.io/cartoonset/).

#### Images re-allocation in folders
The [Datasets](https://github.com/zciccs3/-zciccs3-AMLS_assignment22_23/tree/main/Datasets) folder in this project repository is empty. After downloading these two datasets from the resource, please arrange the training and testing images in two datasets following the structure shown below. Training and testing images within celeba dataset should be copied to celeba folder and celeba_test folder, respectively. Similarly, training and testing images within cartoon_set dataset should be copied to cartoon_set folder and cartoon_set_test folder, respectively. <br/>
![Image](https://github.com/zciccs3/-zciccs3-AMLS_assignment22_23/blob/main/Figures/Dataset%20images%20allocation.jpg)

## Main execution

**To execute the programme, please run the [Main.py](https://github.com/zciccs3/-zciccs3-AMLS_assignment22_23/blob/main/Main.py) in the repository directly, which will execute all methods in each task in the order as mentioned below.**
After the activation of training and testing sources and the definition of classes and batches in CNN training, the execution of four classification tasks starts.

#### Task A1 - gender detection
Three models are implemented to solve gender detection problem, including CNN with 64 * 64 and 32 * 32 resized images, respectively, and the logistic regression model. The comparison of their performances and merits and drawbacks analysis can be found in the paper. <br/>
```
# Method 1 - CNN with full image
train_dataloader_A1, test_dataloader_A1 = A1_CNN_dataloader()
# Train CNN model
A1.A1_CNN.train(train_dataloader_A1)
# Show examples of processed images and classification results
A1.A1_CNN.test_image_examples(test_dataloader_A1, A1_classes, test_batch_size_A1)
# Test the classification accuracy
A1.A1_CNN.test_accuracy_rate(test_dataloader_A1, test_batch_size_A1)
```
With the usage of CNN model, both the training and testing dataloaders are activated to load batches of paires of images and labels in training and testing stages, respectively. Then, the CNN model is trained and stored in the "Good models and results" folder. Examples of processed images and classification results, as well as the classification accuracy are reported in the following lines. <br/>
```
# Method 2 - Logistic Regression with full image
img_size = 128
Accuracy, _ = A1_logisticRegression_full_image(...)
```
As one of the most commonly used binary classification algorithm, logistic regression is used as well but difficult to be convergent with low classification accuracy.

#### Task A2 - smile detection

Four methods by changing the ategory of classifiers and image pre-processing methods are implemented. Both CNN and hybrid CNN-SVM models are used with face detection and mouth localisation for each of them, respectively. <br/>
```
# method 1 - CNN with face recognition
train_dataloader1, test_dataloader1 = A2_CNN_face_recognition_dataloader()
# Train CNN model
A2.A2_CNN_face_recognition.train(train_dataloader1)
# Show examples of processed images and classification results
A2.A2_CNN_face_recognition.test_image_examples(test_dataloader1, A2_classes, test_batch_size_A2)
# Test the classification accuracy
A2.A2_CNN_face_recognition.test_accuracy_rate(test_dataloader1, test_batch_size_A2)
```
CNN model with the input of images whose face regions are detected and cropped is used as the first model, containing training and testing.
```
# method 2 - CNN with mouth localisation
train_dataloader2, test_dataloader2 = A2_CNN_mouth_localisation_dataloader()
# Train CNN model
A2.A2_CNN_mouth.train(train_dataloader2)
# Show examples of processed images and classification results
A2.A2_CNN_mouth.test_image_examples(test_dataloader2, A2_classes, test_batch_size_A2)
# Test the classification accuracy
A2.A2_CNN_mouth.test_accuracy_rate(test_dataloader2, test_batch_size_A2)
```
With the same model structure but only changing the input image by localising the mouth region is implemented.
```
# method 3 - hybrid CNN-SVM with full image
CNN_SVM_full_image_train_and_test()
# method 4 - hybrid CNN-SVM with mouth localisation
CNN_SVM_mouth_localisation_train_and_test()
```
Then, two hybrid CNN-SVM models with the input of full image and the mouth region as features are used. Code lines above show the training and testing of these models.

#### Task B1 - face shape recognition

```
# Method 1 - CNN
train_dataloader_B1, test_dataloader_B1 = B1_CNN_dataloader()
# Train the CNN model
B1.B1_CNN.train(train_dataloader_B1)
# Show examples of processed images and classification results
B1.B1_CNN.test_image_examples(test_dataloader_B1, B1_classes, test_batch_size_B1)
# Test the classification accuracy
B1.B1_CNN.test_accuracy_rate(test_dataloader_B1, test_batch_size_B1)
```
Similarly, the CNN model is used from the activation of both training and testing dataloader to the training and testing of the model.
```
# Method 2 - Random Forest_imageArray method
Accuracy_RF_imageArray, Confusion_matrix = RF_imageArray_train_and_test()
# Method 3 - Random Forest_landmarks method
Prediction = RF_landmarks_train_and_test()
```
Then, two Random Forest models are utilised. The only difference between them is the feature extraction method, where image resize, grayscale and array conversion is used for the first model, and the extraction of 68 facial landmarks is used for the second model.

#### Task B2 - eye colour recognition

```
# Method 1 - RF with double eyes image
Accuracy_double_eyes = B2_random_forest_double_eyes_train_and_test(...)
# Method 2 - RF with single eye image
Accuracy_single_eye = B2_random_forest_single_eye_train_and_test(...)
# Method 3 - RF with single eye image glass removal
Accuracy_glass_removal = B2_random_forest_single_eye_glass_removal_train_and_test(...)
```
In eye-colour recognition, Random Forest models are used again. The only difference between these three methods is the image pre-processing procedure, where double-eye localisation, single-eye localisation, and single-eye localisation with the removal of images whose face is covered by black sunglasses are used, respectively. The second method aims to explore the effect of facial interferences like the existence of glasses frames in different colours, while the target of the third method is to avoid the confusion in training process due to the black sunglasses in some images. 
```
# Method 4 - CNN with double eyes image without glass removal
train_dataloader_B2, test_dataloader_B2 = B2_CNN_dataloader()
# Train CNN model
B2.B2_Eyes_color_recognition_CNN.train(train_dataloader_B2)
# Show examples of processed images and classification results
B2.B2_Eyes_color_recognition_CNN.test_image_examples(test_dataloader_B2, B2_classes, test_batch_size_B2)
# Test the classification accuracy
B2.B2_Eyes_color_recognition_CNN.test_accuracy_rate(test_dataloader_B2, test_batch_size_B2)
```
In addition, CNN model is used as well to compare its performance with Random Forest-based models. 

## Notes

1. Due to the fact that many models are tried on each classification task, it is recommended that models are better to be run each by each via adding comments on other models except for the evaluated one to avoid the confusion of results. 
2. Please copy the file "shape_predictor_68_face_landmarks.dat" inside the Good models and results folder for the extraction of facial landmarks in some proposed models.
3. The well-trained models for CNN-based methods in all tasks are stored in the Good models and results folder. To re-train these models, new well-trained models will substitute the previous model in pkl format. If you only prefer to test the model to show classification results and accuracy, please add comment on the "train" codes and only run test_image_examples and test_accuracy_rate functions so that the previously well-trained models will be called to predict the result and show model performances. Backups of CNN-based well-trained models for each task are stored in the folder of each task. 

