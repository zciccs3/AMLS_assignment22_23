# Brief summary of AMLS project
In this project, four classification tasks are solved, including two binary classification problems (gender detection and smile detection), and two multiclass classification problems (face shape recognition and eye colour recognition), based on various machine learning algorithms, comprising logistic regression, Convolutional Neural Network (CNN), hybrid CNN-Support Vector Machine (SVM) and random forest (RF). Different algorithms are used in each task to compare their performances, thereby discovering the most suitable approach.

Table 1. A brief summary of four tasks in terms of their contexts, models, configurations, etc.
|          | Task A1: gender detection    | Task A2: smile detection     | Task B1: face shape recognition | Task B2: eye colour recognition  
|:------------:| :-----------: | :-----------: | :-----------: | :-----------: |
| Dataset    |   celebA    |   celebA   |   Cartoon set   |   Cartoon set    |
| Number of images | 5000 for training <br/> 1000 for testing  | 5000 for training <br/> 1000 for testing  | 10000 for training <br/> 2500 for testing  |   10000 for training <br/> 2500 for testing  |
| Image size | 178×218  | 178×218 | 500×500 | 500×500 |
| Number of classes | 2  | 2 | 5 | 5 |
| Models | Logistic regression <br/> CNN | CNN <br/> Hybrid CNN-SVM | CNN <br/> Random Forest| CNN <br/> Random Forest |
| Image pre-processing methods for CNN | Resize to 64×64/32×32; Random horizontal flip; Normalisation | Grayscale conversion; mouth localisation/face detection; Resize to 64×64; Normalisation | Grayscale conversion; Image resize to 64×64; Image crop to 48×48; Random horizontal flip; Normalisation | Image resize 64×64; single-eye localisation; Normalisation |
| Image pre-processing methods for other models | Image resize; Convert to arrays | Same as CNN models | Image resize; Grayscale conversion; Convert to arrays | Image resize; Eye localisation; Compare the array mean with threshold 60 (judge wearing sunglasses or not); Convert to arrays |
| Best accuracy and its model | 95.1% (CNN)  | 89.6% (CNN-SVM) | 100% (RF) | 100% (RF) |

## Contributions of this project
1) To detect genders, both logistic regression and CNN model with two image sizes provided to the network are used. Among them, CNN model with 64×64 resized image generate better accuracy of 95.1%. 
2) In smile detection, both CNN model with moth localisation and face detection image pre-processing methods and hybrid CNN-SVM model are used, discovering that performance of CNN-SVM with mouth localisation performs better with accuracy of 89.6%. 
3) In face shape recognition, both CNN and random forest model are implemented, finding that accuracy of the latter model with image array as features is the optimum of 100%. 
4) In eye colour recognition, same models with different feature extraction methods are conducted. The accuracy when removing the image with sunglasses reaches 100% as well.

## Required packages in Python
* **torch**: a machine learning library including scientific computing framework, providing algorithms for deep learning. Packages mainly include nn for neural networks (NNs) and au-tograd, optim, utils for model training and data acquisition.
* **numpy**: support for multi-dimensional matrices and arrays, and the collection of mathematical functions to operate them.
* **pandas**: a manipulation tool for powerful data analysis.
* **os**: manipulate operating systems and PC file systems.
* **matplotlib**: functions for visualisation.
* **face_recognition**: methods for dealing with faces in images. Build on the basis of dlib library.
* **Keras**: interface for artificial NNs and Tensorflow li-brary.
* **dlib**: a cross-platform software library where face detector and landmarks detection are used in this project.
* **sklearn**: support supervised and unsupervised learning and provide classification, regression, clustering, etc. algorithms.
* **seaborn**: this library is used for data visualisation constructed based on matplotlib library, providing an interface for plotting informative statistical figures.
* **cv2**: an open-source library containing valuable computer vision algorithms.
* **PIL**: a library possesses strong image processing capabilities.

## Key modules and their roles
**main.py** can be seen as the start of this project, mainly responsible for calling functions and classes in each task to realise data loading, image pre-processing, model construction, model training and testing, classification results evaluations, etc. Codes run following the logic in this document. 

### A1 - gender detection

**A1_image_preprocessing.py** is used to process images before fitting the model. Function Original_image_feature_extraction is used to extract features of full images and convert to arrays for logistic regression. <br/>
**A1_train_and_test.py** is used to train and test the logistic regression model with processed features and produce accuracy, classification report and confusion matrix to evaluate its performance. <br/>
**A1_CNN module.py** builds the CNN architecture and dataloader, trains and tests the model and collects the accuracy finally. <br/>

### A2 - smile detection

**A2_CNN_face_recognition.py** aims to load the data with face detection, build the CNN model, train and test the model, and finally test the accuracy and classification results.<br/>
**A2_CNN_mouth.py** has the same purpose as last module but only change from face detection to mouth localisation. <br/>
**A2_SVM_model_wrapper.py** is responsible for hybrid CNN-SVM model, connecting the fully connected layer of CNN to SVM, together with the fitting of CNN and SVM as well as the evaluation and prediction of the hybrid-model. <br/>
**A2_build_CNN_model.py** builds up the architecture of CNN from layer to layer and compile the model. <br/>
**A2_image_preprocessing.py** is used to pre-process the image, including methods of mouth localisation and full image resizing. <br/>

### B1 - face shape recognition

**B1_CNN.py** loads image data, build, train and test the CNN model, and finally tests the accuracy and classification results. <br/>
**B1_image_preprocessing.py** realises image pre-processing, including grayscale conversion, image resize, array format conversion, etc. used in random forest model. <br/>
**B1_lab2_landmarks.py** detects faces in images as well as produces facial landmarks to operating as features in random forest model. <br/>

### B2 - eye color recognition

**B2_Eyes_color_recognition_CNN.py** loads images, trains and tests CNN model, and finally tests the accuracy and classification results. <br/>
**B2_image_preprocessing.py** defines three functions corresponding to three image processing methods, including double-eye localisation, single-eye localisation and single-eye localisation with the removal of sunflasses attached images. <br/>
**B2_train_and_test.py** trains and tests random forest models with three image processing methods mentioned in the last module. <br/>
**B2_hyper_parameter_tuning.py** is responsible in tuning hyper-parameters to fit the random forest model based on grid search method. <br/>

## Software

![Image](https://github.com/zciccs3/-zciccs3-AMLS_assignment22_23/blob/main/Figures/Pycharm%20logo.png)

`pip install dlib`

[A1_CNN](https://github.com/zciccs3/-zciccs3-AMLS_assignment22_23/blob/main/A1/A1_CNN.py)
