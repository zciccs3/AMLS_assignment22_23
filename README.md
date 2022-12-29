## Brief summary of AMLS project
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

### Contributions of this project
1) To detect genders, both logistic regression and CNN model with two image sizes provided to the network are used. Among them, CNN model with 64×64 resized image generate better accuracy of 95.1%. 
2) In smile detection, both CNN model with moth localisation and face detection image pre-processing methods and hybrid CNN-SVM model are used, discovering that performance of CNN-SVM with mouth localisation performs better with accuracy of 89.6%. 
3) In face shape recognition, both CNN and random forest model are implemented, finding that accuracy of the latter model with image array as features is the optimum of 100%. 
4) In eye colour recognition, same models with different feature extraction methods are conducted. The accuracy when removing the image with sunglasses reaches 100% as well.

### Required libraries in Python
* *Torch*: 





`pip install dlib`

![Image](https://github.com/zciccs3/-zciccs3-AMLS_assignment22_23/blob/main/Figures/A1_CNN_processed_images_training.png)

[A1_CNN](https://github.com/zciccs3/-zciccs3-AMLS_assignment22_23/blob/main/A1/A1_CNN.py)
