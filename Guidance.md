# Guidance

This file provides the main guidance for running this project successfully.

## Setups

#### Python IDE installation
Install Integrated Development Environment (IDE) for Python.
#### Package installation
Install all packages required to run this project: torch, numpy, pandas, os, matplotlib, face_recognition, Keras, dlib, seaborn and PIL. Specifically, it is necessary to install dlib and cmake before installing the face_recognition package. The detailed introduction and instructions for installing face_recognition step by stwep can be found [here](https://github.com/ageitgey/face_recognition).
#### Code download
Download the programming directory [here](https://github.com/zciccs3/-zciccs3-AMLS_assignment22_23) in Github.

## Datasets preparation

#### Datasets download
This dataset is used in task A for binary classification tasks including gender detection and smile detection. It contains 5000 images for training and 1000 images for testing with faces of celebrities in real-world contexts. Each image is with the resolution of 178×218 in jpg format and annotated with attributes of 2 classes of gender (female and male) and 2 classes of emotion (smile or not) as labels. <br/>
Download [Celeba training and testing datasets](https://bit.ly/dataset_AMLS_22-23). <br/><br/>
Cartoon dataset is used in task B for multiclass classification tasks including face shape recognition and eye colour recognition. It contains 10000 images for training and 2500 images for testing with cartoon faces in blank context. Each image is with the resolution of 500×500 in png format and annotated with attributes of 5 classes of face shape and 5 classes of eye colour labels. <br/>
Download [Cartoon_set training and testing datasets](https://google.github.io/cartoonset/).

#### Images re-allocation in folders
The Datasets folder in this project repository is empty. After downloading these two datasets from the resource, please arrange the training and testing images in two datasets following the structure shown below. 
![Image]



## Main execution

## Notes


