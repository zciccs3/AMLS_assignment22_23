## Brief summary of AMLS project
|          | Task A1: gender detection    | Task A2: smile detection     | Task B1: face shape recognition | Task B2: eye colour recognition  
|:------------:| :-----------: | :-----------: | :-----------: | :-----------: |
| Dataset    |   celebA    |   celebA   |   Cartoon set   |   Cartoon set    |
| Number of images | 5000 for training <br/> 1000 for testing  | 5000 for training <br/> 1000 for testing  | 10000 for training <br/> 2500 for testing  |   10000 for training <br/> 2500 for testing  |
| Image size | 178×218  | 178×218 | 500×500 | 500×500 |
| Models | Logistic regression <br/> CNN | CNN <br/> Hybrid CNN-SVM | CNN <br/> Random Forest | CNN <br/> Random Forest |
| Image pre-processing methods for CNN | Resize to 64×64/32×32; Random horizontal flip; Normalisation | Grayscale conversion; mouth localisation/face detection; Resize to 64×64; Normalisation | Grayscale conversion; Image resize to 64×64; Image crop to 48×48; Random horizontal flip; Normalisation | Image resize 64×64; single-eye localisation; Normalisation |
| Image pre-processing methods for other models | Image resize; Convert to arrays | Same as CNN models | Image resize; Grayscale conversion; Convert to arrays | Image resize; Eye localisation; Compare the array mean with threshold 60 (judge wearing sunglasses or not); Convert to arrays |
