## Brief summary of AMLS project
|          | Task A1: gender detection    | Task A2: smile detection     | Task B1: face shape recognition | Task B2: eye colour recognition  
|:------------:| :-----------: | :-----------: | :-----------: | :-----------: |
| Dataset    |   celebA    |   celebA   |   Cartoon set   |   Cartoon set    |
| Number of images | 5000 for training <br/> 1000 for testing  | 5000 for training <br/> 1000 for testing  | 10000 for training <br/> 2500 for testing  |   10000 for training <br/> 2500 for testing  |
| Image size | 178×218  | 178×218 | 500×500 | 500×500 |
| Models | Logistic regression <br/> CNN | CNN <br/> Hybrid CNN-SVM | CNN <br/> Random Forest | CNN <br/> Random Forest |
| Image pre-processing methods for CNN | Resize to 64×64/32×32 <br/> Random horizontal flip <br/> Normalisation | Grayscale conversion <br/> mouth localisation/face detection <br/> Resize to 64×64 <br/> Normalisation | Grayscale conversion <br/> Image resize to 64×64 <br/> Image crop to 48×48 <br/> Random horizontal flip <br/> Normalisation | Image resize 64×64 <br/> single-eye localisation |
