from keras.models import Model
from numpy import ravel
from sklearn.svm import SVC
from tensorflow.keras.utils import to_categorical
from os.path import dirname, abspath

# Root path
base_dir = dirname(dirname(abspath(__file__)))


# Build the hybrid CNN-SVM model
class FCtoSVM:
    def __init__(self, model, svm=None):
        super().__init__()
        self.model = model
        self.intermediate_model = None
        self.svm = svm
        if svm is None:
            self.svm = SVC(kernel='linear')

    # Add layers in the model
    def add(self, layer):
        return self.model.add(layer)

    # Fit the CNN and then the hybrid model
    def fit(self, x=None, y=None, batch_size=None, epochs=1, verbose=1, callbacks=None, validation_split=0,
            validation_data=None, shuffle=True, class_weight=None, sample_weight=None, initial_epoch=0,
            steps_per_epoch=None, validation_steps=None, **kwargs):
        fit = self.model.fit(x, to_categorical(y), batch_size, epochs, verbose, callbacks, validation_split,
                             validation_data, shuffle, class_weight, sample_weight, initial_epoch, steps_per_epoch,
                             validation_steps, **kwargs)
        # Call the function to fit the hybrid CNN-SVM model
        self.fit_svm(x, y, self.__get_split_layer())
        return fit

    # Fit the hybrid CNN-SVM model
    def fit_svm(self, x, y, split_layer):
        # Store intermediate model
        self.intermediate_model = Model(inputs=self.model.input,
                                        outputs=split_layer.output)
        # Use output of intermediate model to train SVM
        intermediate_output = self.intermediate_model.predict(x)
        self.svm.fit(intermediate_output, y)

    # Evaluate the performance of model - Accuracy
    def evaluate(self, x=None, y=None, batch_size=None, verbose=1, steps=None):
        if self.intermediate_model is None:
            raise Exception("A model must be fit before running evaluate")
        output = self.predict(x, batch_size, verbose, steps)
        correct = [output[i] == y[i]
                   for i in range(len(output))]
        accuracy = sum(correct) / len(correct)
        return accuracy

    # Predict the output in testing stage
    def predict(self, x, batch_size=None, verbose=0, steps=None):

        intermediate_prediction = self.intermediate_model.predict(x, batch_size, verbose, steps)
        output = self.svm.predict(intermediate_prediction)
        return output

    # Alter the last fully connected layer of CNN to SVM as a classifier
    def __get_split_layer(self):
        if len(self.model.layers) < 3:
            raise ValueError('self.layers to small for a relevant split')
        for layer in self.model.layers:
            if layer.name == "split_layer":
                return layer
        # if no specific cut of point is specified we can assume we need to remove only the last (softmax) layer
        return self.model.layers[-3]
