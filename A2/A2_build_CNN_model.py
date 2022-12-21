from keras import models
from keras import layers

def CNN_model():
    model = models.Sequential()
    model.add(layers.Conv2D(16, (5, 5), activation='relu', input_shape=(64, 64, 1)))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(8, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(16, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Flatten(name="intermediate_output"))
    model.add(layers.Dense(120, activation='relu'))
    model.add(layers.Dense(84, activation='relu'))
    # model.add(layers.Dense(2))
    model.add(layers.Dense(2, activation='softmax'))

    # The extra metric is important for the evaluate function
    model.compile(optimizer='rmsprop',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model

