import os
from tensorflow.python.keras.utils.np_utils import to_categorical
from A2_SVM_model_wrapper import FCtoSVM
from A2_build_CNN_model import CNN_model
from A2_image_preprocessing import A2_CNN_SVM_load_images
import matplotlib.pyplot as plt


train_basedir = 'D:/Nottingham/FYP_Chenhao Shi/AMLS_Assignment2022_23_SN22073551/dataset_AMLS_22-23/celeba'
train_labels_filename = 'labels.csv'
train_images_dir = os.path.join(train_basedir, 'img')

test_basedir = 'D:/Nottingham/FYP_Chenhao Shi/AMLS_Assignment2022_23_SN22073551/dataset_AMLS_22-23_test/celeba_test'
test_labels_filename = 'labels.csv'
test_images_dir = os.path.join(test_basedir, 'img')


X_train, y_train = A2_CNN_SVM_load_images(train_basedir, train_labels_filename, train_images_dir)
X_test, y_test = A2_CNN_SVM_load_images(test_basedir, test_labels_filename, test_images_dir)
wrapper = FCtoSVM(CNN_model())
wrapper.model.summary()
train_images = X_train
train_labels = y_train
test_images = X_test
test_labels = y_test

print("Shape of train_images is ", X_train.shape)
print("Shape of train_labels is ", y_train.shape)

wrapper = FCtoSVM(CNN_model())

epochs = 15
performance = {
    "CNN + SVM": [],
    "CNN_softmax": []
}

for i in range(epochs):
    print('Starting epoch: {}'.format(i + 1))
    wrapper.fit(train_images, train_labels, epochs=1, batch_size=16)
    performance["CNN + SVM"].append(wrapper.evaluate(test_images, test_labels))
    print("CNN+SVM: ", wrapper.evaluate(test_images, test_labels))
    performance["CNN_softmax"].append(wrapper.model.evaluate(test_images, to_categorical(test_labels))[1])
    print("CNN_softmax: ", wrapper.model.evaluate(test_images, to_categorical(test_labels))[1])

x = range(epochs)
y1 = performance["CNN + SVM"]
y2 = performance["CNN_softmax"]
# plt.plot(x, y1, color='orangered', marker='o', linestyle='-', label='CNN + SVM')
# plt.plot(x, y2, color='blueviolet', marker='o', linestyle='-', label='CNN_softmax')
plt.plot(x, y1, color='red', linestyle='-', label='CNN + SVM')
plt.plot(x, y2, color='blue', linestyle='-', label='CNN_softmax')
plt.legend()
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.show()

