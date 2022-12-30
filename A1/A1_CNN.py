import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix, classification_report
from torch.autograd import Variable
import torch.optim as optim
import torchvision
import torch.utils.data as Data
from PIL import Image
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
from os.path import dirname, abspath

# Root path
base_dir = dirname(dirname(abspath(__file__)))


# Open the image
def default_loader(path):
    try:
        img = Image.open(path)
        return img.convert('RGB')
    except:
        print('Cannot open {}'.format(path))


# Dataloader: load images and transfer to the preferred format
class Dataset(Data.DataLoader):
    def __init__(self, img_dir, train_csv, transform, loader=default_loader):
        dataframe = pd.read_csv(train_csv, delimiter='\t')
        img_list = dataframe.values[:, 1]
        img_labels = dataframe.values[:, 2]  # Gender labels
        img_labels[img_labels == -1] = 0

        self.imgs = [os.path.join(img_dir, file) for file in img_list]
        self.labels = img_labels
        self.transform = transform
        self.loader = loader

    def __len__(self):
        return len(self.imgs)

    # Generate the transformed images and labels in tensor forms
    def __getitem__(self, index):
        img_path = self.imgs[index]
        label = torch.from_numpy(np.array(self.labels[index], dtype=np.int64))
        img = self.loader(img_path)
        if self.transform is not None:
            try:
                img = self.transform(img)
            except:
                print('Cannot transform the image: {}'.format(img_path))
        return img, label


# Define CNN network
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # Convolution layer
        self.conv1 = nn.Conv2d(3, 6, 5)
        # Pooling layer
        self.pool = nn.MaxPool2d(2, 2)
        # Convolution layer
        self.conv2 = nn.Conv2d(6, 16, 5)
        # Fully connected layer
        self.fc1 = nn.Linear(16 * 13 * 13, 120)  # 64*64 size
        # self.fc1 = nn.Linear(16 * 5 * 5, 120)  # 32*32 size
        # self.fc1 = nn.Linear(16 * 29 * 29, 120)  # 128*128 size
        self.fc2 = nn.Linear(120, 84)
        # Two outputs
        self.fc3 = nn.Linear(84, 2)
        self.dropout = nn.Dropout(p=0.2)

    # Forward transmission
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        # x = F.softmax(self.fc3(x))
        x = self.fc3(x)
        x = self.dropout(x)
        return x


# Train the CNN model
def train(train_dataloader):
    epoch_loss = []
    # Neural network
    net = Net()
    # Optimiser (learning rate = 0.001)
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
    # optimizer = optim.Adam(net.parameters(), lr=0.001, weight_decay=1e-8)
    # Loss function
    criterion = nn.CrossEntropyLoss()

    # Training in several episodes
    for epoch in range(60):
        running_loss = 0
        for i, (img, label) in enumerate(train_dataloader):
            img = Variable(img)
            label = Variable(label)
            # Gradient = 0
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(img)
            # Calculate the loss
            loss = criterion(outputs, label)
            # Back propagation
            loss.backward()
            # Update parameters
            optimizer.step()
            # Accumulate loss
            running_loss += loss.item()

        print('Episode %d - loss: %.3f' % (epoch + 1, running_loss))
        epoch_loss.append(running_loss)

    print('Finished Training')

    # Plot the learning curve - change of cross entropy loss
    iteration = np.arange(60)
    plt.plot(iteration, epoch_loss, ls='-', lw=2, label='A1_CNN')
    plt.legend()
    plt.xlabel('Episode')
    plt.ylabel('Loss in an episode')
    plt.title('Loss of CNN in A1')
    plt.show()
    # Save the well-trained CNN model
    torch.save(net, base_dir + '/Good models and results/A1_CNN_model.pkl')


# Test the classification accuracy
def test_accuracy_rate(test_dataloader, test_batch_size):
    # Load the well-trained CNN model
    module = torch.load(base_dir + '/Good models and results/A1_CNN_model.pkl')
    module.eval()
    accurate_number = 0
    total_number = 0
    Prediction = []
    True_labels = []
    # Load images and relevant labels from the testing database
    for ii, (img, label) in enumerate(test_dataloader):
        img = Variable(img)
        label = Variable(label)
        output = module(img)
        _, predict = torch.max(output.data, 1)

        Prediction.append(predict.tolist())
        True_labels.append(label.tolist())

        total_number += test_batch_size
        for i in range(test_batch_size):
            if predict[i] == label[i]:
                accurate_number += 1
            # else:
            #     imshow(img[i])
            #     print(predict[i], ii, i)
    print('Accuracy = %f' % (accurate_number / total_number))

    Prediction = np.array(Prediction).reshape(1, -1)[0]
    True_labels = np.array(True_labels).reshape(1, -1)[0]

    # Generate the confusion matrix
    Confusion_matrix = confusion_matrix(True_labels, Prediction)
    Confusion_matrix = Confusion_matrix / Confusion_matrix.astype(np.float).sum(axis=1)

    # Plot the confusion matrix
    sns.set()
    f, ax = plt.subplots()
    sns.heatmap(Confusion_matrix, annot=True, ax=ax)
    # ax.set_title('Confusion matrix')
    ax.set_xlabel('Predictions')
    ax.set_ylabel('True labels')
    plt.show()

    # print(classification_report(True_labels, Prediction))


# Function used to show processed images
def imshow(img):
    img = img / 2 + 0.5
    npimg = img.numpy()
    npimg = np.transpose(npimg, (1, 2, 0))
    plt.imshow(npimg)
    plt.show()


# Show examples of testing images with predicted and true labels
def test_image_examples(test_dataloader, classes, test_batch_size):
    module = torch.load(base_dir + '/Good models and results/A1_CNN_model.pkl')
    dataiter = iter(test_dataloader)
    img, label = next(dataiter, 0)
    imshow(torchvision.utils.make_grid(img, nrow=4))
    print('True labels: ', " ".join('%5s' % classes[label[j]] for j in range(test_batch_size)))

    output = module(Variable(img))
    _, predict = torch.max(output.data, 1)
    print('Predicted labels: ', " ".join('%5s' % classes[predict[j]] for j in range(test_batch_size)))



