# CNN with face recognition
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.optim as optim
import torchvision
import torch.utils.data as Data
from PIL import Image
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import face_recognition
from os.path import dirname, abspath

# Root path
base_dir = dirname(dirname(abspath(__file__)))


# Open images
def default_loader(path):
    try:
        img = Image.open(path)
        return img
    except:
        print('Cannot open {}'.format(path))

# Build the dataloader
class Dataset(Data.DataLoader):
    def __init__(self, img_dir, train_csv, transform, loader=default_loader):
        dataframe = pd.read_csv(train_csv, delimiter='\t')
        img_list = dataframe.values[:, 1]
        img_labels = dataframe.values[:, 3]  # Smile labels
        img_labels[img_labels == -1] = 0

        self.imgs = [os.path.join(img_dir, file) for file in img_list]
        self.labels = img_labels
        self.transform = transform
        self.loader = loader

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, index):
        img_path = self.imgs[index]
        label = torch.from_numpy(np.array(self.labels[index], dtype=np.int64))

        img = self.loader(img_path)
        # Detect the facial region in the full
        image = face_recognition.load_image_file(img_path)
        # Localise the coordination of faces
        face_location = face_recognition.face_locations(image)

        if np.array(face_location).size == 0:
            img2 = img
        else:
            # Define the corner coordinates of the face
            for top, right, bottom, left in face_location:
                img2 = img.crop((left, top, right, bottom))
        # Image transformation
        if self.transform is not None:
            try:
                img = self.transform(img2)
            except:
                print('Cannot transform the image: {}'.format(img_path))
        return img, label


# Define CNN network
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # Convolution layer 1
        self.conv1 = nn.Conv2d(1, 16, 5)
        # Pooling layer
        self.pool = nn.MaxPool2d(2, 2)
        # Convolution layer 2
        self.conv2 = nn.Conv2d(16, 8, 3)
        # Convolution layer 3
        self.conv3 = nn.Conv2d(8, 16, 3)
        # Fully connected layer
        self.fc1 = nn.Linear(16 * 6 * 6, 120)
        self.fc2 = nn.Linear(120, 84)
        # Two outputs
        self.fc3 = nn.Linear(84, 2)
        self.dropout = nn.Dropout(p=0.2)

    # Forward transmission
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        x = self.dropout(x)
        return x

# CNN training
def train(train_dataloader):
    epoch_loss = []
    # Neural network
    net = Net()
    # Optimiser (learning rate = 0.001)
    # optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
    optimizer = optim.Adam(net.parameters(), lr=0.001, weight_decay=1e-8)
    # Loss function
    criterion = nn.CrossEntropyLoss()

    # Training process
    for epoch in range(100):
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

    # Plot the learning curve - change of loss
    iteration = np.arange(100)
    plt.plot(iteration, epoch_loss, ls='-', lw=2, label='A2_CNN with face recognition')
    plt.legend()
    plt.xlabel('Episode')
    plt.ylabel('Loss of an episode')
    plt.title('Loss of face recognition CNN in A2')
    plt.show()
    # Save the well-trained model
    torch.save(net, base_dir + '/Good models and results/A2_face_recognition_CNN.pkl')

# Test the classification accuracy with testing database
def test_accuracy_rate(test_dataloader, test_batch_size):
    module = torch.load(base_dir + '/Good models and results/A2_face_recognition_CNN.pkl')
    module.eval()

    accurate_number = 0
    total_number = 0
    # Extract processed images and labels from the testing dataset
    for ii, (img, label) in enumerate(test_dataloader):
        img = Variable(img)
        label = Variable(label)
        output = module(img)
        # Predict the result based on the well-trained model
        _, predict = torch.max(output.data, 1)
        total_number += test_batch_size
        # Count the number of images that are wrongly classified
        for i in range(test_batch_size):
            if predict[i] == label[i]:
                accurate_number += 1

    print(total_number)
    print('Accuracy = %f' % (accurate_number / total_number))

# Show images in a table
def imshow(img):
    img = img / 2 + 0.5
    npimg = img.numpy()
    npimg = np.transpose(npimg, (1, 2, 0))
    # cv2.putText(npimg, "female", (20, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.1, (255, 255, 255), cv2.LINE_AA)
    # cv2.imshow("add_text", npimg)
    # cv2.waitKey()
    plt.imshow(npimg)
    plt.show()

# Show some examples of classification results in the testing dataset
def test_image_examples(test_dataloader, classes, test_batch_size):
    module = torch.load(base_dir + '/Good models and results/A2_face_recognition_CNN.pkl')
    dataiter = iter(test_dataloader)
    img, label = next(dataiter, 0)
    imshow(torchvision.utils.make_grid(img, nrow=4))
    print('True labels: ', " ".join('%5s' % classes[label[j]] for j in range(test_batch_size)))

    output = module(Variable(img))
    _, predict = torch.max(output.data, 1)
    print('Predicted labels: ', " ".join('%5s' % classes[predict[j]] for j in range(test_batch_size)))


