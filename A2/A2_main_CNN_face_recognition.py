import torch.utils.data as Data
from torchvision import transforms
from A2_CNN_face_recognition import Dataset, train, test_image_examples, test_accuracy_rate

classes = ('Not', 'smiling')

img_root_train = 'D:/Nottingham/FYP_Chenhao Shi/AMLS_Assignment2022_23_SN22073551/dataset_AMLS_22-23/celeba/img'
img_root_test = 'D:/Nottingham/FYP_Chenhao Shi/AMLS_Assignment2022_23_SN22073551/dataset_AMLS_22-23_test/celeba_test/img'
train_csv = 'D:/Nottingham/FYP_Chenhao Shi/AMLS_Assignment2022_23_SN22073551/dataset_AMLS_22-23/celeba/labels.csv'
test_csv = 'D:/Nottingham/FYP_Chenhao Shi/AMLS_Assignment2022_23_SN22073551/dataset_AMLS_22-23_test/celeba_test/labels.csv'

train_batch_size = 16
test_batch_size = 16

transform = transforms.Compose([transforms.Grayscale(num_output_channels=1),
                                transforms.Resize((64, 64)),
                                transforms.ToTensor(),
                                transforms.Normalize(mean=0.5, std=0.5)
                               ])

train_dataset = Dataset(img_dir=img_root_train, train_csv=train_csv, transform=transform)
train_dataloader = Data.DataLoader(train_dataset, batch_size=train_batch_size, shuffle=True)
test_dataset = Dataset(img_dir=img_root_test, train_csv=test_csv, transform=transform)
test_dataloader = Data.DataLoader(test_dataset, batch_size=test_batch_size, shuffle=True)

# train(train_dataloader)
test_image_examples(test_dataloader, classes, test_batch_size)
# test_accuracy_rate(test_dataloader, test_batch_size)
