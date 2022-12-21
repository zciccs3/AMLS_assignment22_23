from torchvision import transforms
from B1_CNN import Dataset, train, test_image_examples, test_accuracy_rate
import torch.utils.data as Data


classes = (0, 1, 2, 3, 4)

img_root_train = 'D:/Nottingham/FYP_Chenhao Shi/AMLS_Assignment2022_23_SN22073551/dataset_AMLS_22-23/cartoon_set/img'
img_root_test = 'D:/Nottingham/FYP_Chenhao Shi/AMLS_Assignment2022_23_SN22073551/dataset_AMLS_22-23_test/cartoon_set_test/img'
train_csv = 'D:/Nottingham/FYP_Chenhao Shi/AMLS_Assignment2022_23_SN22073551/dataset_AMLS_22-23/cartoon_set/labels.csv'
test_csv = 'D:/Nottingham/FYP_Chenhao Shi/AMLS_Assignment2022_23_SN22073551/dataset_AMLS_22-23_test/cartoon_set_test/labels.csv'

train_batch_size = 16
test_batch_size = 16

transform = transforms.Compose([transforms.Resize((64, 64)),
                               transforms.CenterCrop(48),
                               transforms.Grayscale(num_output_channels=1),
                               transforms.RandomHorizontalFlip(),
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
