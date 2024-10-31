#  Basado en: https://www.kaggle.com/code/shtrausslearning/pytorch-cnn-binary-image-classification#6-%7C-Creating-Dataloaders
import os
from PIL import Image
import torch.nn as nn
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader, random_split

# Basado en : https://towardsdatascience.com/pytorch-vision-binary-image-classification-d9a227705cf9
class AnimalClassifierS(nn.Module):
    def __init__(self, num_classes=2):
        super(AnimalClassifierS, self).__init__()
        self.layer1 = self.conv_layer(c_in=3, c_out=32, dropout=0.1, kernel_size=3, stride=1, padding=2)
        self.layer2 = self.conv_layer(c_in=32, c_out=16, dropout=0.1, kernel_size=3, stride=1, padding=1)
        self.layer3 = self.conv_layer(c_in=16, c_out=8, dropout=0.1, kernel_size=3, stride=1, padding=1)
        self.final_conv = nn.Conv2d(in_channels=8, out_channels=2, kernel_size=3, stride=1, padding=0)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.num_flatten = 54 * 54 * 2

        # Capas densas
        self.fc1 = nn.Linear(self.num_flatten, 64)
        self.fc2 = nn.Linear(64, num_classes)

    def forward(self, x):
        x = self.layer1(x)
        x = self.pool(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.pool(x)
        x = self.final_conv(x)

        # Capas Densas
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.fc2(x)
        return x

    def conv_layer(self, c_in, c_out, dropout, **kwargs):
        return nn.Sequential(
            nn.Conv2d(in_channels=c_in, out_channels=c_out, **kwargs),
            nn.BatchNorm2d(num_features=c_out),
            nn.ReLU(),
            nn.Dropout2d(p=dropout)
        )
    

class AnimalClassifierM(nn.Module):
    def __init__(self, num_classes=2):
        super(AnimalClassifierM, self).__init__()
        self.layer1 = self.conv_layer(c_in=3, c_out=64, dropout=0.1, kernel_size=3, stride=1, padding=2)
        self.layer2 = self.conv_layer(c_in=64, c_out=32, dropout=0.1, kernel_size=3, stride=1, padding=1)
        self.layer3 = self.conv_layer(c_in=32, c_out=16, dropout=0.1, kernel_size=3, stride=1, padding=1)
        self.layer4 = self.conv_layer(c_in=16, c_out=8, dropout=0.1, kernel_size=3, stride=1, padding=1)
        self.final_conv = nn.Conv2d(in_channels=8, out_channels=4, kernel_size=3, stride=1, padding=0)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Calculate flattened size
        self.num_flatten = 54 * 54 * 4

        # Capas densas
        self.fc1 = nn.Linear(self.num_flatten, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.layer1(x)
        x = self.pool(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.pool(x)
        x = self.layer4(x)
        x = self.final_conv(x)

        # Flatten
        x = x.view(x.size(0), -1)
        # Fully connected layers
        x = self.fc1(x)
        x = self.fc2(x)
        return x

    def conv_layer(self, c_in, c_out, dropout, **kwargs):
        return nn.Sequential(
            nn.Conv2d(in_channels=c_in, out_channels=c_out, **kwargs),
            nn.BatchNorm2d(num_features=c_out),
            nn.ReLU(),
            nn.Dropout2d(p=dropout)
        )
    
class pytorch_data(Dataset):
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.train_dir = os.path.join(data_dir, "train")
        self.test_dir = os.path.join(data_dir, "test")
        self.classes = os.listdir(self.train_dir)
        self.transforms = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

        self.image_paths = []
        self.labels = []

        # Cargar imágenes y labels de entrenamiento y prueba
        for class_name in self.classes:
            class_dir = os.path.join(self.train_dir, class_name)
            for filename in os.listdir(class_dir):
                self.image_paths.append(os.path.join(class_dir, filename))
                self.labels.append(self.classes.index(class_name))

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        image_path = self.image_paths[index]
        label = self.labels[index]
        image = Image.open(image_path)
        image = self.transforms(image)
        return image, label

    def create_dataloaders(self, train_size=0.8, batch_size=32):
        len_img = len(self)
        len_train = int(train_size * len_img)
        len_val = len_img - len_train

        # División del conjunto de datos en entrenamiento y validación
        train_ts, val_ts = random_split(self, [len_train, len_val])

        # Creación de los dataloaders
        train_dl = DataLoader(train_ts, batch_size=batch_size, shuffle=True)
        val_dl = DataLoader(val_ts, batch_size=batch_size, shuffle=False)

        #print(f"Tamaño del conjunto de entrenamiento: {len(train_ts)}")
        #print(f"Tamaño del conjunto de validación: {len(val_ts)}")

        return train_dl, val_dl