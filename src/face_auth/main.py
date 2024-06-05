import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, random_split
import matplotlib.pyplot as plt
import numpy as np
import os

# Hyperparameters
batch_size = 32
learning_rate = 0.001
num_epochs = 25
validation_split = 0.2

# Data directories
data_dir = 'train_data'

# Data transformations
data_transforms = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Load the dataset
full_dataset = datasets.ImageFolder(data_dir, transform=data_transforms)

# Split the dataset into training and validation sets
dataset_size = len(full_dataset)
val_size = int(np.floor(validation_split * dataset_size))
train_size = dataset_size - val_size
train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

# Data loaders
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

# Class names
class_names = full_dataset.classes

# Check if GPU is available
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Load the pre-trained ResNet18 model with the updated weights parameter
weights = models.ResNet18_Weights.DEFAULT
model = models.resnet18(weights=weights)

# Modify the final layer to match the number of classes
num_features = model.fc.in_features
model.fc = nn.Linear(num_features, len(class_names))

# Move the model to the device
model = model.to(device)

# Loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Training the model
def train_model(model, criterion, optimizer, num_epochs=25):
    best_model_wts = model.state_dict()
    best_acc = 0.0

    for epoch in range(num_epochs):
        print(f'Epoch {epoch}/{num_epochs - 1}')
        print('-' * 10)

        for phase in ['train', 'valid']:
            if phase == 'train':
                model.train()
                dataloader = train_loader
                dataset_size = train_size
            else:
                model.eval()
                dataloader = val_loader
                dataset_size = val_size

            running_loss = 0.0
            running_corrects = 0

            for inputs, labels in dataloader:
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / dataset_size
            epoch_acc = running_corrects.double() / dataset_size

            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

            if phase == 'valid' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = model.state_dict()

        print()

    print(f'Best val Acc: {best_acc:4f}')

    model.load_state_dict(best_model_wts)
    return model

if __name__ == '__main__':
    model = train_model(model, criterion, optimizer, num_epochs=num_epochs)

    # Save the best model
    torch.save(model.state_dict(), 'best_model.pth')

    # Plot a few validation images along with predictions
    def imshow(inp, title=None):
        inp = inp.numpy().transpose((1, 2, 0))
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        inp = std * inp + mean
        inp = np.clip(inp, 0, 1)
        plt.imshow(inp)
        if title is not None:
            plt.title(title)
        plt.pause(0.001)

    # Get a batch of validation data
    inputs, classes = next(iter(val_loader))

    inputs = inputs.to(device)
    classes = classes.to(device)

    outputs = model(inputs)
    _, preds = torch.max(outputs, 1)

    # Plot the images and predictions
    plt.figure(figsize=(15, 9))
    for idx in range(min(batch_size, len(inputs))):
        ax = plt.subplot(4, 8, idx + 1)
        ax.axis('off')
        ax.set_title(f'predicted: {class_names[preds[idx]]}')
        imshow(inputs.cpu().data[idx])

    plt.show()
