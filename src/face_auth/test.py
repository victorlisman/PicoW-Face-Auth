import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, random_split
import matplotlib.pyplot as plt
import numpy as np
import os
from PIL import Image

# Function to load the trained model
def load_model(model_path, num_classes):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    weights = models.ResNet18_Weights.DEFAULT
    model = models.resnet18(weights=weights)
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, num_classes)
    model = model.to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model

# Function to preprocess the image
def preprocess_image(image_path):
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    image = Image.open(image_path)
    image = preprocess(image).unsqueeze(0)
    return image

# Function to make a prediction
def predict(model, image_tensor):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    image_tensor = image_tensor.to(device)
    with torch.no_grad():
        outputs = model(image_tensor)
        _, preds = torch.max(outputs, 1)
    return preds

# Function to display the image and prediction
def display_prediction(image_path, prediction, class_names):
    image = Image.open(image_path)
    plt.imshow(image)
    plt.title(f'Predicted: {class_names[prediction]}')
    plt.axis('off')
    plt.show()

data_transforms = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

if __name__ == '__main__':
    # Paths and settings
    model_path = 'best_model.pth'
    image_path = 'Lisman_Victor_003.jpg'  # Path to the new image
    data_dir = 'train_data'  # Path to the train data to get class names

    # Load the dataset to get class names
    full_dataset = datasets.ImageFolder(data_dir, transform=data_transforms)
    class_names = full_dataset.classes

    # Load the trained model
    model = load_model(model_path, len(class_names))

    # Preprocess the image
    image_tensor = preprocess_image(image_path)

    # Make a prediction
    prediction = predict(model, image_tensor)

    # Display the image and prediction
    display_prediction(image_path, prediction.item(), class_names)
