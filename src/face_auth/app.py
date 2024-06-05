import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, random_split
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
from PIL import Image


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


def predict(model, image_tensor):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    image_tensor = image_tensor.to(device)
    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
        confidence, preds = torch.max(probabilities, 1)
    return confidence.item(), preds.item()

if __name__ == '__main__':
    if len(sys.argv) != 3:
        print("Usage: python app.py <path_to_model> <path_to_image>")
        sys.exit(1)

    model_path = sys.argv[1]
    image_path = sys.argv[2]

    if not os.path.exists(model_path):
        print(f"Model path {model_path} does not exist.")
        sys.exit(1)

    if not os.path.exists(image_path):
        print(f"Image path {image_path} does not exist.")
        sys.exit(1)


    data_dir = 'train_data' 
    data_transforms = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    full_dataset = datasets.ImageFolder(data_dir, transform=data_transforms)
    class_names = full_dataset.classes


    model = load_model(model_path, len(class_names))


    image_tensor = preprocess_image(image_path)


    confidence, prediction = predict(model, image_tensor)

    if confidence > 0.5:
        print(f"Predicted: {class_names[prediction]} with confidence {confidence:.2f}")
    else:
        print("Not Recognized")
