#!/usr/bin/env python3
import multiprocessing
from multiprocessing import freeze_support
import torch
import torch.nn as nn
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os
import random


def main():
    # -------- Configuration --------
    data_dir = 'data/cropped_faces'
    batch_size = 32
    num_workers = 4
    model_path = 'models/model_resnet18_73p.pth'

    # -------- Device Setup --------
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # -------- Transforms --------
    test_transforms = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])

    # -------- Dataset & Loader --------
    test_dataset = datasets.ImageFolder(data_dir, transform=test_transforms)
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers
    )

    # -------- Model Setup --------
    model = models.resnet18(pretrained=False)
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, len(test_dataset.classes))
    state = torch.load(model_path, map_location=device)
    model.load_state_dict(state)
    model = model.to(device)
    model.eval()

    # -------- Testing --------
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # -------- Metrics --------
    accuracy = np.mean(np.array(all_preds) == np.array(all_labels))
    print(f"Test Accuracy: {accuracy * 100:.2f}%")

    # Detailed classification report
    print("\nClassification Report:")
    print(classification_report(all_labels, all_preds, target_names=test_dataset.classes))

    # Confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    print("\nConfusion Matrix:")
    print(cm)

    # -------- Plotting --------
    # Confusion matrix heatmap
    fig, ax = plt.subplots(figsize=(8, 8))
    cax = ax.matshow(cm, cmap='Blues')
    fig.colorbar(cax)
    ax.set_xticks(range(len(test_dataset.classes)))
    ax.set_yticks(range(len(test_dataset.classes)))
    ax.set_xticklabels(test_dataset.classes, rotation=90)
    ax.set_yticklabels(test_dataset.classes)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('True')
    plt.tight_layout()
    plt.show()

    # Per-class accuracy bar chart
    acc_per_class = cm.diagonal() / cm.sum(axis=1)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(test_dataset.classes, acc_per_class)
    ax.set_xticklabels(test_dataset.classes, rotation=90)
    ax.set_ylabel('Accuracy')
    ax.set_title('Per-Class Accuracy')
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    freeze_support()
    main()