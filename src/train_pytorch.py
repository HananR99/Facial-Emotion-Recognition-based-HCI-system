
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler

# Configuration
data_dir = '/content/dataset/data/cropped_faces'
batch_size = 32
num_epochs = 20
learning_rate = 1e-3
valid_split = 0.2  # Fraction of data for validation
test_split = 0.1   # Fraction of data for test
shuffle_dataset = True
random_seed = 42
num_workers = 4

# Device setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Data transformations
train_transforms = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])
val_splits_transforms = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Prepare dataset and split indices
full_dataset = datasets.ImageFolder(data_dir)
dataset_size = len(full_dataset)
indices = list(range(dataset_size))
if shuffle_dataset:
    np.random.seed(random_seed)
    np.random.shuffle(indices)

# Compute split sizes
val_size = int(np.floor(valid_split * dataset_size))
test_size = int(np.floor(test_split * dataset_size))
train_size = dataset_size - val_size - test_size

# Split indices
graphic_start = 0
val_idx = indices[graphic_start:graphic_start + val_size]
test_idx = indices[graphic_start + val_size:graphic_start + val_size + test_size]
train_idx = indices[graphic_start + val_size + test_size:]

# Samplers for DataLoaders
train_sampler = SubsetRandomSampler(train_idx)
valid_sampler = SubsetRandomSampler(val_idx)
test_sampler = SubsetRandomSampler(test_idx)

# Datasets with transforms
def make_dataset(transform):
    return datasets.ImageFolder(data_dir, transform=transform)

train_dataset = make_dataset(train_transforms)
val_dataset = make_dataset(val_splits_transforms)
test_dataset = make_dataset(val_splits_transforms)

# DataLoaders
train_loader = DataLoader(
    train_dataset, batch_size=batch_size, sampler=train_sampler, num_workers=num_workers
)
val_loader = DataLoader(
    val_dataset, batch_size=batch_size, sampler=valid_sampler, num_workers=num_workers
)
test_loader = DataLoader(
    test_dataset, batch_size=batch_size, sampler=test_sampler, num_workers=num_workers
)

# Model: fine-tune a pretrained ResNet18
model = models.resnet18(pretrained=True)
num_features = model.fc.in_features
model.fc = nn.Linear(num_features, len(full_dataset.classes))
model = model.to(device)

# Loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Training loop
def train_one_epoch(epoch):
    model.train()
    running_loss = 0.0
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * inputs.size(0)
    epoch_loss = running_loss / train_size
    print(f"Epoch {epoch} Train Loss: {epoch_loss:.4f}")

# Validation loop
def validate(loader, epoch, phase="Validation"):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    acc = correct / total
    print(f"Epoch {epoch} {phase} Accuracy: {acc:.4f}")

# Run training and validation
for epoch in range(1, num_epochs + 1):
    train_one_epoch(epoch)
    validate(val_loader, epoch, "Validation")

# Test accuracy after training
validate(test_loader, num_epochs, "Test")

# Save model
torch.save(model.state_dict(), 'model_resnet18.pth')
print("Training complete. Model saved to model_resnet18.pth")
