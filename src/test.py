import os
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import argparse

def load_model(pth_path, num_classes, device):
    # Initialize the model architecture
    model = models.resnet18(pretrained=False)
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, num_classes)
    model.load_state_dict(torch.load(pth_path, map_location=device))
    model.to(device)
    model.eval()
    return model

def predict_image(model, img_path, class_names, device):
    # Define the same transforms you used for validation
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])

    # Load image
    img = Image.open(img_path).convert('RGB')
    inp = preprocess(img).unsqueeze(0).to(device)  # add batch dim

    # Forward pass
    with torch.no_grad():
        outputs = model(inp)
        probs = torch.softmax(outputs, dim=1)
        conf, pred_idx = torch.max(probs, 1)

    return class_names[pred_idx.item()], conf.item()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Image Classification Inference')
    parser.add_argument('image', help='Path to input image')
    parser.add_argument('--model', default='models/model_resnet18_73p.pth',
                        help='Path to .pth model file')
    parser.add_argument('--data-dir', required=True,
                        help='Root data directory (must have same class subfolders)')
    args = parser.parse_args()

    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Recover class names from your training dataset folder structure
    from torchvision.datasets import ImageFolder
    dataset = ImageFolder(args.data_dir)
    class_names = dataset.classes

    # Load model
    model = load_model(args.model, num_classes=len(class_names), device=device)

    # Predict
    label, confidence = predict_image(model, args.image, class_names, device)
    print(f'Predicted: {label} ({confidence*100:.1f}% confidence)')
