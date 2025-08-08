import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms, datasets, models
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import numpy as np
import argparse
import os
from PIL import Image

# Import all necessary model classes
from transformers import AutoImageProcessor, ViTForImageClassification, Dinov2ForImageClassification

def is_valid_image(path):
    """Checks if an image file can be opened and is not corrupted."""
    try:
        with Image.open(path) as img:
            img.verify() # Verify the image integrity
        return True
    except (IOError, SyntaxError, Image.UnidentifiedImageError):
        print(f"Skipping corrupted or invalid image: {path}")
        return False

def evaluate(args):
    """Evaluates the model on a custom test set."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- 1. Define Normalization and Transformations ---
    # First, determine the correct normalization based on the model
    if "dinov2" in args.model_name or "google/vit" in args.model_name:
        extractor = AutoImageProcessor.from_pretrained(args.model_name, use_fast=True)
        normalize = transforms.Normalize(mean=extractor.image_mean, std=extractor.image_std)
    elif args.model_name == "resnet50":
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    else:
        raise ValueError(f"Normalization stats not defined for model: {args.model_name}")

    # Use the same transformations as in the validation set (no augmentation)
    test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        normalize
    ])

    # --- 2. Load Dataset ---
    test_dataset = datasets.ImageFolder(
        root=args.test_dir, 
        transform=test_transform, 
        allow_empty=True, 
        is_valid_file=is_valid_image
    )
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
    class_names = test_dataset.classes
    print(f"Found {len(test_dataset)} images in {len(class_names)} classes.")

    # --- 3. Load Model ---
    if "dinov2" in args.model_name:
        model = Dinov2ForImageClassification.from_pretrained(
            args.model_name, num_labels=len(class_names), ignore_mismatched_sizes=True
        ).to(device)
    elif "google/vit" in args.model_name:
        model = ViTForImageClassification.from_pretrained(
            args.model_name, num_labels=len(class_names), ignore_mismatched_sizes=True
        ).to(device)
    elif args.model_name == "resnet50":
        model = models.resnet50(weights='IMAGENET1K_V1')
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, len(class_names))
        model = model.to(device)
    else:
        raise ValueError(f"Unsupported model name: {args.model_name}")

    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model.eval()

    # --- 4. Run Evaluation ---
    all_preds, all_labels = [], []
    correct_top1, correct_top5, total = 0, 0, 0

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            
            # Conditionally handle model output
            model_output = model(images)
            if args.model_name == "resnet50":
                outputs = model_output
            else: # For Hugging Face models
                outputs = model_output.logits
            
            # Top-1 accuracy
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct_top1 += (predicted == labels).sum().item()
            
            # Top-5 accuracy
            _, top5_preds = outputs.topk(5, 1, True, True)
            correct_top5 += torch.eq(top5_preds, labels.view(-1, 1)).sum().item()

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # --- 5. Print and Save Results ---
    top1_acc = 100 * correct_top1 / total
    top5_acc = 100 * correct_top5 / total
    print(f"\n--- Test Set Results ---")
    print(f"Top-1 Accuracy: {top1_acc:.2f}%")
    print(f"Top-5 Accuracy: {top5_acc:.2f}%")

    # Generate and save confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(22, 22))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names, annot_kws={"size": 8})
    plt.xlabel("Predicted", fontsize=14)
    plt.ylabel("True", fontsize=14)
    plt.title("Confusion Matrix (Custom Test Set)", fontsize=16)
    plt.tight_layout()
    
    save_path = os.path.join(os.path.dirname(args.model_path), "test_set_confusion_matrix.png")
    plt.savefig(save_path, dpi=300)
    print(f"Confusion matrix saved to {save_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate a trained model on the custom test set.")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the trained model's .pth file.")
    parser.add_argument("--test_dir", type=str, required=True, help="Path to the root directory of the test set.")
    parser.add_argument("--model_name", type=str, default="google/vit-base-patch16-224", help="Name of the model architecture.")
    parser.add_argument("--batch_size", type=int, default=32)
    args = parser.parse_args()
    evaluate(args)
