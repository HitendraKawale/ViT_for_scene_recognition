import torch
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import argparse
import os
import sys
from PIL import Image

# Add project root to Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from utils.device import get_device
from utils.models import build_model, get_normalization, forward_logits

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
    device = get_device()
    print(f"Using device: {device}")

    # --- 1. Define Normalization and Transformations ---
    mean, std = get_normalization(args.model_name)
    normalize = transforms.Normalize(mean=mean, std=std)

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
    model = build_model(args.model_name, len(class_names)).to(device)
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model.eval()

    # --- 4. Run Evaluation ---
    all_preds, all_labels = [], []
    correct_top1, correct_top5, total = 0, 0, 0
    topk = min(5, len(class_names))  # guard against datasets with fewer than 5 classes

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = forward_logits(model, args.model_name, images)


            # Top-1 accuracy
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct_top1 += (predicted == labels).sum().item()
            
            # Top-5 accuracy (top-k where k = min(5, num_classes))
            _, top5_preds = outputs.topk(topk, 1, True, True)
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
