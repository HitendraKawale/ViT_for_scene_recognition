import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
import matplotlib.pyplot as plt
import numpy as np
import argparse
import sys
import os
from torchvision import models
from transformers import ViTForImageClassification, Dinov2ForImageClassification

# Add project root to Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from utils.dataset import PlacesDataset
from transformers import AutoImageProcessor, ViTForImageClassification

def plot_predictions(images, true_labels, top5_probs, top5_classes, class_names, num_images=5):
    """Plots images with their true and predicted labels."""
    plt.figure(figsize=(15, 3 * num_images))
    for i in range(num_images):
        ax = plt.subplot(num_images, 1, i + 1)
        
        # Un-normalize the image for display
        img = images[i].cpu().numpy().transpose((1, 2, 0))
        mean = np.array([0.5, 0.5, 0.5]) # These are the default ViT stats
        std = np.array([0.5, 0.5, 0.5])
        img = std * img + mean
        img = np.clip(img, 0, 1)
        
        ax.imshow(img)
        ax.axis("off")
        
        # Prepare the title and prediction text
        true_label_name = class_names[true_labels[i]]
        title_color = 'green' if true_label_name == top5_classes[i][0] else 'red'
        ax.set_title(f"True Label: {true_label_name}", color=title_color, fontweight='bold')
        
        pred_text = "Top-5 Predictions:\n"
        for j in range(5):
            pred_text += f"{j+1}. {top5_classes[i][j]} ({top5_probs[i][j]:.2f})\n"
            
        # Add text box with predictions
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
        ax.text(1.02, 0.5, pred_text, transform=ax.transAxes, fontsize=10,
                verticalalignment='center', bbox=props)

    plt.tight_layout(rect=[0, 0, 0.85, 1]) # Adjust layout to make space for text
    plt.show()

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- Load Model and Data (Corrected Logic) ---
    if "dinov2" in args.model_name or "google/vit" in args.model_name:
        extractor = AutoImageProcessor.from_pretrained(args.model_name, use_fast=True)
        normalize = transforms.Normalize(mean=extractor.image_mean, std=extractor.image_std)
    elif args.model_name == "resnet50":
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    else:
        raise ValueError(f"Normalization stats not defined for model: {args.model_name}")

    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        normalize
    ])
    
    full_dataset = PlacesDataset(args.data_dir)
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    _, val_data = random_split(full_dataset, [train_size, val_size], generator=torch.Generator().manual_seed(args.seed))
    
    val_data.dataset.transform = val_transform
    val_loader = DataLoader(val_data, batch_size=args.batch_size, shuffle=True)
    class_names = full_dataset.classes
    
    # Load your best model using the correct architecture
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

    # --- Find and Store Predictions ---
    found_images = 0
    batch_images, batch_labels, batch_top5_probs, batch_top5_classes = [], [], [], []

    print(f"Searching for {args.num_images} {'correct' if args.correct else 'incorrect'} predictions...")
    with torch.no_grad():
        for images, labels in val_loader:
            if found_images >= args.num_images:
                break
            
            images, labels = images.to(device), labels.to(device)
            
            # Handle different model output formats
            model_output = model(images)
            if args.model_name == "resnet50":
                outputs = model_output
            else: # For Hugging Face models
                outputs = model_output.logits
                
            probs = F.softmax(outputs, dim=1)
            top5_probs, top5_indices = torch.topk(probs, 5)
            preds = top5_indices[:, 0]
            
            matches = (preds == labels) if args.correct else (preds != labels)
            
            for i in range(len(matches)):
                if matches[i] and found_images < args.num_images:
                    batch_images.append(images[i])
                    batch_labels.append(labels[i].item())
                    batch_top5_probs.append(top5_probs[i].cpu().numpy())
                    
                    predicted_class_names = [class_names[idx] for idx in top5_indices[i].cpu().numpy()]
                    batch_top5_classes.append(predicted_class_names)
                    
                    found_images += 1

    # --- Plot Results ---
    if found_images > 0:
        plot_predictions(batch_images, batch_labels, batch_top5_probs, batch_top5_classes, class_names, num_images=found_images)
    else:
        print("Could not find enough matching examples.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize model predictions on the validation set.")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the trained model's .pth file.")
    parser.add_argument("--model_name", type=str, default="google/vit-base-patch16-224", help="Name of the model architecture.")
    parser.add_argument("--data_dir", type=str, default="data/Places2_simp", help="Path to the root dataset directory.")
    parser.add_argument("--num_images", type=int, default=5, help="Number of images to visualize.")
    parser.add_argument("--seed", type=int, default=42, help="Seed for the train/val split.")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for data loading.")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--correct", action="store_true", help="Visualize correctly classified images.")
    group.add_argument("--incorrect", action="store_true", help="Visualize incorrectly classified images.")
    args = parser.parse_args()

    # A simple trick to handle the logic based on the mutually exclusive group
    if args.incorrect:
        args.correct = False

    main(args)
