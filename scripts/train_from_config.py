
import argparse
import os
import sys
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from transformers import AutoImageProcessor, ViTForImageClassification
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import csv
from datetime import datetime
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from utils.dataset import PlacesDataset

def train(config):
    """
    Main training and evaluation function.
    """
    # --- 1. CONFIGURATION AND DIRECTORY SETUP ---
    
    # Create a unique run name with a timestamp
    run_name = f'{config["run_name"]}_{datetime.now().strftime("%Y%m%d_%H%M%S")}'
    save_dir = os.path.join(config["log_dir"], run_name)
    os.makedirs(save_dir, exist_ok=True)

    # Save the config file used for this run
    with open(os.path.join(save_dir, "config.yaml"), "w") as f:
        yaml.dump(config, f)
    
    # Device and precision setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = config["precision"] == "fp16" and device.type == "cuda"
    scaler = torch.amp.GradScaler(device="cuda", enabled=use_amp)
    print(f"Using device: {device} | Mixed Precision: {'Enabled' if use_amp else 'Disabled'}")

    # --- 2. DATA LOADING AND TRANSFORMATION ---
    
    # Load feature extractor for normalization values
    extractor = AutoImageProcessor.from_pretrained(config["model_name"], use_fast=True)
    normalize = transforms.Normalize(mean=extractor.image_mean, std=extractor.image_std)

    # Dynamically build augmentation pipeline from config
    train_transforms_list = [transforms.Resize((224, 224))]
    
    # Map string names to torchvision transform classes
    transform_map = {
        "RandomHorizontalFlip": transforms.RandomHorizontalFlip,
        "ColorJitter": transforms.ColorJitter,
        "RandomRotation": transforms.RandomRotation,
        "RandomAffine": transforms.RandomAffine,
        "RandomGrayscale": transforms.RandomGrayscale
    }

    if "augmentations" in config:
        for aug in config["augmentations"]:
            if aug["name"] in transform_map:
                # Add the transform with its parameters
                train_transforms_list.append(transform_map[aug["name"]](**aug["params"]))
                print(f"Added augmentation: {aug['name']}")

    # Add the mandatory transforms at the end
    train_transforms_list.extend([transforms.ToTensor(), normalize])
    
    train_transform = transforms.Compose(train_transforms_list)

    # No augmentation for the validation set
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        normalize
    ])

    # [cite_start]Load dataset and apply 80/20 split [cite: 72]
    full_dataset = PlacesDataset(config["data_dir"], transform=train_transform)
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_data, val_data = random_split(full_dataset, [train_size, val_size], generator=torch.Generator().manual_seed(config["seed"]))
    
    # Apply the correct transformation to the validation set
    val_data.dataset.transform = val_transform
    
    train_loader = DataLoader(train_data, batch_size=config["batch_size"], shuffle=True, num_workers=4)
    val_loader = DataLoader(val_data, batch_size=config["batch_size"], shuffle=False, num_workers=4)
    class_names = full_dataset.classes
    print(f"Classes: {len(class_names)} | Train images: {len(train_data)} | Val images: {len(val_data)}")

# --- 3. MODEL, OPTIMIZER, AND LOSS FUNCTION ---
    
    model = ViTForImageClassification.from_pretrained(
        config["model_name"],
        num_labels=len(class_names),
        ignore_mismatched_sizes=True
    ).to(device)

    # Optimizer setup from config
    optimizer_config = config["optimizer"]
    if optimizer_config["name"] == "AdamW":
        optimizer = optim.AdamW(model.parameters(), 
                                lr=optimizer_config["lr"], 
                                weight_decay=optimizer_config.get("weight_decay", 0.01))
        print("Using AdamW optimizer.")
    elif optimizer_config["name"] == "SGD":
        optimizer = optim.SGD(model.parameters(), 
                              lr=optimizer_config["lr"], 
                              momentum=optimizer_config.get("momentum", 0.9))
        print("Using SGD optimizer.")
    else:
        raise ValueError(f"Unsupported optimizer: {optimizer_config['name']}")

    # Scheduler setup from config
    scheduler_config = config["scheduler"]
    if scheduler_config["name"] == "cosine":
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config["epochs"])
        print("Using CosineAnnealingLR scheduler.")
    elif scheduler_config["name"] == "step":
        scheduler = optim.lr_scheduler.StepLR(optimizer, 
                                              step_size=scheduler_config["step_size"], 
                                              gamma=scheduler_config["lr_gamma"])
        print("Using StepLR scheduler.")
    else:
        scheduler = None
        print("No scheduler selected.")
        
    # Loss function
    criterion = nn.CrossEntropyLoss()

    # --- 4. LOGGING SETUP ---
    
    writer = SummaryWriter(log_dir=save_dir) # TensorBoard logger [cite: 77]
    metrics_path = os.path.join(save_dir, "metrics.csv")
    with open(metrics_path, "w", newline="") as f:
        csv_writer = csv.writer(f)
        csv_writer.writerow(["epoch", "train_loss", "train_acc", "val_acc", "val_top5_acc"])

    # --- 5. TRAINING AND VALIDATION LOOP ---
    
    best_val_acc = 0
    patience_counter = 0

    for epoch in range(1, config["epochs"] + 1):
        model.train()
        total_loss, correct, total = 0, 0, 0
        
        loop = tqdm(train_loader, desc=f"Epoch {epoch}/{config['epochs']}", leave=False)
        for images, labels in loop:
            images, labels = images.to(device), labels.to(device)
            
            with torch.amp.autocast(device_type="cuda", enabled=use_amp):
                outputs = model(images).logits
                loss = criterion(outputs, labels)

            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            loop.set_postfix(loss=loss.item())

        train_loss = total_loss / len(train_loader)
        train_acc = correct / total
        
        model.eval()
        val_correct_top1, val_correct_top5, val_total = 0, 0, 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images).logits
                
                # Top-1 accuracy
                _, predicted = torch.max(outputs, 1)
                val_total += labels.size(0)
                val_correct_top1 += (predicted == labels).sum().item()

                # Top-5 accuracy
                _, top5_preds = outputs.topk(5, 1, True, True)
                val_correct_top5 += torch.eq(top5_preds, labels.view(-1, 1)).sum().item()

        val_acc = val_correct_top1 / val_total
        val_top5_acc = val_correct_top5 / val_total
        
        print(f"Epoch {epoch} -> Train Loss: {train_loss:.3f}, Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}, Val Top-5: {val_top5_acc:.4f}")
        
        # Log to TensorBoard and CSV
        writer.add_scalar("Loss/train", train_loss, epoch)
        writer.add_scalar("Accuracy/train", train_acc, epoch)
        writer.add_scalar("Accuracy/validation_Top1", val_acc, epoch)
        writer.add_scalar("Accuracy/validation_Top5", val_top5_acc, epoch)
        writer.add_scalar("Misc/learning_rate", scheduler.get_last_lr()[0], epoch)
        with open(metrics_path, "a", newline="") as f:
            csv_writer = csv.writer(f)
            csv_writer.writerow([epoch, f"{train_loss:.4f}", f"{train_acc:.4f}", f"{val_acc:.4f}", f"{val_top5_acc:.4f}"])

        scheduler.step()
        
        # Save best model and handle early stopping
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), os.path.join(save_dir, "vit_best.pth"))
            print(f" Saved best model with Val Acc: {best_val_acc:.4f}")
            patience_counter = 0
        else:
            patience_counter += 1
            print(f" No improvement for {patience_counter} epoch(s).")
            if patience_counter >= config["patience"]:
                print(" Early stopping triggered.")
                break

    writer.close()

    # --- 6. FINAL EVALUATION AND CONFUSION MATRIX ---
    
    print("Generating confusion matrix with best model...")
    model.load_state_dict(torch.load(os.path.join(save_dir, "vit_best.pth")))
    model.eval()
    
    all_preds, all_labels = [], []
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images).logits
            all_preds.extend(outputs.argmax(1).cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(22, 22))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names, annot_kws={"size": 8})
    plt.xlabel("Predicted", fontsize=14)
    plt.ylabel("True", fontsize=14)
    plt.title("Confusion Matrix (Validation Set)", fontsize=16)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "confusion_matrix.png"), dpi=300)
    print(f" Confusion matrix saved to {save_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to the YAML config file.")
    args = parser.parse_args()
    
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)
        
    train(config)

