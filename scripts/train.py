import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import torch
from torch import nn
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from transformers import AutoModelForImageClassification
from utils.dataset import PlacesDataset
from tqdm import tqdm

# Allow import from parent directory
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# Utility: Top-K Accuracy
def top_k_accuracy(output, target, k=5):
    with torch.no_grad():
        _, pred = output.topk(k, dim=1)
        correct = pred.eq(target.view(-1, 1).expand_as(pred))
        return correct.any(dim=1).float().mean().item()

# Paths and parameters
data_path = "data/Places2_simp"
num_classes = 40
batch_size = 32
epochs = 5
best_val_acc = 0.0

# Transforms
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(degrees=20),
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
    transforms.ToTensor()
])

val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

# Dataset and Splits
full_dataset = PlacesDataset(root_dir=data_path, transform=train_transform)
train_size = int(0.8 * len(full_dataset))
val_size = len(full_dataset) - train_size
train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
val_dataset.dataset.transform = val_transform

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

# Load and modify ViT
model = AutoModelForImageClassification.from_pretrained(
    "google/vit-base-patch16-224",
    num_labels=num_classes,
    ignore_mismatched_sizes=True
).to(device)

optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)
loss_fn = nn.CrossEntropyLoss()

# Training loop
for epoch in range(epochs):
    model.train()
    total_loss, correct, top5_total = 0.0, 0, 0

    for images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
        images, labels = images.to(device), labels.to(device)
        outputs = model(images).logits
        loss = loss_fn(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        correct += (outputs.argmax(dim=1) == labels).sum().item()
        top5_total += top_k_accuracy(outputs, labels, k=5) * images.size(0)

    train_acc = correct / len(train_loader.dataset)
    train_top5_acc = top5_total / len(train_loader.dataset)
    print(f"Epoch {epoch+1} - Train loss: {total_loss:.4f} - Train Acc: {train_acc:.4f} - Top-5: {train_top5_acc:.4f}")

    # Validation
    model.eval()
    val_correct, val_top5_total = 0, 0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images).logits
            val_correct += (outputs.argmax(dim=1) == labels).sum().item()
            val_top5_total += top_k_accuracy(outputs, labels, k=5) * images.size(0)

    val_acc = val_correct / len(val_loader.dataset)
    val_top5_acc = val_top5_total / len(val_loader.dataset)
    print(f"Epoch {epoch+1} - Val Acc: {val_acc:.4f} - Top-5: {val_top5_acc:.4f}")

    # Save best model
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        os.makedirs("results/checkpoints", exist_ok=True)
        save_path = "results/checkpoints/vit_best.pth"
        torch.save(model.state_dict(), save_path)
        print(f"✔️ Saved new best model with Val Acc: {val_acc:.4f}")

