# test_dataset.py

import torch
from torchvision import transforms
from utils.dataset import PlacesDataset
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import torchvision
# Simple transforms: resize to ViT input size
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

dataset = PlacesDataset(root_dir="data/Places2_simp", transform=transform)
dataloader = DataLoader(dataset, batch_size=8, shuffle=True)

# Visualize a few samples
images, labels = next(iter(dataloader))

print("Batch shape:", images.shape)
print("Labels:", labels)

# Show images
grid_img = torchvision.utils.make_grid(images[:4], nrow=4, normalize=True, scale_each=True)
plt.figure(figsize=(8, 8))
plt.imshow(grid_img.permute(1, 2, 0))
plt.title("Sample batch from Places2_simp")
plt.axis("off")
plt.show()
