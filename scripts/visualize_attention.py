import argparse
import os
import sys
import torch
import torch.nn.functional as F
from torchvision import transforms
from transformers import AutoFeatureExtractor, ViTForImageClassification
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

# Add project root to Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from utils.dataset import PlacesDataset 


def plot_attention_map(original_image, attention_map, save_path):
    """Overlays the attention map on the original image and saves it."""
    fig, ax = plt.subplots(figsize=(6, 6))
    
    # Ensure attention map is a numpy array
    if isinstance(attention_map, torch.Tensor):
        attention_map = attention_map.cpu().numpy()
        
    attention_map = (attention_map - np.min(attention_map)) / (np.max(attention_map) - np.min(attention_map))

    ax.imshow(original_image)
    ax.imshow(attention_map, cmap='viridis', alpha=0.6) 
    ax.axis('off')
    
    plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
    plt.close(fig) # Close the figure to free up memory
    print(f"Attention map saved to {save_path}")

def main(args):
    # --- 1. SETUP ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load feature extractor for normalization values
    extractor = AutoFeatureExtractor.from_pretrained(args.model_name)
    normalize = transforms.Normalize(mean=extractor.image_mean, std=extractor.image_std)

    # Use the same transformations as validation (no data augmentation)
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        normalize
    ])

    # --- 2. LOAD MODEL ---
    # We need to know the number of classes the model was trained on
    # A simple way is to instantiate a dummy dataset to get the class list
    dummy_dataset = PlacesDataset(args.data_dir)
    num_classes = len(dummy_dataset.classes)

    model = ViTForImageClassification.from_pretrained(
        args.model_name,
        num_labels=num_classes,
        ignore_mismatched_sizes=True
    ).to(device)

    # Load the state dict from your best trained model
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model.eval()

    # --- 3. PROCESS IMAGE AND GET ATTENTION ---
    img = Image.open(args.image_path).convert("RGB")
    img_tensor = transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        # Add 'output_attentions=True' to get the attention weights
        outputs = model(img_tensor, output_attentions=True)
    
    # Attention weights will be a tuple, one for each layer.
    # We'll use the last layer's attention scores.
    # Shape: (batch_size, num_heads, num_patches+1, num_patches+1)
    attentions = outputs.attentions[-1] # Last layer's attentions
    num_heads = attentions.shape[1]
    
    # Average the attention weights across all heads
    avg_attention = torch.mean(attentions, dim=1).squeeze(0)

    # To visualize, we look at the attention from the [CLS] token to all other patches
    cls_attention = avg_attention[0, 1:] # Exclude attention to self ([CLS] token)
    
    # Reshape the attention scores to a 2D grid
    # For a 224x224 image with 16x16 patches, this is 14x14
    patch_size = 16
    grid_size = 224 // patch_size
    attention_grid = cls_attention.reshape(grid_size, grid_size).cpu().numpy()

    # --- 4. VISUALIZE ---
    # Resize the 14x14 attention grid to the original image size
    resized_attention = Image.fromarray(attention_grid).resize(img.size, resample=Image.BILINEAR)

    plot_attention_map(img, resized_attention, args.save_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize Vision Transformer Attention Maps.")
    parser.add_argument("--image_path", type=str, required=True, help="Path to the input image.")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the trained model's .pth file.")
    parser.add_argument("--save_path", type=str, required=True, help="Path to save the output attention map.")
    parser.add_argument("--model_name", type=str, default="google/vit-base-patch16-224", help="Name of the pre-trained ViT model.")
    parser.add_argument("--data_dir", type=str, default="data/Places2_simp", help="Path to the dataset directory to infer class count.")
    args = parser.parse_args()
    main(args)
