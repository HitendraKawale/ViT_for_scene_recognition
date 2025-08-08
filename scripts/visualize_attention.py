import argparse
import os
import sys
import torch
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

# Add project root to Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from utils.dataset import PlacesDataset

# Import all necessary model classes
from transformers import AutoImageProcessor, ViTForImageClassification, Dinov2ForImageClassification

def plot_attention_map(original_image, attention_map, save_path):
    """Overlays the attention map on the original image and saves it."""
    fig, ax = plt.subplots(figsize=(6, 6))
    
    # Ensure attention map is a numpy array and normalize for visualization
    if isinstance(attention_map, torch.Tensor):
        attention_map = attention_map.cpu().numpy()
    attention_map = (attention_map - np.min(attention_map)) / (np.max(attention_map) - np.min(attention_map))

    ax.imshow(original_image)
    ax.imshow(attention_map, cmap='viridis', alpha=0.6) 
    ax.axis('off')
    
    plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
    plt.close(fig)
    print(f"Attention map saved to {save_path}")

def main(args):
    # Handle the ResNet case first
    if args.model_name == "resnet50":
        print("Attention map visualization is not applicable to ResNet models.")
        return

    # --- 1. SETUP ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load feature extractor for normalization values
    extractor = AutoImageProcessor.from_pretrained(args.model_name)
    normalize = transforms.Normalize(mean=extractor.image_mean, std=extractor.image_std)

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        normalize
    ])

    # --- 2. LOAD MODEL ---
    dummy_dataset = PlacesDataset(args.data_dir)
    num_classes = len(dummy_dataset.classes)

    if "dinov2" in args.model_name:
        model = Dinov2ForImageClassification.from_pretrained(
            args.model_name, num_labels=num_classes, ignore_mismatched_sizes=True
        ).to(device)
    else: # Default to ViT
        model = ViTForImageClassification.from_pretrained(
            args.model_name, num_labels=num_classes, ignore_mismatched_sizes=True
        ).to(device)

    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model.eval()

    # --- 3. PROCESS IMAGE AND GET ATTENTION ---
    img = Image.open(args.image_path).convert("RGB")
    img_tensor = transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(img_tensor, output_attentions=True)
    
    attentions = outputs.attentions[-1]
    avg_attention = torch.mean(attentions, dim=1).squeeze(0)
    cls_attention = avg_attention[0, 1:]
    
    patch_size = 16
    grid_size = 224 // patch_size
    attention_grid = cls_attention.reshape(grid_size, grid_size)

    # --- 4. VISUALIZE ---
    resized_attention = transforms.functional.to_pil_image(attention_grid.unsqueeze(0))
    resized_attention = resized_attention.resize(img.size, resample=Image.BILINEAR)

    plot_attention_map(img, resized_attention, args.save_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize Vision Transformer Attention Maps.")
    parser.add_argument("--image_path", type=str, required=True, help="Path to the input image.")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the trained model's .pth file.")
    parser.add_argument("--save_path", type=str, required=True, help="Path to save the output attention map.")
    parser.add_argument("--model_name", type=str, default="google/vit-base-patch16-224", help="Name of the model architecture.")
    parser.add_argument("--data_dir", type=str, default="data/Places2_simp", help="Path to the dataset directory to infer class count.")
    args = parser.parse_args()
    main(args)
