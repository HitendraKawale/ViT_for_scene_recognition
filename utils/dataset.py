import os
from PIL import Image
from torch.utils.data import Dataset

class PlacesDataset(Dataset):
    IMG_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")

    def __init__(self, root_dir, transform=None, max_per_class=None):
        """Image-folder dataset.

        Args:
            root_dir: directory with one sub-folder per class.
            transform: torchvision transform applied to each image.
            max_per_class: optional cap on images loaded per class, handy for
                quick experiments or laptop-friendly training runs.
        """
        self.root_dir = root_dir
        self.transform = transform
        # Only real sub-directories count as classes. This skips stray files
        # such as macOS's .DS_Store that would otherwise become phantom classes.
        self.classes = sorted(
            entry for entry in os.listdir(root_dir)
            if os.path.isdir(os.path.join(root_dir, entry))
        )
        self.class_to_idx = {cls_name: idx for idx, cls_name in enumerate(self.classes)}

        self.image_paths = []
        self.labels = []

        for class_name in self.classes:
            class_path = os.path.join(root_dir, class_name)
            images = sorted(
                img for img in os.listdir(class_path)
                if img.lower().endswith(self.IMG_EXTENSIONS)
            )
            if max_per_class is not None:
                images = images[:max_per_class]
            for img_name in images:
                self.image_paths.append(os.path.join(class_path, img_name))
                self.labels.append(self.class_to_idx[class_name])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        label = self.labels[idx]
        image = Image.open(image_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, label

