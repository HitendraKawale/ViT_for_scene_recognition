import os
from PIL import Image
from torch.utils.data import Dataset

class PlacesDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.classes = sorted(os.listdir(root_dir))
        self.class_to_idx = {cls_name: idx for idx, cls_name in enumerate(self.classes)}

        self.image_paths = []
        self.labels = []

        for class_name in self.classes:
            class_path = os.path.join(root_dir, class_name)
            if not os.path.isdir(class_path):
                continue
            for img_name in os.listdir(class_path):
                if img_name.endswith(('.jpg', '.png', '.jpeg')):
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

