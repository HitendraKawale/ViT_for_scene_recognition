import os
from PIL import Image
import argparse

def clean_images(directory):
    """
    finds and deletes invalid/corrupted images.
    """
    deleted_count = 0
    for root, _, files in os.walk(directory):
        for filename in files:
            # Check for common image extensions
            if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
                file_path = os.path.join(root, filename)
                try:
                    with Image.open(file_path) as img:
                        img.verify() # image valid or not
                except (IOError, SyntaxError, Image.UnidentifiedImageError):
                    print(f"Deleting invalid image: {file_path}")
                    os.remove(file_path)
                    deleted_count += 1
    
    print(f"\nCleanup complete. Deleted {deleted_count} invalid image(s).")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Clean a dataset by deleting corrupted images.")
    parser.add_argument("--dir", type=str, required=True, help="Path to the dataset directory to clean.")
    args = parser.parse_args()
    clean_images(args.dir)
