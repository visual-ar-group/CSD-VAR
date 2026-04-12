
"""
This script creates a grid visualization of images from the evaluation directory.
"""

from pathlib import Path

import matplotlib.pyplot as plt
from PIL import Image


def create_image_grid():
    # Get all jpg images from the evaluation folder and its subfolders
    eval_dir = Path("data/evaluation")
    image_paths = list(eval_dir.rglob("*.jpg"))

    if not image_paths:
        print("No .jpg images found in data/evaluation directory or its subfolders")
        return

    # Sort paths to ensure consistent ordering
    image_paths = sorted(image_paths)

    # Calculate grid dimensions
    n_images = len(image_paths)
    n_rows = 4  # Fixed number of rows
    n_cols = (n_images + n_rows - 1) // n_rows  # Ceiling division to ensure all images fit

    # Create figure with subplots
    fig = plt.figure(figsize=(5 * n_cols, 20))  # Adjusted figure size for new layout

    # Load and display images
    for idx, img_path in enumerate(image_paths):
        # Read image
        img = Image.open(img_path)

        # Create subplot
        # Convert from row-major to column-major indexing
        row = idx % n_rows
        col = idx // n_rows
        ax = fig.add_subplot(n_rows, n_cols, col * n_rows + row + 1)

        ax.imshow(img)
        ax.axis("off")
        # Show relative path from evaluation directory
        relative_path = img_path.relative_to(eval_dir)
        ax.set_title(str(relative_path), fontsize=8)

    # Adjust layout and save
    plt.tight_layout()
    plt.savefig("grid_visualization.png")
    plt.close()


if __name__ == "__main__":
    create_image_grid()
