
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap

from src.dataset.create_dataloaders import create_dataloaders


CLASS_NAMES = {
    0: "Background",
    1: "Building",
    2: "Woodland",
    3: "Water",
    4: "Road"
}

# Une couleur par classe
COLORS = [
    "#000000",  # Background
    "#FF0000",  # Building
    "#008000",  # Woodland
    "#0000FF",  # Water
    "#FFFF00"   # Road
]

cmap = ListedColormap(COLORS)


def visualize_batch():

    train_loader, _, _ = create_dataloaders()

    images, masks = next(iter(train_loader))

    fig, axes = plt.subplots(4, 3, figsize=(15, 18))

    for i in range(4):

        # Image RGB : [C, H, W] -> [H, W, C]
        image = images[i].permute(1, 2, 0).numpy()

        # Masque
        mask = masks[i].numpy()

        # Overlay
        axes[i, 0].imshow(image)
        axes[i, 0].set_title(f"Image {i + 1}")

        axes[i, 1].imshow(mask, cmap=cmap, vmin=0, vmax=4)
        axes[i, 1].set_title("Masque")

        axes[i, 2].imshow(image)
        axes[i, 2].imshow(
            mask,
            cmap=cmap,
            vmin=0,
            vmax=4,
            alpha=0.4
        )
        axes[i, 2].set_title("Overlay")

        for j in range(3):
            axes[i, j].axis("off")

    plt.tight_layout()
    plt.savefig("reports/visualization_batch.png", dpi=150)
    plt.show()


if __name__ == "__main__":
    visualize_batch()
