
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import rasterio
from rasterio.windows import Window


# Chemins des données
DATASET_DIR = Path("data/raw/LandCover.ai")
IMAGES_DIR = DATASET_DIR / "images"
MASKS_DIR = DATASET_DIR / "masks"

# Correspondance entre les identifiants et les classes
CLASS_NAMES = {
    0: "Background",
    1: "Building",
    2: "Woodland",
    3: "Water",
    4: "Road",
}

# Couleurs RGB pour visualiser les classes
CLASS_COLORS = np.array([
    [0, 0, 0],        # Background : noir
    [255, 0, 0],      # Building : rouge
    [0, 180, 0],      # Woodland : vert
    [0, 100, 255],    # Water : bleu
    [255, 220, 0],    # Road : jaune
], dtype=np.uint8)


def visualize_sample(image_path, mask_path):
    """
    Affiche une image aérienne, son masque et leur superposition.
    On utilise une fenêtre de 1024x1024 pour limiter la mémoire.
    """

    window_size = 1024

    with rasterio.open(image_path) as image_src:
        width = min(window_size, image_src.width)
        height = min(window_size, image_src.height)

        # Extraire une zone centrale de l'image
        col = (image_src.width - width) // 2
        row = (image_src.height - height) // 2

        window = Window(col, row, width, height)

        # Lecture des trois bandes RGB
        image = image_src.read([1, 2, 3], window=window)
        image = np.transpose(image, (1, 2, 0))

    with rasterio.open(mask_path) as mask_src:
        mask = mask_src.read(1, window=window)

    # Construire une image colorée à partir des identifiants
    colored_mask = CLASS_COLORS[mask]

    # Superposer le masque à l'image RGB
    overlay = image.astype(np.float32).copy()
    valid_pixels = mask > 0

    overlay[valid_pixels] = (
        0.5 * image[valid_pixels].astype(np.float32)
        + 0.5 * colored_mask[valid_pixels].astype(np.float32)
    )

    overlay = np.clip(overlay, 0, 255).astype(np.uint8)

    # Affichage des trois visualisations
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    axes[0].imshow(image)
    axes[0].set_title("Image aérienne RGB")

    axes[1].imshow(colored_mask)
    axes[1].set_title("Masque de segmentation")

    axes[2].imshow(overlay)
    axes[2].set_title("Superposition image + masque")

    for ax in axes:
        ax.axis("off")

    # Légende des classes
    legend = [
        plt.Line2D(
            [0], [0],
            marker="s",
            color="w",
            markerfacecolor=CLASS_COLORS[class_id] / 255,
            markersize=10,
            label=class_name,
        )
        for class_id, class_name in CLASS_NAMES.items()
    ]

    fig.legend(handles=legend, loc="lower center", ncol=5)
    plt.tight_layout()
    plt.show()


def main():
    # Sélectionner une image et son masque
    image_path = IMAGES_DIR / "M-33-20-D-c-4-2.tif"
    mask_path = MASKS_DIR / "M-33-20-D-c-4-2.tif"

    if not image_path.exists() or not mask_path.exists():
        raise FileNotFoundError("Image ou masque introuvable.")

    visualize_sample(image_path, mask_path)


if __name__ == "__main__":
    main()
