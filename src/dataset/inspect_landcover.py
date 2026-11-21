
"""
Module : inspect_landcover.py

Objectif :
    Examiner la structure du dataset LandCover.ai,
    vérifier les images et leurs masques,
    identifier les classes présentes.

Auteur : Imane
Projet : Segmentation sémantique d'images aériennes
"""

from pathlib import Path
from collections import Counter

import numpy as np
import rasterio


# Chemin vers le dataset original
DATASET_DIR = Path("data/raw/LandCover.ai")

IMAGES_DIR = DATASET_DIR / "images"
MASKS_DIR = DATASET_DIR / "masks"


# Correspondance entre les valeurs des pixels et les classes
CLASS_NAMES = {
    0: "Background",
    1: "Building",
    2: "Woodland",
    3: "Water",
    4: "Road"
}


def inspect_image(image_path):
    """
    Affiche les caractéristiques techniques d'une image.

    On lit uniquement les métadonnées pour éviter
    de charger toute l'image en mémoire.
    """

    with rasterio.open(image_path) as src:

        print("\n--- IMAGE ---")
        print("Nom :", image_path.name)
        print("Dimensions :", src.width, "x", src.height)
        print("Nombre de bandes :", src.count)
        print("Type :", src.dtypes)
        print("CRS :", src.crs)


def inspect_mask(mask_path):
    """
    Analyse les valeurs présentes dans un masque.

    Le traitement par fenêtres permet d'éviter
    de charger le masque complet en mémoire.
    """

    class_counts = Counter()

    with rasterio.open(mask_path) as src:

        print("\n--- MASQUE ---")
        print("Nom :", mask_path.name)
        print("Dimensions :", src.width, "x", src.height)
        print("Nombre de bandes :", src.count)
        print("Type :", src.dtypes)
        print("CRS :", src.crs)

        # Parcourir les blocs du masque
        for _, window in src.block_windows(1):

            mask = src.read(1, window=window)

            # Compter les pixels de chaque classe
            values, counts = np.unique(
                mask,
                return_counts=True
            )

            for value, count in zip(values, counts):
                class_counts[int(value)] += int(count)

    print("\nDistribution des classes :")

    total_pixels = sum(class_counts.values())

    for class_id, count in sorted(class_counts.items()):

        class_name = CLASS_NAMES.get(
            class_id,
            "Classe inconnue"
        )

        percentage = 100 * count / total_pixels

        print(
            f"{class_id} - {class_name}: "
            f"{count:,} pixels ({percentage:.2f}%)"
        )


def main():
    """
    Point d'entrée du programme.
    """

    if not IMAGES_DIR.exists() or not MASKS_DIR.exists():
        raise FileNotFoundError(
            "Les dossiers images et masks sont introuvables."
        )

    # Récupérer les fichiers TIFF
    images = sorted(IMAGES_DIR.glob("*.tif"))
    masks = sorted(MASKS_DIR.glob("*.tif"))

    print("Nombre d'images :", len(images))
    print("Nombre de masques :", len(masks))

    # Vérifier la correspondance des fichiers
    image_names = {p.stem for p in images}
    mask_names = {p.stem for p in masks}

    print(
        "Images sans masque :",
        len(image_names - mask_names)
    )

    print(
        "Masques sans image :",
        len(mask_names - image_names)
    )

    if not images or not masks:
        raise ValueError("Aucun fichier TIFF trouvé.")

    # Examiner une paire représentative
    common_names = sorted(image_names & mask_names)

    if not common_names:
        raise ValueError("Aucune paire image/masque correspondante.")

    sample_name = common_names[0]

    image_path = IMAGES_DIR / f"{sample_name}.tif"
    mask_path = MASKS_DIR / f"{sample_name}.tif"

    inspect_image(image_path)
    inspect_mask(mask_path)


if __name__ == "__main__":
    main()
