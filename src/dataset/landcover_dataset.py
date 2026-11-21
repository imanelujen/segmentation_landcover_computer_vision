
from pathlib import Path

import numpy as np
import rasterio
import torch
from torch.utils.data import Dataset


class LandCoverDataset(Dataset):
    """
    Dataset PyTorch pour la segmentation sémantique
    du jeu de données LandCover.ai.

    Chaque élément contient :
    - Une image RGB normalisée.
    - Un masque avec les identifiants de classes 0 à 4.
    """

    def __init__(self, root_dir, split):
        self.root_dir = Path(root_dir) / split

        self.images_dir = self.root_dir / "images"
        self.masks_dir = self.root_dir / "masks"

        self.image_paths = sorted(
            self.images_dir.glob("*.tif")
        )

        if not self.image_paths:
            raise RuntimeError(
                f"Aucune image trouvée dans {self.images_dir}"
            )

        # Vérifier que chaque image possède son masque
        for image_path in self.image_paths:
            mask_path = self.masks_dir / image_path.name

            if not mask_path.exists():
                raise FileNotFoundError(
                    f"Masque manquant : {mask_path}"
                )

    def __len__(self):
        """Retourne le nombre d'images dans le dataset."""
        return len(self.image_paths)

    def __getitem__(self, index):
        """
        Charge une image et son masque à partir de leur index.
        """

        image_path = self.image_paths[index]
        mask_path = self.masks_dir / image_path.name

        # Lecture de l'image RGB
        with rasterio.open(image_path) as src:
            image = src.read([1, 2, 3])

        # Conversion CHW vers HWC
        image = np.transpose(image, (1, 2, 0))

        # Normalisation des valeurs entre 0 et 1
        image = image.astype(np.float32) / 255.0

        # Conversion HWC vers CHW
        image = np.transpose(image, (2, 0, 1))

        # Lecture du masque
        with rasterio.open(mask_path) as src:
            mask = src.read(1)

        # Vérifier les identifiants des classes
        if not np.isin(mask, [0, 1, 2, 3, 4]).all():
            raise ValueError(
                f"Identifiant de classe inconnu : {mask_path}"
            )

        # Conversion en tenseurs PyTorch
        image = torch.from_numpy(
            np.ascontiguousarray(image)
        )

        mask = torch.from_numpy(
            mask.astype(np.int64)
        )

        return image, mask
