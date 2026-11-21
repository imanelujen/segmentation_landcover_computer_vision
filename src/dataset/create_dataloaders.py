
import torch
from torch.utils.data import DataLoader

from src.dataset.landcover_dataset import LandCoverDataset


def create_dataloaders(batch_size=4, num_workers=2):

    root_dir = "data/processed/official/splits"

    # Chargement des datasets
    train_dataset = LandCoverDataset(root_dir, "train")
    val_dataset = LandCoverDataset(root_dir, "val")
    test_dataset = LandCoverDataset(root_dir, "test")

    # Création des DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=False
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=False
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=False
    )

    return train_loader, val_loader, test_loader


if __name__ == "__main__":

    train_loader, val_loader, test_loader = create_dataloaders()

    print("Nombre de batches :")
    print("Train :", len(train_loader))
    print("Validation :", len(val_loader))
    print("Test :", len(test_loader))

    # Récupération d'un batch
    images, masks = next(iter(train_loader))

    print("\n--- BATCH ---")
    print("Images :", images.shape)
    print("Masks :", masks.shape)

    print("Images dtype :", images.dtype)
    print("Masks dtype :", masks.dtype)

    print("Classes présentes :", torch.unique(masks).tolist())
