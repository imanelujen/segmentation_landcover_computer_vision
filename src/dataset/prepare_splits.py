
from pathlib import Path
import os

DATASET_DIR = Path("data/raw/LandCover.ai")
PATCH_DIR = Path("data/processed/official")

SPLITS_DIR = PATCH_DIR / "splits"

SPLITS = ["train", "val", "test"]


def prepare_split(split_name):
    """
    Organise les patches selon les listes officielles.

    Les fichiers sont liés symboliquement pour éviter
    de dupliquer les données sur le disque.
    """

    split_file = DATASET_DIR / f"{split_name}.txt"

    images_source = PATCH_DIR / "images"
    masks_source = PATCH_DIR / "masks"

    images_destination = SPLITS_DIR / split_name / "images"
    masks_destination = SPLITS_DIR / split_name / "masks"

    images_destination.mkdir(parents=True, exist_ok=True)
    masks_destination.mkdir(parents=True, exist_ok=True)

    patch_ids = [
        line.strip()
        for line in split_file.read_text().splitlines()
        if line.strip()
    ]

    for patch_id in patch_ids:

        image_source = images_source / f"{patch_id}.tif"
        mask_source = masks_source / f"{patch_id}.tif"

        image_destination = images_destination / image_source.name
        mask_destination = masks_destination / mask_source.name

        if not image_source.exists() or not mask_source.exists():
            raise FileNotFoundError(
                f"Patch incomplet : {patch_id}"
            )

        if not image_destination.exists():
            os.symlink(
                image_source.resolve(),
                image_destination
            )

        if not mask_destination.exists():
            os.symlink(
                mask_source.resolve(),
                mask_destination
            )

    print(
        f"{split_name}: {len(patch_ids)} patches organisés"
    )


def main():
    for split_name in SPLITS:
        prepare_split(split_name)

    print("\nPréparation des splits terminée.")


if __name__ == "__main__":
    main()
