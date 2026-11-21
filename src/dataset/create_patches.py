
from pathlib import Path

import rasterio
from rasterio.windows import Window
from tqdm import tqdm


DATASET_DIR = Path("data/raw/LandCover.ai")
IMAGES_DIR = DATASET_DIR / "images"
MASKS_DIR = DATASET_DIR / "masks"

OUTPUT_DIR = Path("data/processed/official")
OUTPUT_IMAGES = OUTPUT_DIR / "images"
OUTPUT_MASKS = OUTPUT_DIR / "masks"

PATCH_SIZE = 512


def create_patches(image_path, mask_path):
    """
    Reproduit le découpage officiel de LandCover.ai.

    Les patches sont numérotés dans l'ordre de parcours :
    gauche à droite, puis de haut en bas.

    Tous les patches complets sont conservés,
    y compris ceux contenant uniquement Background.
    """

    image_name = image_path.stem
    patch_count = 0
    tile_index = 0

    with rasterio.open(image_path) as image_src, \
         rasterio.open(mask_path) as mask_src:

        if (
            image_src.width != mask_src.width
            or image_src.height != mask_src.height
        ):
            raise ValueError(
                f"Dimensions incompatibles : {image_name}"
            )

        # Parcours identique à celui du script officiel
        for row in range(0, image_src.height, PATCH_SIZE):
            for col in range(0, image_src.width, PATCH_SIZE):

                window = Window(
                    col, row, PATCH_SIZE, PATCH_SIZE
                )

                # Vérifier que le patch est complet
                if (
                    row + PATCH_SIZE <= image_src.height
                    and col + PATCH_SIZE <= image_src.width
                ):

                    image_patch = image_src.read(
                        window=window
                    )

                    mask_patch = mask_src.read(
                        1, window=window
                    )

                    # Identifiant utilisé dans train.txt,
                    # val.txt et test.txt
                    patch_name = f"{image_name}_{tile_index}.tif"

                    image_profile = image_src.profile.copy()
                    image_profile.update(
                        height=PATCH_SIZE,
                        width=PATCH_SIZE,
                        transform=rasterio.windows.transform(
                            window, image_src.transform
                        ),
                    )

                    mask_profile = mask_src.profile.copy()
                    mask_profile.update(
                        height=PATCH_SIZE,
                        width=PATCH_SIZE,
                        transform=rasterio.windows.transform(
                            window, mask_src.transform
                        ),
                    )

                    with rasterio.open(
                        OUTPUT_IMAGES / patch_name,
                        "w",
                        **image_profile
                    ) as dst:
                        dst.write(image_patch)

                    with rasterio.open(
                        OUTPUT_MASKS / patch_name,
                        "w",
                        **mask_profile
                    ) as dst:
                        dst.write(mask_patch, 1)

                    patch_count += 1

                # Important : même un patch incomplet
                # fait avancer l'index, comme dans split.py
                tile_index += 1

    return patch_count


def main():
    OUTPUT_IMAGES.mkdir(parents=True, exist_ok=True)
    OUTPUT_MASKS.mkdir(parents=True, exist_ok=True)

    image_files = sorted(IMAGES_DIR.glob("*.tif"))

    total_patches = 0

    for image_path in tqdm(image_files, desc="Images"):

        mask_path = MASKS_DIR / image_path.name

        if not mask_path.exists():
            raise FileNotFoundError(
                f"Masque manquant : {mask_path}"
            )

        total_patches += create_patches(
            image_path, mask_path
        )

    print("\nDécoupage officiel reproduit.")
    print(f"Nombre total de patches : {total_patches}")
    print(f"Images : {OUTPUT_IMAGES}")
    print(f"Masques : {OUTPUT_MASKS}")


if __name__ == "__main__":
    main()
