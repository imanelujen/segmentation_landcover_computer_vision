
import torch
import numpy as np

from PIL import Image
from pathlib import Path

from src.models.unet import UNet


# Configuration
NUM_CLASSES = 5
IMAGE_SIZE = 512

CLASS_NAMES = [
    "Background",
    "Building",
    "Woodland",
    "Water",
    "Road"
]

CLASS_COLORS = np.array([
    [0, 0, 0],       # Background
    [255, 0, 0],     # Building
    [0, 180, 0],     # Woodland
    [0, 0, 255],     # Water
    [255, 255, 0]    # Road
], dtype=np.uint8)


def load_model(checkpoint_path=None, device=None):
    """
    Charge le modèle U-Net et ses poids entraînés.
    """

    if device is None:
        device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

    if checkpoint_path is None:
        project_root = Path(__file__).resolve().parents[2]
        checkpoint_path = (
            project_root / "checkpoints" / "unet_best.pth"
        )

    checkpoint_path = Path(checkpoint_path)

    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Checkpoint introuvable : {checkpoint_path}"
        )

    model = UNet(
        in_channels=3,
        num_classes=NUM_CLASSES
    ).to(device)

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=False
    )

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    print(f"Modèle chargé depuis : {checkpoint_path}")
    print(f"Meilleure époque : {checkpoint['epoch']}")

    return model, device


def predict_tile(tile, model, device):
    """
    Effectue la prédiction sur une tuile de 512x512.
    """

    tile_array = tile.astype(np.float32) / 255.0

    # HWC -> CHW
    tensor = torch.from_numpy(
        tile_array.transpose(2, 0, 1)
    ).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(tensor)
        prediction = torch.argmax(outputs, dim=1)

    return prediction.squeeze(0).cpu().numpy().astype(np.uint8)


def predict_large_image(original, model, device):
    """
    Découpe une grande image en tuiles de 512x512,
    prédit chaque tuile et reconstruit le masque.
    """

    height, width = original.shape[:2]

    # Masque final
    full_mask = np.zeros((height, width), dtype=np.uint8)

    for y in range(0, height, IMAGE_SIZE):

        for x in range(0, width, IMAGE_SIZE):

            # Dimensions réelles de la tuile
            y_end = min(y + IMAGE_SIZE, height)
            x_end = min(x + IMAGE_SIZE, width)

            tile = original[y:y_end, x:x_end]

            tile_height, tile_width = tile.shape[:2]

            # Compléter les bords pour obtenir 512x512
            padded_tile = np.zeros(
                (IMAGE_SIZE, IMAGE_SIZE, 3),
                dtype=np.uint8
            )

            padded_tile[:tile_height, :tile_width] = tile

            # Prédiction
            tile_prediction = predict_tile(
                padded_tile,
                model,
                device
            )

            # Reconstituer uniquement la partie réelle
            full_mask[y:y_end, x:x_end] = (
                tile_prediction[:tile_height, :tile_width]
            )

    return full_mask


def predict_image(image, model, device):
    """
    Segmentation d'une image aérienne de taille quelconque.

    Entrée :
        image : image PIL ou tableau NumPy RGB

    Sortie :
        original
        mask
        colored_mask
        overlay
        class_percentages
    """

    # Conversion en RGB
    if isinstance(image, np.ndarray):
        image = Image.fromarray(image.astype(np.uint8))

    image = image.convert("RGB")

    original = np.array(image)

    # Prédiction par tuiles
    model.eval()

    mask = predict_large_image(
        original,
        model,
        device
    )

    # Carte colorée
    colored_mask = CLASS_COLORS[mask]

    # Overlay
    overlay = (
        0.5 * original.astype(np.float32)
        + 0.5 * colored_mask.astype(np.float32)
    )

    overlay = np.clip(
        overlay, 0, 255
    ).astype(np.uint8)

    # Pourcentage de chaque classe
    total_pixels = mask.size

    class_percentages = {}

    for class_id, class_name in enumerate(CLASS_NAMES):

        percentage = (
            np.sum(mask == class_id) / total_pixels
        ) * 100

        class_percentages[class_name] = round(
            float(percentage), 2
        )

    return {
        "original": original,
        "mask": mask,
        "colored_mask": colored_mask,
        "overlay": overlay,
        "class_percentages": class_percentages
    }
