
from pathlib import Path
from PIL import Image

# Dossier contenant les orthophotos originales
input_dir = Path("data/raw/LandCover.ai/images")

# Dossier de sortie
output_dir = Path("data/processed/format_tests")
output_dir.mkdir(parents=True, exist_ok=True)

# Trouver une image TIFF
tif_files = list(input_dir.glob("*.tif"))

if not tif_files:
    tif_files = list(input_dir.glob("*.tiff"))

if not tif_files:
    raise FileNotFoundError("Aucune image TIFF trouvée.")

image_path = tif_files[0]

print("Image source :", image_path)

# Charger l'image
image = Image.open(image_path).convert("RGB")

# Extraire une zone de 512x512
patch = image.crop((0, 0, 512, 512))

# Sauvegarder dans les deux formats
patch.save(output_dir / "test_image.png")
patch.save(output_dir / "test_image.jpg", quality=95)

print("Images créées :")
print(output_dir / "test_image.png")
print(output_dir / "test_image.jpg")
print("Dimensions :", patch.size)
