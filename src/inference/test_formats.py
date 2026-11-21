
from pathlib import Path
from PIL import Image

from src.inference.predict import load_model, predict_image


# Charger le modèle une seule fois
model, device = load_model()

# Dossier des résultats
output_dir = Path("reports/format_tests")
output_dir.mkdir(parents=True, exist_ok=True)

# Images à tester
image_paths = [
     Path("data/processed/format_tests"),
]

# Récupérer des images JPG et PNG disponibles
extensions = {".jpg", ".jpeg", ".png"}

files = []

for folder in image_paths:
    if folder.exists():
        files.extend(
            p for p in folder.rglob("*")
            if p.is_file() and p.suffix.lower() in extensions
        )

if not files:
    print("Aucune image JPG ou PNG trouvée.")
else:
    for image_path in files[:3]:

        print(f"\nTest : {image_path}")

        image = Image.open(image_path).convert("RGB")

        result = predict_image(image, model, device)

        output_name = image_path.stem

        Image.fromarray(result["colored_mask"]).save(
            output_dir / f"{output_name}_mask.png"
        )

        Image.fromarray(result["overlay"]).save(
            output_dir / f"{output_name}_overlay.png"
        )

        print("Dimensions :", result["original"].shape)
        print("Répartition :", result["class_percentages"])

print("\nTests terminés.")
