
from src.dataset.landcover_dataset import LandCoverDataset

# Charger le dataset d'entraînement
dataset = LandCoverDataset(
    root_dir="data/processed/official/splits",
    split="train"
)

print("Nombre d'images :", len(dataset))

# Charger un échantillon
image, mask = dataset[0]

print("\n--- IMAGE ---")
print("Shape :", image.shape)
print("Type :", image.dtype)
print("Min :", image.min().item())
print("Max :", image.max().item())

print("\n--- MASQUE ---")
print("Shape :", mask.shape)
print("Type :", mask.dtype)
print("Classes :", mask.unique().tolist())
