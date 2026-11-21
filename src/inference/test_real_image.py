import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

from src.inference.predict import load_model, predict_image

# Charger le modèle
model, device = load_model()

# Remplacer par le chemin réel trouvé
image_path = "data/processed/official/splits/test/images/M-34-55-B-b-4-1_46.tif"

image = Image.open(image_path).convert("RGB")

# Prédiction
result = predict_image(image, model, device)

print("\n--- DIMENSIONS ---")
print("Original :", result["original"].shape)
print("Mask :", result["mask"].shape)
print("Overlay :", result["overlay"].shape)

print("\n--- CLASSES PRESENTES ---")
print(np.unique(result["mask"], return_counts=True))
# Affichage
fig, axes = plt.subplots(1, 3, figsize=(18, 6))

axes[0].imshow(result["original"])
axes[0].set_title("Original image")

axes[1].imshow(result["colored_mask"])
axes[1].set_title("Predicted segmentation")

axes[2].imshow(result["overlay"])
axes[2].set_title("Overlay")

for ax in axes:
    ax.axis("off")

plt.tight_layout()
plt.show()

print("\nClass percentages:")
for name, percentage in result["class_percentages"].items():
    print(f"{name}: {percentage}%")