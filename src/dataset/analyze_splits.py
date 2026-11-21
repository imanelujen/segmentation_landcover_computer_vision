
from pathlib import Path
from collections import Counter

import numpy as np
import rasterio
from tqdm import tqdm
import pandas as pd
import matplotlib.pyplot as plt


BASE_DIR = Path("data/processed/official/splits")

CLASS_NAMES = {
    0: "Background",
    1: "Building",
    2: "Woodland",
    3: "Water",
    4: "Road",
}


def analyze_split(split_name):
    """
    Calcule le nombre de pixels appartenant à chaque classe
    dans un ensemble donné.
    """

    mask_dir = BASE_DIR / split_name / "masks"
    mask_paths = sorted(mask_dir.glob("*.tif"))

    pixel_counts = Counter()

    for mask_path in tqdm(
        mask_paths,
        desc=f"Analyse {split_name}"
    ):
        with rasterio.open(mask_path) as src:
            mask = src.read(1)

        values, counts = np.unique(
            mask,
            return_counts=True
        )

        for value, count in zip(values, counts):
            pixel_counts[int(value)] += int(count)

    total_pixels = sum(pixel_counts.values())

    results = {}

    for class_id, class_name in CLASS_NAMES.items():
        count = pixel_counts[class_id]

        percentage = (
            100 * count / total_pixels
            if total_pixels > 0 else 0
        )

        results[class_name] = {
            "pixels": count,
            "percentage": percentage,
        }

    return results


def main():
    all_results = {}

    for split in ["train", "val", "test"]:
        all_results[split] = analyze_split(split)

    # Affichage des résultats
    rows = []

    for split, results in all_results.items():
        for class_name, values in results.items():
            rows.append({
                "Split": split,
                "Classe": class_name,
                "Pixels": values["pixels"],
                "Pourcentage": round(
                    values["percentage"], 3
                ),
            })

    df = pd.DataFrame(rows)

    print("\nDistribution des classes :")
    print(df.to_string(index=False))

    # Tableau croisé pour le graphique
    pivot = df.pivot(
        index="Classe",
        columns="Split",
        values="Pourcentage"
    )

    ax = pivot.plot(
        kind="bar",
        figsize=(11, 6)
    )

    ax.set_title("Distribution des classes par ensemble")
    ax.set_ylabel("Pourcentage des pixels (%)")
    ax.set_xlabel("Classe")
    ax.legend(title="Ensemble")
    plt.xticks(rotation=30)
    plt.tight_layout()
    plt.savefig(
        "reports/class_distribution.png",
        dpi=150
    )
    plt.show()


if __name__ == "__main__":
    main()
