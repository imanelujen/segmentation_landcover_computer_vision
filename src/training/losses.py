
import torch
import torch.nn as nn


def get_loss_function():
    """
    Retourne la fonction de perte CrossEntropyLoss
    pour la segmentation sémantique.
    """
    return nn.CrossEntropyLoss()


if __name__ == "__main__":

    # Initialiser la fonction de perte
    criterion = get_loss_function()

    # Simuler les prédictions de U-Net
    # Batch = 2, Classes = 5, Hauteur = 512, Largeur = 512
    predictions = torch.randn(2, 5, 512, 512)

    # Simuler les masques réels
    # Chaque pixel possède une classe entre 0 et 4
    targets = torch.randint(0, 5, (2, 512, 512))

    # Calculer la perte
    loss = criterion(predictions, targets)

    print("=== TEST LOSS FUNCTION ===")
    print("Predictions shape :", predictions.shape)
    print("Targets shape :", targets.shape)
    print("Predictions dtype :", predictions.dtype)
    print("Targets dtype :", targets.dtype)
    print("Loss :", loss.item())
