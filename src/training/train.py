
import torch
import torch.nn as nn
import torch.optim as optim

from src.models.unet import UNet
from src.dataset.create_dataloaders import create_dataloaders
from src.training.losses import get_loss_function


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()

    total_loss = 0.0

    for images, masks in loader:
        images = images.to(device)
        masks = masks.to(device)

        # Réinitialiser les gradients
        optimizer.zero_grad()

        # Prédictions
        outputs = model(images)

        # Calcul de la perte
        loss = criterion(outputs, masks)

        # Backpropagation
        loss.backward()

        # Mise à jour des paramètres
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(loader)


def validate(model, loader, criterion, device):
    model.eval()

    total_loss = 0.0

    with torch.no_grad():
        for images, masks in loader:
            images = images.to(device)
            masks = masks.to(device)

            outputs = model(images)

            loss = criterion(outputs, masks)

            total_loss += loss.item()

    return total_loss / len(loader)


def main():

    # Reproductibilité
    torch.manual_seed(42)

    # Utiliser le GPU s'il est disponible
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print("Device :", device)

    # Charger les DataLoaders
    train_loader, val_loader, test_loader = create_dataloaders()

    # Initialiser U-Net
    model = UNet(
        in_channels=3,
        num_classes=5,
        base_channels=16
    ).to(device)

    # Fonction de perte
    criterion = get_loss_function()

    # Optimiseur
    optimizer = optim.Adam(
        model.parameters(),
        lr=0.001,
        weight_decay=0.0001
    )

    # Nombre d'epochs
    epochs = 30

    best_val_loss = float("inf")

    # Créer le dossier checkpoints
    import os
    os.makedirs("checkpoints", exist_ok=True)

    print("\n=== DEBUT ENTRAINEMENT ===")

    for epoch in range(epochs):

        train_loss = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device
        )

        val_loss = validate(
            model,
            val_loader,
            criterion,
            device
        )

        print(
            f"Epoch [{epoch + 1}/{epochs}] "
            f"Train Loss: {train_loss:.4f} "
            f"Val Loss: {val_loss:.4f}"
        )

        # Sauvegarder le meilleur modèle
        if val_loss < best_val_loss:

            best_val_loss = val_loss

            torch.save(
                model.state_dict(),
                "checkpoints/unet_best.pth"
            )

            print("Meilleur modèle sauvegardé !")

    print("\n=== ENTRAINEMENT TERMINE ===")
    print("Meilleure Val Loss :", best_val_loss)


if __name__ == "__main__":
    main()
