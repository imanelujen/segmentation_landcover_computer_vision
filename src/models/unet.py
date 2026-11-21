
import torch
import torch.nn as nn


class DoubleConv(nn.Module):
    """
    Deux convolutions successives :
    Conv2d -> BatchNorm -> ReLU
    Conv2d -> BatchNorm -> ReLU
    """

    def __init__(self, in_channels, out_channels):

        super().__init__()

        self.block = nn.Sequential(

            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=3,
                padding=1
            ),

            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),

            nn.Conv2d(
                out_channels,
                out_channels,
                kernel_size=3,
                padding=1
            ),

            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.block(x)


class Encoder(nn.Module):

    def __init__(self, in_channels, out_channels):

        super().__init__()

        self.conv = DoubleConv(in_channels, out_channels)

        self.pool = nn.MaxPool2d(
            kernel_size=2,
            stride=2
        )

    def forward(self, x):

        features = self.conv(x)

        pooled = self.pool(features)

        return features, pooled


class Bottleneck(nn.Module):

    def __init__(self, in_channels, out_channels):

        super().__init__()

        self.conv = DoubleConv(
            in_channels,
            out_channels
        )

    def forward(self, x):

        return self.conv(x)


class Decoder(nn.Module):

    def __init__(self, in_channels, skip_channels, out_channels):

        super().__init__()

        self.up = nn.ConvTranspose2d(
            in_channels,
            out_channels,
            kernel_size=2,
            stride=2
        )

        self.conv = DoubleConv(
            out_channels + skip_channels,
            out_channels
        )

    def forward(self, x, skip):

        # Augmenter la résolution
        x = self.up(x)

        # Concaténer avec les caractéristiques de l'Encoder
        x = torch.cat([x, skip], dim=1)

        # Raffiner les caractéristiques
        x = self.conv(x)

        return x

class UNet(nn.Module):
    def __init__(self, in_channels=3, num_classes=5, base_channels=16):
        super().__init__()

        # Encoder
        self.enc1 = Encoder(in_channels, base_channels)
        self.enc2 = Encoder(base_channels, base_channels * 2)
        self.enc3 = Encoder(base_channels * 2, base_channels * 4)
        self.enc4 = Encoder(base_channels * 4, base_channels * 8)

        # Bottleneck
        self.bottleneck = Bottleneck(
            base_channels * 8,
            base_channels * 16
        )

        # Decoder
        self.dec4 = Decoder(
            base_channels * 16,
            base_channels * 8,
            base_channels * 8
        )

        self.dec3 = Decoder(
            base_channels * 8,
            base_channels * 4,
            base_channels * 4
        )

        self.dec2 = Decoder(
            base_channels * 4,
            base_channels * 2,
            base_channels * 2
        )

        self.dec1 = Decoder(
            base_channels * 2,
            base_channels,
            base_channels
        )

        # Classification de chaque pixel
        self.final_conv = nn.Conv2d(
            base_channels,
            num_classes,
            kernel_size=1
        )

    def forward(self, x):
        # Encoder
        skip1, x = self.enc1(x)
        skip2, x = self.enc2(x)
        skip3, x = self.enc3(x)
        skip4, x = self.enc4(x)

        # Bottleneck
        x = self.bottleneck(x)

        # Decoder
        x = self.dec4(x, skip4)
        x = self.dec3(x, skip3)
        x = self.dec2(x, skip2)
        x = self.dec1(x, skip1)

        # Output
        x = self.final_conv(x)

        return x

if __name__ == "__main__":
    model = UNet(
        in_channels=3,
        num_classes=5,
        base_channels=16
    )

    x = torch.randn(1, 3, 512, 512)

    with torch.no_grad():
        output = model(x)

    print("Input shape :", x.shape)
    print("Output shape :", output.shape)
    print("Nombre de paramètres :", sum(p.numel() for p in model.parameters()))



