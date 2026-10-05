"""
==============================================================================
ARQUITECTURA U-NET CLÁSICA PURA (Ronneberger et al., 2015)
==============================================================================
Implementación genérica parametrizable por número de canales de entrada.
Usada para modelos RGB (3), 4 bandas (4) y Multiespectral (6).

Entrada:  Tensor (Batch, in_channels, 256, 256) en float32
Salida:   Logits (Batch, 1, 256, 256) -> Sigmoide -> Máscara de Salar
==============================================================================
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    """Bloque fundamental: (Conv2d -> BatchNorm -> ReLU) x 2"""
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super(DoubleConv, self).__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Nivel del Encoder: MaxPool 2x2 seguido de DoubleConv"""
    def __init__(self, in_channels, out_channels):
        super(Down, self).__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Nivel del Decoder: Upsampling + Concatenación de Skip Connection + DoubleConv"""
    def __init__(self, in_channels, out_channels):
        super(Up, self).__init__()
        self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        if diffX != 0 or diffY != 0:
            x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                            diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    """Capa final: Convolución 1x1 para proyectar a la cantidad de clases deseadas."""
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


class UNet(nn.Module):
    """
    Red Neuronal Convolucional U-Net para Segmentación Semántica.
    Parametrizable por número de canales de entrada.
    """
    def __init__(self, in_channels=3, out_channels=1, base_channels=64):
        super(UNet, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.base_channels = base_channels

        b = base_channels
        # Encoder (Contracting Path)
        self.inc = DoubleConv(in_channels, b)
        self.down1 = Down(b, b * 2)
        self.down2 = Down(b * 2, b * 4)
        self.down3 = Down(b * 4, b * 8)

        # Cuello de botella (Bottleneck)
        self.down4 = Down(b * 8, b * 16)

        # Decoder (Expanding Path)
        self.up1 = Up(b * 16, b * 8)
        self.up2 = Up(b * 8, b * 4)
        self.up3 = Up(b * 4, b * 2)
        self.up4 = Up(b * 2, b)

        # Output Head
        self.outc = OutConv(b, out_channels)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits

    def contar_parametros(self):
        """Retorna el número total de parámetros entrenables."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


if __name__ == "__main__":
    for n_ch, nombre in [(3, "RGB"), (4, "4 Bandas (RGB+NIR)"), (6, "Multiespectral")]:
        print(f"\n=== PRUEBA U-NET {nombre} ({n_ch} canales) ===")
        modelo = UNet(in_channels=n_ch, out_channels=1, base_channels=64)
        print(f"Parámetros entrenables: {modelo.contar_parametros():,}")
        x = torch.randn(2, n_ch, 256, 256)
        y = modelo(x)
        print(f"Entrada: {x.shape} -> Salida: {y.shape}")
        assert y.shape == (2, 1, 256, 256), f"Dimensiones incorrectas para {nombre}"
        print(f"Test {nombre} passed!")
