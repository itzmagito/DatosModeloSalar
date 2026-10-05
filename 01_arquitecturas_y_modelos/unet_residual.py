"""
==============================================================================
ARQUITECTURA RESUNET (Zhang et al., 2018)
==============================================================================
U-Net con Bloques Residuales en lugar de DoubleConv.

Cada bloque de convolución tiene una conexión residual (shortcut):
  output = ReLU(Conv(Conv(x)) + skip(x))

La conexión residual permite que los gradientes fluyan directamente,
mejorando la convergencia y estabilidad del entrenamiento.

Estructura:
  - Encoder: 4 niveles de MaxPool + ResDoubleConv
  - Bottleneck: ResDoubleConv(512 → 1024)
  - Decoder: 4 niveles de ConvTranspose2d + Skip + ResDoubleConv

Ventaja: Entrenamiento más estable y rápida convergencia comparado con
U-Net clásica, especialmente útil para entrenamientos largos (350 épocas).

Entrada:  Tensor (Batch, in_channels, 256, 256) en float32
Salida:   Logits (Batch, 1, 256, 256) -> Sigmoide -> Máscara de Salar
==============================================================================
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================================
# BLOQUE RESIDUAL (reemplaza DoubleConv)
# ============================================================================

class ResDoubleConv(nn.Module):
    """
    Bloque Residual: (Conv2d -> BatchNorm -> ReLU -> Conv2d -> BatchNorm) + skip.
    
    Si in_channels != out_channels, se usa una convolución 1x1 para ajustar
    las dimensiones del shortcut.
    
    Fórmula: output = ReLU(F(x) + skip(x))
    donde F(x) = BN(Conv(ReLU(BN(Conv(x)))))
    y skip(x) = Conv1x1(x) si in_ch != out_ch, else x
    """
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super(ResDoubleConv, self).__init__()
        if not mid_channels:
            mid_channels = out_channels
        
        # Rama principal: dos convoluciones 3x3
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
        )
        
        # Shortcut: ajuste de canales si es necesario
        if in_channels != out_channels:
            self.skip = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.skip = nn.Identity()
        
        # Activación final post-suma
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        identity = self.skip(x)
        out = self.double_conv(x)
        return self.relu(out + identity)


# ============================================================================
# BLOQUES ENCODER/DECODER
# ============================================================================

class Down(nn.Module):
    """Nivel del Encoder: MaxPool 2x2 seguido de ResDoubleConv"""
    def __init__(self, in_channels, out_channels):
        super(Down, self).__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            ResDoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Nivel del Decoder: ConvTranspose2d + Concatenación Skip + ResDoubleConv"""
    def __init__(self, in_channels, out_channels):
        super(Up, self).__init__()
        self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
        self.conv = ResDoubleConv(in_channels, out_channels)

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
    """Capa final: Convolución 1x1"""
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


# ============================================================================
# ARQUITECTURA PRINCIPAL: RESUNET
# ============================================================================

class ResUNet(nn.Module):
    """
    ResUNet para Segmentación de Salares.
    
    Idéntica estructura a la U-Net clásica pero con conexiones residuales
    en cada bloque convolucional. Los gradientes fluyen directamente a través
    de los shortcuts, permitiendo entrenamiento más estable y profundo.
    
    Misma cantidad de parámetros que U-Net clásica + ~2% extra por las
    convoluciones 1x1 de los shortcuts.
    """
    def __init__(self, in_channels=3, out_channels=1, base_channels=64):
        super(ResUNet, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.base_channels = base_channels

        b = base_channels  # 64

        # ===== ENCODER con ResDoubleConv =====
        self.inc = ResDoubleConv(in_channels, b)          # in -> 64 (256x256)
        self.down1 = Down(b, b * 2)                       # 64 -> 128 (128x128)
        self.down2 = Down(b * 2, b * 4)                   # 128 -> 256 (64x64)
        self.down3 = Down(b * 4, b * 8)                   # 256 -> 512 (32x32)

        # ===== BOTTLENECK con ResDoubleConv =====
        self.down4 = Down(b * 8, b * 16)                  # 512 -> 1024 (16x16)

        # ===== DECODER con ResDoubleConv =====
        self.up1 = Up(b * 16, b * 8)                      # 1024+512 -> 512 (32x32)
        self.up2 = Up(b * 8, b * 4)                       # 512+256 -> 256 (64x64)
        self.up3 = Up(b * 4, b * 2)                       # 256+128 -> 128 (128x128)
        self.up4 = Up(b * 2, b)                           # 128+64 -> 64 (256x256)

        # ===== OUTPUT HEAD =====
        self.outc = OutConv(b, out_channels)              # 64 -> 1 (256x256)

    def forward(self, x):
        # Encoder con Skip Connections
        x1 = self.inc(x)       # (B, 64, 256, 256)
        x2 = self.down1(x1)    # (B, 128, 128, 128)
        x3 = self.down2(x2)    # (B, 256, 64, 64)
        x4 = self.down3(x3)    # (B, 512, 32, 32)
        x5 = self.down4(x4)    # (B, 1024, 16, 16)

        # Decoder con Skip Connections
        x = self.up1(x5, x4)   # (B, 512, 32, 32)
        x = self.up2(x, x3)    # (B, 256, 64, 64)
        x = self.up3(x, x2)    # (B, 128, 128, 128)
        x = self.up4(x, x1)    # (B, 64, 256, 256)

        logits = self.outc(x)   # (B, 1, 256, 256)
        return logits

    def contar_parametros(self):
        """Retorna el número total de parámetros entrenables."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


if __name__ == "__main__":
    for n_ch, nombre in [(3, "RGB"), (4, "4 Bandas (RGB+NIR)"), (6, "Multiespectral")]:
        print(f"\n{'='*60}")
        print(f" PRUEBA RESUNET {nombre} ({n_ch} canales)")
        print(f"{'='*60}")
        modelo = ResUNet(
            in_channels=n_ch, out_channels=1, base_channels=64
        )
        total_params = modelo.contar_parametros()
        print(f"Parámetros totales: {total_params:,}")

        x = torch.randn(2, n_ch, 256, 256)
        y = modelo(x)
        print(f"Entrada: {x.shape} -> Salida: {y.shape}")
        assert y.shape == (2, 1, 256, 256), f"Dimensiones incorrectas para {nombre}"
        print(f"Test {nombre} PASSED!")
