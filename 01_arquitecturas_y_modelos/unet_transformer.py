"""
==============================================================================
ARQUITECTURA U-NET TRANSFORMER PARA SEGMENTACIÓN DE SALARES
==============================================================================
Combina la U-Net clásica (Ronneberger 2015) con un bloque Transformer en el
cuello de botella (inspirado en TransUNet, Chen et al. 2021).

Estructura:
  - Encoder CNN: 4 niveles de DoubleConv + MaxPool (idéntico a U-Net clásica)
  - Bottleneck Transformer:
      * Reshape espacial → secuencia: (B, C, H, W) → (B, H*W, C)
      * Positional Encoding aprendido
      * N capas de Multi-Head Self-Attention + Feed-Forward
      * Reshape secuencia → espacial: (B, H*W, C) → (B, C, H, W)
  - Decoder CNN: 4 niveles con Skip Connections (idéntico a U-Net clásica)

Ventaja: Las convoluciones capturan features locales (texturas, bordes),
mientras que la Self-Attention en el bottleneck (16x16=256 tokens) captura
relaciones GLOBALES entre regiones distantes del salar.

Entrada:  Tensor (Batch, in_channels, 256, 256) en float32
Salida:   Logits (Batch, 1, 256, 256) -> Sigmoide -> Máscara de Salar
==============================================================================
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================================
# BLOQUES CNN (idénticos a U-Net clásica)
# ============================================================================

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
    """Nivel del Decoder: ConvTranspose2d + Concatenación Skip + DoubleConv"""
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
    """Capa final: Convolución 1x1"""
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


# ============================================================================
# BLOQUES TRANSFORMER (para el Bottleneck)
# ============================================================================

class TransformerBottleneck(nn.Module):
    """
    Bloque Transformer para el cuello de botella de la U-Net.

    Convierte el feature map espacial en una secuencia de tokens,
    aplica Self-Attention multi-cabeza para capturar dependencias globales,
    y reconstruye el feature map espacial.

    Para un input de 256x256 con 4 niveles de pooling:
      Feature map = 16x16 = 256 tokens (ligero y eficiente)
    """
    def __init__(self, embed_dim=1024, num_heads=8, num_layers=2,
                 ff_dim=2048, dropout=0.1, spatial_size=16):
        super(TransformerBottleneck, self).__init__()
        self.embed_dim = embed_dim
        self.spatial_size = spatial_size
        self.num_tokens = spatial_size * spatial_size  # 16*16 = 256

        # Positional Encoding aprendido (no sinusoidal, más flexible)
        self.pos_embedding = nn.Parameter(
            torch.randn(1, self.num_tokens, embed_dim) * 0.02
        )

        # Layer Norm de entrada
        self.input_norm = nn.LayerNorm(embed_dim)

        # Capas Transformer (Encoder-only, tipo ViT)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True  # Pre-LN (más estable para entrenamiento)
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers
        )

        # Layer Norm de salida
        self.output_norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        """
        x: (B, C, H, W) donde C=embed_dim, H=W=spatial_size
        returns: (B, C, H, W) con atención global aplicada
        """
        B, C, H, W = x.shape

        # 1. Reshape espacial → secuencia: (B, C, H, W) → (B, H*W, C)
        x_seq = x.flatten(2).transpose(1, 2)  # (B, num_tokens, embed_dim)

        # 2. Añadir Positional Encoding
        x_seq = x_seq + self.pos_embedding

        # 3. Layer Norm de entrada
        x_seq = self.input_norm(x_seq)

        # 4. Self-Attention multi-cabeza (captura relaciones globales)
        x_seq = self.transformer_encoder(x_seq)

        # 5. Layer Norm de salida
        x_seq = self.output_norm(x_seq)

        # 6. Reshape secuencia → espacial: (B, H*W, C) → (B, C, H, W)
        x_out = x_seq.transpose(1, 2).view(B, C, H, W)

        # 7. Conexión residual con la entrada original
        return x_out + x


# ============================================================================
# ARQUITECTURA PRINCIPAL: U-NET TRANSFORMER
# ============================================================================

class UNetTransformer(nn.Module):
    """
    U-Net con Transformer Bottleneck para Segmentación de Salares.

    Encoder CNN (features locales) → Transformer Bottleneck (contexto global)
    → Decoder CNN con Skip Connections (reconstrucción a resolución original)
    """
    def __init__(self, in_channels=3, out_channels=1, base_channels=64,
                 transformer_heads=8, transformer_layers=2,
                 transformer_ff_dim=2048, transformer_dropout=0.1):
        super(UNetTransformer, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.base_channels = base_channels

        b = base_channels  # 64

        # ===== ENCODER CNN (idéntico a U-Net clásica) =====
        self.inc = DoubleConv(in_channels, b)             # in -> 64 (256x256)
        self.down1 = Down(b, b * 2)                       # 64 -> 128 (128x128)
        self.down2 = Down(b * 2, b * 4)                   # 128 -> 256 (64x64)
        self.down3 = Down(b * 4, b * 8)                   # 256 -> 512 (32x32)
        self.down4 = Down(b * 8, b * 16)                  # 512 -> 1024 (16x16)

        # ===== TRANSFORMER BOTTLENECK (NUEVO) =====
        # Opera sobre el feature map 16x16 = 256 tokens de dimensión 1024
        self.transformer_bottleneck = TransformerBottleneck(
            embed_dim=b * 16,           # 1024
            num_heads=transformer_heads,
            num_layers=transformer_layers,
            ff_dim=transformer_ff_dim,
            dropout=transformer_dropout,
            spatial_size=16             # 256/2^4 = 16
        )

        # ===== DECODER CNN (idéntico a U-Net clásica) =====
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

        # Transformer Bottleneck (Self-Attention global)
        x5 = self.transformer_bottleneck(x5)  # (B, 1024, 16, 16) con atención global

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

    def contar_parametros_transformer(self):
        """Retorna el número de parámetros en el bloque Transformer."""
        return sum(p.numel() for p in self.transformer_bottleneck.parameters() if p.requires_grad)


if __name__ == "__main__":
    for n_ch, nombre in [(3, "RGB"), (4, "4 Bandas (RGB+NIR)"), (6, "Multiespectral")]:
        print(f"\n{'='*60}")
        print(f" PRUEBA U-NET TRANSFORMER {nombre} ({n_ch} canales)")
        print(f"{'='*60}")
        modelo = UNetTransformer(
            in_channels=n_ch, out_channels=1, base_channels=64,
            transformer_heads=8, transformer_layers=2,
            transformer_ff_dim=2048, transformer_dropout=0.1
        )
        total_params = modelo.contar_parametros()
        trans_params = modelo.contar_parametros_transformer()
        print(f"Parámetros totales:      {total_params:,}")
        print(f"Parámetros Transformer:  {trans_params:,} ({trans_params/total_params*100:.1f}%)")
        print(f"Parámetros CNN:          {total_params - trans_params:,}")

        x = torch.randn(2, n_ch, 256, 256)
        y = modelo(x)
        print(f"Entrada: {x.shape} -> Salida: {y.shape}")
        assert y.shape == (2, 1, 256, 256), f"Dimensiones incorrectas para {nombre}"
        print(f"Test {nombre} PASSED!")
