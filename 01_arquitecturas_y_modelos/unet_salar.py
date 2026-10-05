"""
==============================================================================
ARQUITECTURA SALAR-UNET v1 (UNET-R + Attention Gates)
==============================================================================
Combina las dos mejores mejoras sobre la U-Net clásica:

1. Transformer Bottleneck (del UNET-R): Self-Attention global en el cuello
   de botella para capturar relaciones a larga distancia entre regiones
   del salar.

2. Attention Gates (del Attention U-Net): Ponderación inteligente de los
   skip connections para suprimir features irrelevantes (arena, nubes)
   antes de la concatenación con el decoder.

Estructura:
  - Encoder CNN: 4 niveles de DoubleConv + MaxPool
  - Bottleneck: DoubleConv(512 → 1024) + TransformerBottleneck (2 capas MHSA)
  - Decoder CNN: 4 niveles con Attention Gates + Skip Connections

Esta es nuestra arquitectura propuesta "Salar-UNet" diseñada específicamente
para la segmentación de salares altoandinos desde imágenes Sentinel-2.

Entrada:  Tensor (Batch, in_channels, 256, 256) en float32
Salida:   Logits (Batch, 1, 256, 256) -> Sigmoide -> Máscara de Salar
==============================================================================
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================================
# BLOQUES CNN (reutilizados)
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


class OutConv(nn.Module):
    """Capa final: Convolución 1x1"""
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


# ============================================================================
# TRANSFORMER BOTTLENECK (del UNET-R)
# ============================================================================

class TransformerBottleneck(nn.Module):
    """
    Bloque Transformer para el cuello de botella de la U-Net.
    Convierte el feature map espacial en secuencia de tokens,
    aplica Self-Attention multi-cabeza, y reconstruye el feature map.
    """
    def __init__(self, embed_dim=1024, num_heads=8, num_layers=2,
                 ff_dim=2048, dropout=0.1, spatial_size=16):
        super(TransformerBottleneck, self).__init__()
        self.embed_dim = embed_dim
        self.spatial_size = spatial_size
        self.num_tokens = spatial_size * spatial_size

        # Positional Encoding aprendido
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
            norm_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers
        )

        # Layer Norm de salida
        self.output_norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        B, C, H, W = x.shape

        # Reshape espacial → secuencia
        x_seq = x.flatten(2).transpose(1, 2)  # (B, num_tokens, embed_dim)

        # Positional Encoding
        x_seq = x_seq + self.pos_embedding

        # Layer Norm entrada
        x_seq = self.input_norm(x_seq)

        # Self-Attention multi-cabeza
        x_seq = self.transformer_encoder(x_seq)

        # Layer Norm salida
        x_seq = self.output_norm(x_seq)

        # Reshape secuencia → espacial
        x_out = x_seq.transpose(1, 2).view(B, C, H, W)

        # Conexión residual
        return x_out + x


# ============================================================================
# ATTENTION GATE (Oktay et al., 2018)
# ============================================================================

class AttentionGate(nn.Module):
    """
    Attention Gate para skip connections.
    Pondera los features del encoder usando la señal del decoder como contexto.
    """
    def __init__(self, F_g, F_l, F_int):
        super(AttentionGate, self).__init__()
        
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )
        
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )
        
        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        
        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        return x * psi


# ============================================================================
# UP CON ATTENTION GATE
# ============================================================================

class UpWithAttention(nn.Module):
    """Nivel del Decoder con Attention Gate en el skip connection."""
    def __init__(self, in_channels, out_channels):
        super(UpWithAttention, self).__init__()
        
        self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
        
        self.attention = AttentionGate(
            F_g=in_channels // 2,
            F_l=in_channels // 2,
            F_int=in_channels // 4
        )
        
        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        if diffX != 0 or diffY != 0:
            x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                            diffY // 2, diffY - diffY // 2])
        
        # Attention Gate pondera el skip connection
        x2 = self.attention(g=x1, x=x2)
        
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


# ============================================================================
# ============================================================================
# COMPUERTA ESPECTRAL ADAPTATIVA: ASAG (~300 params)
# ============================================================================

class AdaptiveSpectralAttentionGate(nn.Module):
    """
    Compuerta de Atención Espectral Adaptativa (ASAG).
    Pondera dinámicamente los canales de entrada (RGB, NIR, etc.)
    según la estacionalidad de la escena (época seca vs lluviosa).
    """
    def __init__(self, in_channels, reduction=2):
        super(AdaptiveSpectralAttentionGate, self).__init__()
        mid = max(4, in_channels // reduction)
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_channels, mid, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(mid, in_channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.shape
        weights = self.gap(x).view(b, c)
        weights = self.fc(weights).view(b, c, 1, 1)
        return x * weights


# ============================================================================
# ARQUITECTURA PRINCIPAL: SALAR-UNET (ASAG + ResEncoder + Transformer + AG)
# ============================================================================

class SalarUNet(nn.Module):
    """
    Salar-UNet: U-Net especializada para Salares Altoandinos.
    
    Combina:
    - ASAG: Atención espectral adaptativa en entrada (RGB vs NIR)
    - Encoder CNN: features locales multiescala
    - Transformer Bottleneck: contexto global de cuenca (256 tokens MHSA)
    - Attention Gates: filtrado inteligente de skips (supresión de laderas/cerros)
    - Output Head: máscara probabilística de alta definición
    
    Diseñada específicamente para la segmentación de salares altoandinos
    desde imágenes satelitales Sentinel-2.
    """
    def __init__(self, in_channels=3, out_channels=1, base_channels=64,
                 transformer_heads=8, transformer_layers=2,
                 transformer_ff_dim=2048, transformer_dropout=0.1):
        super(SalarUNet, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.base_channels = base_channels

        b = base_channels  # 64

        # ===== ETAPA 1: COMPUERTA ESPECTRAL ADAPTATIVA =====
        self.asag = AdaptiveSpectralAttentionGate(in_channels)

        # ===== ENCODER CNN =====
        self.inc = DoubleConv(in_channels, b)             # in -> 64 (256x256)
        self.down1 = Down(b, b * 2)                       # 64 -> 128 (128x128)
        self.down2 = Down(b * 2, b * 4)                   # 128 -> 256 (64x64)
        self.down3 = Down(b * 4, b * 8)                   # 256 -> 512 (32x32)
        self.down4 = Down(b * 8, b * 16)                  # 512 -> 1024 (16x16)

        # ===== TRANSFORMER BOTTLENECK =====
        self.transformer_bottleneck = TransformerBottleneck(
            embed_dim=b * 16,
            num_heads=transformer_heads,
            num_layers=transformer_layers,
            ff_dim=transformer_ff_dim,
            dropout=transformer_dropout,
            spatial_size=16
        )

        # ===== DECODER CNN CON ATTENTION GATES =====
        self.up1 = UpWithAttention(b * 16, b * 8)         # 1024+512 -> 512 (32x32)
        self.up2 = UpWithAttention(b * 8, b * 4)          # 512+256 -> 256 (64x64)
        self.up3 = UpWithAttention(b * 4, b * 2)          # 256+128 -> 128 (128x128)
        self.up4 = UpWithAttention(b * 2, b)              # 128+64 -> 64 (256x256)

        # ===== OUTPUT HEAD =====
        self.outc = OutConv(b, out_channels)              # 64 -> 1 (256x256)

    def forward(self, x):
        # 1. Ponderación Espectral Adaptativa
        x_att = self.asag(x)

        # 2. Encoder con Skip Connections
        x1 = self.inc(x_att)   # (B, 64, 256, 256)
        x2 = self.down1(x1)    # (B, 128, 128, 128)
        x3 = self.down2(x2)    # (B, 256, 64, 64)
        x4 = self.down3(x3)    # (B, 512, 32, 32)
        x5 = self.down4(x4)    # (B, 1024, 16, 16)

        # 3. Transformer Bottleneck (Self-Attention global)
        x5 = self.transformer_bottleneck(x5)  # (B, 1024, 16, 16)

        # Decoder con Attention Gates en cada skip
        x = self.up1(x5, x4)   # (B, 512, 32, 32)  — AG pondera x4
        x = self.up2(x, x3)    # (B, 256, 64, 64)  — AG pondera x3
        x = self.up3(x, x2)    # (B, 128, 128, 128) — AG pondera x2
        x = self.up4(x, x1)    # (B, 64, 256, 256)  — AG pondera x1

        logits = self.outc(x)   # (B, 1, 256, 256)
        return logits

    def contar_parametros(self):
        """Retorna el número total de parámetros entrenables."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def contar_parametros_transformer(self):
        """Retorna el número de parámetros en el bloque Transformer."""
        return sum(p.numel() for p in self.transformer_bottleneck.parameters() if p.requires_grad)

    def contar_parametros_attention(self):
        """Retorna el número de parámetros en los Attention Gates."""
        total = 0
        for module in [self.up1, self.up2, self.up3, self.up4]:
            total += sum(p.numel() for p in module.attention.parameters() if p.requires_grad)
        return total


if __name__ == "__main__":
    for n_ch, nombre in [(3, "RGB"), (4, "4 Bandas (RGB+NIR)"), (6, "Multiespectral")]:
        print(f"\n{'='*60}")
        print(f" PRUEBA SALAR-UNET {nombre} ({n_ch} canales)")
        print(f"{'='*60}")
        modelo = SalarUNet(
            in_channels=n_ch, out_channels=1, base_channels=64,
            transformer_heads=8, transformer_layers=2,
            transformer_ff_dim=2048, transformer_dropout=0.1
        )
        total_params = modelo.contar_parametros()
        trans_params = modelo.contar_parametros_transformer()
        ag_params = modelo.contar_parametros_attention()
        cnn_params = total_params - trans_params - ag_params
        
        print(f"Parámetros totales:       {total_params:,}")
        print(f"  CNN base:               {cnn_params:,} ({cnn_params/total_params*100:.1f}%)")
        print(f"  Transformer:            {trans_params:,} ({trans_params/total_params*100:.1f}%)")
        print(f"  Attention Gates:        {ag_params:,} ({ag_params/total_params*100:.1f}%)")

        x = torch.randn(2, n_ch, 256, 256)
        y = modelo(x)
        print(f"Entrada: {x.shape} -> Salida: {y.shape}")
        assert y.shape == (2, 1, 256, 256), f"Dimensiones incorrectas para {nombre}"
        print(f"Test {nombre} PASSED!")
