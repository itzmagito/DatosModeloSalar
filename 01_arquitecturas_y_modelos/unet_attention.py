"""
==============================================================================
ARQUITECTURA ATTENTION U-NET (Oktay et al., 2018)
==============================================================================
U-Net con Attention Gates en las Skip Connections.

Las Attention Gates ponderan los feature maps del encoder usando la señal
del decoder como "contexto", suprimiendo regiones irrelevantes y resaltando
las relevantes antes de la concatenación.

Estructura:
  - Encoder CNN: 4 niveles de DoubleConv + MaxPool (idéntico a U-Net clásica)
  - Bottleneck: DoubleConv(512 → 1024) (idéntico a U-Net clásica)
  - Decoder CNN: 4 niveles con Attention Gates + Skip Connections
  - Cada skip pasa por un AttentionGate(g=decoder, x=encoder) antes de Cat

Ventaja: Los AG eliminan el ruido de las skip connections, suprimiendo
features de arena, nubes y otras regiones no-salar que el decoder ya sabe
que son irrelevantes, mejorando la precisión de bordes.

Entrada:  Tensor (Batch, in_channels, 256, 256) en float32
Salida:   Logits (Batch, 1, 256, 256) -> Sigmoide -> Máscara de Salar
==============================================================================
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================================
# BLOQUES CNN (reutilizados de U-Net clásica)
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
# ATTENTION GATE (Oktay et al., 2018)
# ============================================================================

class AttentionGate(nn.Module):
    """
    Attention Gate para skip connections.
    
    Pondera los features del encoder (x) usando la señal del decoder (g)
    como contexto. Genera un mapa de atención α ∈ [0,1] que suprime
    regiones irrelevantes del skip connection.
    
    Fórmula:
        q_g = W_g(g)      →  Proyección de la señal gating
        q_x = W_x(x)      →  Proyección de los features del encoder
        ψ = σ(W_ψ(ReLU(q_g + q_x)))  →  Coeficientes de atención
        salida = x * ψ    →  Features ponderados
    
    Args:
        F_g:   Canales de la señal gating (decoder output)
        F_l:   Canales del skip connection (encoder output)  
        F_int: Canales intermedios para la proyección (típicamente F_l // 2)
    """
    def __init__(self, F_g, F_l, F_int):
        super(AttentionGate, self).__init__()
        
        # Proyección de la señal gating (decoder)
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )
        
        # Proyección del skip connection (encoder)
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )
        
        # Coeficiente de atención
        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        
        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        """
        Args:
            g: Señal gating del decoder (B, F_g, H, W)
            x: Features del encoder / skip connection (B, F_l, H, W)
        Returns:
            Features ponderados por atención (B, F_l, H, W)
        """
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)     # (B, 1, H, W) — mapa de atención
        return x * psi          # Ponderar skip features


# ============================================================================
# UP CON ATTENTION GATE
# ============================================================================

class UpWithAttention(nn.Module):
    """
    Nivel del Decoder con Attention Gate:
    1. ConvTranspose2d para upsampling
    2. AttentionGate pondera el skip connection
    3. Concatenación del skip ponderado + upsampled
    4. DoubleConv para fusión
    """
    def __init__(self, in_channels, out_channels):
        super(UpWithAttention, self).__init__()
        
        # Upsampling aprendible
        self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
        
        # Attention Gate: g tiene in_channels//2 canales (post-upsample)
        # x (skip) tiene in_channels//2 canales (output del encoder)
        self.attention = AttentionGate(
            F_g=in_channels // 2,       # Canales del decoder (post-upsample)
            F_l=in_channels // 2,       # Canales del encoder (skip)
            F_int=in_channels // 4      # Canales intermedios
        )
        
        # DoubleConv para fusionar skip + upsampled
        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        """
        Args:
            x1: Feature map del nivel inferior (decoder) — se hace upsample
            x2: Skip connection del encoder — se pondera con AG
        """
        x1 = self.up(x1)
        
        # Ajustar tamaño si hay diferencia (por padding/tamaños impares)
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        if diffX != 0 or diffY != 0:
            x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                            diffY // 2, diffY - diffY // 2])
        
        # Attention Gate: ponderar skip con contexto del decoder
        x2 = self.attention(g=x1, x=x2)
        
        # Concatenar y fusionar
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


# ============================================================================
# ARQUITECTURA PRINCIPAL: ATTENTION U-NET
# ============================================================================

class AttentionUNet(nn.Module):
    """
    Attention U-Net para Segmentación de Salares.
    
    Idéntica a la U-Net clásica pero con Attention Gates en cada skip connection
    del decoder, que ponderan los features del encoder suprimiendo regiones
    irrelevantes (arena, nubes) antes de la concatenación.
    """
    def __init__(self, in_channels=3, out_channels=1, base_channels=64):
        super(AttentionUNet, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.base_channels = base_channels

        b = base_channels  # 64

        # ===== ENCODER CNN (idéntico a U-Net clásica) =====
        self.inc = DoubleConv(in_channels, b)             # in -> 64 (256x256)
        self.down1 = Down(b, b * 2)                       # 64 -> 128 (128x128)
        self.down2 = Down(b * 2, b * 4)                   # 128 -> 256 (64x64)
        self.down3 = Down(b * 4, b * 8)                   # 256 -> 512 (32x32)

        # ===== BOTTLENECK (idéntico a U-Net clásica) =====
        self.down4 = Down(b * 8, b * 16)                  # 512 -> 1024 (16x16)

        # ===== DECODER CNN CON ATTENTION GATES =====
        self.up1 = UpWithAttention(b * 16, b * 8)         # 1024+512 -> 512 (32x32)
        self.up2 = UpWithAttention(b * 8, b * 4)          # 512+256 -> 256 (64x64)
        self.up3 = UpWithAttention(b * 4, b * 2)          # 256+128 -> 128 (128x128)
        self.up4 = UpWithAttention(b * 2, b)              # 128+64 -> 64 (256x256)

        # ===== OUTPUT HEAD =====
        self.outc = OutConv(b, out_channels)              # 64 -> 1 (256x256)

    def forward(self, x):
        # Encoder con Skip Connections
        x1 = self.inc(x)       # (B, 64, 256, 256)
        x2 = self.down1(x1)    # (B, 128, 128, 128)
        x3 = self.down2(x2)    # (B, 256, 64, 64)
        x4 = self.down3(x3)    # (B, 512, 32, 32)
        x5 = self.down4(x4)    # (B, 1024, 16, 16)

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

    def contar_parametros_attention(self):
        """Retorna el número de parámetros en los Attention Gates."""
        total = 0
        for module in [self.up1, self.up2, self.up3, self.up4]:
            total += sum(p.numel() for p in module.attention.parameters() if p.requires_grad)
        return total


if __name__ == "__main__":
    for n_ch, nombre in [(3, "RGB"), (4, "4 Bandas (RGB+NIR)"), (6, "Multiespectral")]:
        print(f"\n{'='*60}")
        print(f" PRUEBA ATTENTION U-NET {nombre} ({n_ch} canales)")
        print(f"{'='*60}")
        modelo = AttentionUNet(
            in_channels=n_ch, out_channels=1, base_channels=64
        )
        total_params = modelo.contar_parametros()
        ag_params = modelo.contar_parametros_attention()
        print(f"Parámetros totales:       {total_params:,}")
        print(f"Parámetros Attention:     {ag_params:,} ({ag_params/total_params*100:.1f}%)")
        print(f"Parámetros CNN base:      {total_params - ag_params:,}")

        x = torch.randn(2, n_ch, 256, 256)
        y = modelo(x)
        print(f"Entrada: {x.shape} -> Salida: {y.shape}")
        assert y.shape == (2, 1, 256, 256), f"Dimensiones incorrectas para {nombre}"
        print(f"Test {nombre} PASSED!")
