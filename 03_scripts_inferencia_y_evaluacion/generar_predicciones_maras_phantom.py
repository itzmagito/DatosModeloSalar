import os
import sys
import numpy as np
import rasterio
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch

# ==============================================================================
# SCRIPT DE INFERENCIA EN SERVIDOR PHANTOM — SALINERAS DE MARAS (4 CUADRANTES)
# ==============================================================================

DATA_DIR = "/data/mmedina/datos_multiespectral/DATASET/DATASET_MULTIESPECTRAL/SALINERAS DE MARAS URUBAMBA"
OUTPUT_DIR = "/data/mmedina/predicciones_maras"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Buscar carpetas de código compartido
for p in [
    "/data/mmedina/paquete_salar_unet/codigo_compartido",
    "/data/mmedina/paquete_replicas_servidor/codigo_compartido",
    "/data/mmedina/paquete_salar_unet",
    "/data/mmedina/paquete_replicas_servidor"
]:
    if os.path.exists(p) and p not in sys.path:
        sys.path.insert(0, p)

from unet_salar import SalarUNet

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"[INFO] Dispositivo de inferencia: {device}")

# Función para cargar Salar-UNet detectando si es BASE (48M) o LARGE (82M)
def cargar_salar(checkpoint_path, in_channels):
    if not os.path.exists(checkpoint_path):
        print(f"[ALERTA] Checkpoint no encontrado: {checkpoint_path}")
        return None
    
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=True)
    state = ckpt.get('modelo_state_dict', ckpt)
    base_ch = ckpt.get('base_channels', 64)
    
    # Auto-detección de hiperparámetros
    t_layers = 4 if any('transformer_encoder.layers.3' in k for k in state.keys()) else 2
    t_heads = 16 if any('transformer_encoder.layers.0.linear1.weight' in k and state[k].shape[0] >= 4096 for k in state.keys()) else 8
    t_ff = 4096 if t_heads == 16 else 2048
    
    print(f"[OK] Cargando Salar-UNet ({in_channels} canales, {t_layers} capas, {t_heads} cabezas, ff={t_ff})")
    modelo = SalarUNet(
        in_channels=in_channels,
        out_channels=1,
        base_channels=base_ch,
        transformer_layers=t_layers,
        transformer_heads=t_heads,
        transformer_ff_dim=t_ff
    ).to(device)
    modelo.load_state_dict(state)
    modelo.eval()
    return modelo

# 1. Cargar modelos Salar-UNet (4B y Multiespectral)
ck_multi_large = "/data/mmedina/paquete_salar_unet/salar_multiespectral/resultados/500_epocas_large/checkpoints/best_model.pth"
ck_4b_large = "/data/mmedina/paquete_salar_unet/salar_4bandas/resultados/500_epocas_large/checkpoints/best_model.pth"

model_multi = cargar_salar(ck_multi_large, in_channels=6)
model_4b = cargar_salar(ck_4b_large, in_channels=4)

# 2. Los 4 cuadrantes con MENOS nubes (Junio y Julio 2025 - Estación seca)
cuadrantes = [
    ("00000_S2A_20250604_p00", "04-Jun-2025 (Centro Salineras)"),
    ("00001_S2A_20250724_p00", "24-Jul-2025 (Centro Salineras)"),
    ("00002_S2A_20250604_p00_e01", "04-Jun-2025 (Borde Norte)"),
    ("00003_S2A_20250604_p00_e02", "04-Jun-2025 (Borde Sur)")
]

def calcular_metricas(pred, gt):
    inter = np.logical_and(pred, gt).sum()
    union = np.logical_or(pred, gt).sum()
    iou = (inter + 1e-7) / (union + 1e-7)
    dice = (2.0 * inter + 1e-7) / (pred.sum() + gt.sum() + 1e-7)
    return float(iou), float(dice)

print("\n" + "="*70)
print(" GENERANDO PREDICCIONES VISUALES PARA SALINERAS DE MARAS")
print("="*70)

for idx, (base_name, desc) in enumerate(cuadrantes, start=1):
    tif_path = os.path.join(DATA_DIR, "imagen", f"{base_name}.tif")
    mask_path = os.path.join(DATA_DIR, "mascara", f"{base_name}.png")
    
    if not os.path.exists(tif_path) or not os.path.exists(mask_path):
        print(f"[!] No se encontró {base_name}, saltando...")
        continue
    
    # Leer datos satelitales (6 bandas GeoTIFF)
    with rasterio.open(tif_path) as src:
        data_6b = src.read().astype(np.float32) # (6, 256, 256)
    
    # Leer máscara Ground Truth
    gt = np.array(Image.open(mask_path)).astype(np.float32)
    if gt.max() > 1.0:
        gt = (gt > 127).astype(np.float32)
    
    # Composición RGB visual (B04 Rojo, B03 Verde, B02 Azul)
    rgb = np.stack([data_6b[2], data_6b[1], data_6b[0]], axis=-1)
    rgb_vis = np.clip(rgb * 3.5, 0.0, 1.0)
    
    # -------------------------------------------------------------
    # Inferir con Salar-UNet 4 Bandas
    # -------------------------------------------------------------
    pred_4b_bin = None
    iou_4b, dice_4b = 0.0, 0.0
    if model_4b is not None:
        t_4b = torch.from_numpy(data_6b[0:4]).unsqueeze(0).to(device)
        with torch.no_grad():
            logits_4b = model_4b(t_4b)
            probs_4b = torch.sigmoid(logits_4b)
            pred_4b_bin = (probs_4b[0, 0] > 0.5).cpu().numpy().astype(np.float32)
        iou_4b, dice_4b = calcular_metricas(pred_4b_bin, gt)

    # -------------------------------------------------------------
    # Inferir con Salar-UNet Multiespectral (6 Bandas)
    # -------------------------------------------------------------
    pred_multi_bin = None
    iou_multi, dice_multi = 0.0, 0.0
    if model_multi is not None:
        t_multi = torch.from_numpy(data_6b[0:6]).unsqueeze(0).to(device)
        with torch.no_grad():
            logits_multi = model_multi(t_multi)
            probs_multi = torch.sigmoid(logits_multi)
            pred_multi_bin = (probs_multi[0, 0] > 0.5).cpu().numpy().astype(np.float32)
        iou_multi, dice_multi = calcular_metricas(pred_multi_bin, gt)
    
    # -------------------------------------------------------------
    # Generar Panel Comparativo de Alta Resolución (4 Columnas)
    # -------------------------------------------------------------
    fig, axs = plt.subplots(1, 4, figsize=(18, 5), dpi=200)
    fig.patch.set_facecolor('#FFFFFF')
    
    # 1. RGB
    axs[0].imshow(rgb_vis)
    axs[0].set_title(f"1. RGB Color Real (Sentinel-2)\n{desc}", fontsize=10.5, fontweight='bold', color='#0F172A')
    axs[0].axis('off')
    
    # 2. Ground Truth
    axs[1].imshow(gt, cmap='gray', vmin=0, vmax=1)
    axs[1].set_title("2. Ground Truth Referencia\n(Anotación Curada)", fontsize=10.5, fontweight='bold', color='#0F172A')
    axs[1].axis('off')
    
    # 3. Predicción Salar-UNet 4B
    if pred_4b_bin is not None:
        axs[2].imshow(pred_4b_bin, cmap='gray', vmin=0, vmax=1)
        axs[2].set_title(f"3. Salar-UNet 4B (82M)\nIoU: {iou_4b*100:.2f}% | Dice: {dice_4b*100:.2f}%", 
                         fontsize=10.5, fontweight='bold', color='#2563EB')
    else:
        axs[2].text(0.5, 0.5, "Modelo 4B no disponible", ha='center', va='center')
    axs[2].axis('off')
    
    # 4. Predicción Salar-UNet Multiespectral
    if pred_multi_bin is not None:
        axs[3].imshow(pred_multi_bin, cmap='gray', vmin=0, vmax=1)
        axs[3].set_title(f"4. Salar-UNet Multi (82M) SOTA\nIoU: {iou_multi*100:.2f}% | Dice: {dice_multi*100:.2f}%", 
                         fontsize=10.5, fontweight='bold', color='#16A34A')
    else:
        axs[3].text(0.5, 0.5, "Modelo Multi no disponible", ha='center', va='center')
    axs[3].axis('off')
    
    plt.suptitle(f"Evaluación de Generalización en Salineras de Maras (Cusco) — Muestra {idx:02d}", 
                 fontsize=13, fontweight='bold', y=0.98)
    plt.tight_layout()
    
    out_fig = os.path.join(OUTPUT_DIR, f"panel_maras_muestra_{idx:02d}_{base_name}.png")
    plt.savefig(out_fig, bbox_inches='tight')
    plt.close()
    
    print(f"  [OK] Cuadrante {idx}: {os.path.basename(out_fig)}")
    if pred_4b_bin is not None:
        print(f"       -> Salar-UNet 4B:    IoU = {iou_4b*100:.2f}% | Dice = {dice_4b*100:.2f}%")
    if pred_multi_bin is not None:
        print(f"       -> Salar-UNet Multi: IoU = {iou_multi*100:.2f}% | Dice = {dice_multi*100:.2f}%")

print("\n" + "="*70)
print(f"TODAS LAS PREDICCIONES FUERON GUARDADAS EN: {OUTPUT_DIR}")
print("="*70)
