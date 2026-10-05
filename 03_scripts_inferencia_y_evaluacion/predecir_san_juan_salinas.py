import os
import sys
import rasterio
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch

# ==============================================================================
# SCRIPT DE INFERENCIA MULTI-MODELO — SAN JUAN DE SALINAS (PUNO)
# ==============================================================================
# Evalúa y compara en simultáneo:
#   1. U-Net Multiespectral Clásico (350 ep)
#   2. UNET-R Transformer Multiespectral (350 ep)
#   3. Salar-UNet 4-Bandas Large 82M (500 ep)
#   4. Salar-UNet Multiespectral Large 82M SOTA (500 ep)
# ==============================================================================

DATA_DIR = "/data/mmedina/datos_multiespectral/DATASET/DATASET_MULTIESPECTRAL/SAN JUAN DE SALINAS - PUNO"
OUTPUT_DIR = "/data/mmedina/predicciones_san_juan_salinas"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Incorporar carpetas de código compartido al path
for p in [
    "/data/mmedina/paquete_salar_unet/codigo_compartido",
    "/data/mmedina/paquete_replicas_servidor/codigo_compartido",
    "/data/mmedina/datos_multiespectral/modelo_multiespectral",
    "/data/mmedina/paquete_salar_unet",
    "/data/mmedina/paquete_replicas_servidor"
]:
    if os.path.exists(p) and p not in sys.path:
        sys.path.insert(0, p)

from unet_salar import SalarUNet
from unet import UNet
from unet_transformer import UNetTransformer

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"[INFO] Dispositivo de inferencia: {device}")

# 1. Definición de candidatos a evaluar
model_candidates = [
    {
        "nombre": "U-Net Multi Clásico",
        "tipo": "unet",
        "canales": 6,
        "color": "#0284C7",
        "paths": [
            "/data/mmedina/datos_multiespectral/modelo_multiespectral/checkpoints/best_model.pth",
            "/data/mmedina/paquete_replicas_servidor/unet_multiespectral/resultados/RUN_3/350_epocas/checkpoints/best_model.pth",
            "/data/mmedina/paquete_replicas_servidor/unet_multiespectral/resultados/350_epocas/checkpoints/best_model.pth",
            "/data/mmedina/paquete_replicas_servidor/unet_multiespectral/resultados/RUN_2/350_epocas/checkpoints/best_model.pth"
        ]
    },
    {
        "nombre": "UNET-R Transformer Multi",
        "tipo": "transformer",
        "canales": 6,
        "color": "#7C3AED",
        "paths": [
            "/data/mmedina/paquete_replicas_servidor/transformer_multiespectral/resultados/RUN_3/350_epocas/checkpoints/best_model.pth",
            "/data/mmedina/paquete_replicas_servidor/transformer_multiespectral/resultados/350_epocas/checkpoints/best_model.pth",
            "/data/mmedina/paquete_replicas_servidor/transformer_multiespectral/resultados/RUN_2/350_epocas/checkpoints/best_model.pth"
        ]
    },
    {
        "nombre": "Salar-UNet 4B (82M)",
        "tipo": "salar",
        "canales": 4,
        "color": "#2563EB",
        "paths": [
            "/data/mmedina/paquete_salar_unet/salar_4bandas/resultados/500_epocas_large/checkpoints/best_model.pth"
        ]
    },
    {
        "nombre": "Salar-UNet Multi (82M) SOTA",
        "tipo": "salar",
        "canales": 6,
        "color": "#16A34A",
        "paths": [
            "/data/mmedina/paquete_salar_unet/salar_multiespectral/resultados/500_epocas_large/checkpoints/best_model.pth"
        ]
    }
]

# Función de carga robusta
modelos_cargados = []
for mc in model_candidates:
    encontrado = None
    for p in mc["paths"]:
        if os.path.exists(p):
            encontrado = p
            break
    
    if not encontrado:
        print(f"[!] No encontrado checkpoint para: {mc['nombre']}")
        continue

    print(f"[OK] Cargando {mc['nombre']} desde: {encontrado}")
    ckpt = torch.load(encontrado, map_location=device, weights_only=True)
    state = ckpt.get('modelo_state_dict', ckpt)
    base_ch = ckpt.get('base_channels', 64)
    in_ch = mc["canales"]

    if mc["tipo"] == "unet":
        m = UNet(in_channels=in_ch, out_channels=1, base_channels=base_ch).to(device)
    elif mc["tipo"] == "transformer":
        m = UNetTransformer(in_channels=in_ch, out_channels=1, base_channels=base_ch).to(device)
    elif mc["tipo"] == "salar":
        t_layers = 4 if any('transformer_encoder.layers.3' in k for k in state.keys()) else 2
        t_heads = 16 if any('transformer_encoder.layers.0.linear1.weight' in k and state[k].shape[0] >= 4096 for k in state.keys()) else 8
        t_ff = 4096 if t_heads == 16 else 2048
        m = SalarUNet(in_channels=in_ch, out_channels=1, base_channels=base_ch,
                      transformer_layers=t_layers, transformer_heads=t_heads, transformer_ff_dim=t_ff).to(device)
    
    m.load_state_dict(state)
    m.eval()
    modelos_cargados.append({
        "nombre": mc["nombre"],
        "modelo": m,
        "canales": in_ch,
        "color": mc["color"]
    })

print(f"\n[INFO] Total de modelos listos para inferencia: {len(modelos_cargados)}")

# 2. Los 4 cuadrantes con MEJOR contraste y MENOS nubes de San Juan de Salinas (Puno)
cuadrantes = [
    ("00001_S2C_20250629_p00", "29-Jun-2025 (Núcleo de Salar — Estiaje Seco Puno, 0% nubes)"),
    ("00000_S2B_20250515_p00", "15-May-2025 (Transición — Cubeta Salina Despejada)"),
    ("00003_S2B_20250912_p00", "12-Sep-2025 (Sequía Extrema — Costra Salina Reluciente)"),
    ("00004_S2B_20250515_p00_e01", "15-May-2025 (Borde Costero — Salmuera vs. Ribera)")
]

def metricas(p, g):
    inter = np.logical_and(p, g).sum()
    union = np.logical_or(p, g).sum()
    iou = (inter + 1e-7) / (union + 1e-7)
    dice = (2.0 * inter + 1e-7) / (p.sum() + g.sum() + 1e-7)
    return float(iou), float(dice)

print("\n" + "="*70)
print(" GENERANDO PREDICCIONES COMPARATIVAS — SAN JUAN DE SALINAS (PUNO)")
print("="*70)

for idx, (bname, desc) in enumerate(cuadrantes, 1):
    tif_p = os.path.join(DATA_DIR, "imagen", f"{bname}.tif")
    mask_p = os.path.join(DATA_DIR, "mascara", f"{bname}.png")
    
    if not os.path.exists(tif_p) or not os.path.exists(mask_p):
        print(f"[!] No se encontró {bname}, saltando...")
        continue
    
    with rasterio.open(tif_p) as src:
        data_6b = src.read().astype(np.float32) # (6, 256, 256)
    
    gt = np.array(Image.open(mask_p)).astype(np.float32)
    if gt.max() > 1.0:
        gt = (gt > 127).astype(np.float32)
    
    # Composición RGB Realce Visual (B04 Rojo, B03 Verde, B02 Azul)
    rgb = np.stack([data_6b[2], data_6b[1], data_6b[0]], axis=-1)
    rgb_vis = np.clip(rgb * 3.5, 0.0, 1.0)
    
    # Inferencia con cada modelo
    preds_info = []
    for mc in modelos_cargados:
        m = mc["modelo"]
        c = mc["canales"]
        t_in = torch.from_numpy(data_6b[0:c]).unsqueeze(0).to(device)
        with torch.no_grad():
            logits = m(t_in)
            probs = torch.sigmoid(logits)
            p_bin = (probs[0, 0] > 0.5).cpu().numpy().astype(np.float32)
        iou_val, dice_val = metricas(p_bin, gt)
        preds_info.append({
            "nombre": mc["nombre"],
            "pred": p_bin,
            "iou": iou_val,
            "dice": dice_val,
            "color": mc["color"]
        })
    
    # =========================================================================
    # Gran Panel Comparativo Multi-Modelo (2 + N columnas)
    # =========================================================================
    n_cols = 2 + len(preds_info)
    fig_w = 4.2 * n_cols
    fig, axs = plt.subplots(1, n_cols, figsize=(fig_w, 4.8), dpi=220)
    fig.patch.set_facecolor('#FFFFFF')
    
    # Col 1: RGB
    axs[0].imshow(rgb_vis)
    axs[0].set_title(f"1. RGB Color Real\n{desc}", fontsize=9.5, fontweight='bold', color='#0F172A')
    axs[0].axis('off')
    
    # Col 2: Ground Truth
    axs[1].imshow(gt, cmap='gray', vmin=0, vmax=1)
    axs[1].set_title("2. Ground Truth\n(Anotación Curada)", fontsize=9.5, fontweight='bold', color='#0F172A')
    axs[1].axis('off')
    
    # Cols 3 en adelante: Modelos
    for c_i, p_inf in enumerate(preds_info, start=2):
        axs[c_i].imshow(p_inf["pred"], cmap='gray', vmin=0, vmax=1)
        axs[c_i].set_title(f"{c_i+1}. {p_inf['nombre']}\nIoU: {p_inf['iou']*100:.2f}% | Dice: {p_inf['dice']*100:.2f}%",
                           fontsize=9.5, fontweight='bold', color=p_inf['color'])
        axs[c_i].axis('off')
    
    plt.suptitle(f"Evaluación Comparativa Multimodelo — San Juan de Salinas (Puno) | Muestra {idx:02d}",
                 fontsize=12.5, fontweight='bold', y=0.98)
    plt.tight_layout()
    
    out_panel = os.path.join(OUTPUT_DIR, f"panel_sanjuan_salinas_muestra_{idx:02d}.png")
    plt.savefig(out_panel, bbox_inches='tight')
    plt.close()
    
    print(f"\n[OK] Panel guardado: {os.path.basename(out_panel)}")
    for p_inf in preds_info:
        print(f"     -> {p_inf['nombre']:28s}: IoU = {p_inf['iou']*100:6.2f}% | Dice = {p_inf['dice']*100:6.2f}%")

print("\n" + "="*70)
print(f"[ÉXITO] Todos los paneles comparativos guardados en:\n  {OUTPUT_DIR}")
print("="*70)
