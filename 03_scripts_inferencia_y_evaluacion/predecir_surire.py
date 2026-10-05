import os
import sys
import argparse
import rasterio
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch

# ==============================================================================
# SCRIPT DE INFERENCIA POR MODELO INDIVIDUAL — SALAR DE SURIRE (PUTRE, CHILE)
# ==============================================================================
# Evalúa de forma individual y detallada los 4 modelos de la investigación:
#   1. U-Net Multiespectral Clásico (350 épocas)
#   2. UNET-R Transformer Multiespectral (350 épocas)
#   3. Salar-UNet 4-Bandas Large 82M (500 épocas)
#   4. Salar-UNet Multiespectral Large 82M SOTA (500 épocas)
#
# Para cada modelo genera un panel de 3 columnas:
#   [1. Imagen Real RGB] | [2. Máscara Manual (Ground Truth)] | [3. Predicción del Modelo]
# ==============================================================================

# 1. Rutas de búsqueda automáticas para código compartido
rutas_codigo = [
    "/data/mmedina/paquete_salar_unet/codigo_compartido",
    "/data/mmedina/paquete_replicas_servidor/codigo_compartido",
    "/data/mmedina/datos_multiespectral/modelo_multiespectral",
    "/data/mmedina/paquete_salar_unet",
    "/data/mmedina/paquete_replicas_servidor",
    r"c:\Users\Mathias\Desktop\TESIS2\paquete_replicas_servidor\codigo_compartido",
    r"c:\Users\Mathias\Desktop\TESIS2\exportar_30_09_2026\02_CODIGO_ARQUITECTURAS_NUEVAS"
]
for p in rutas_codigo:
    if os.path.exists(p) and p not in sys.path:
        sys.path.insert(0, p)

from unet_salar import SalarUNet
from unet import UNet
from unet_transformer import UNetTransformer

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"[INFO] Dispositivo de cómputo para inferencia: {device}")

# 2. Rutas de búsqueda para el dataset de Surire
rutas_datos_candidatas = [
    "/data/mmedina/datos_multiespectral/DATASET/DATASET_MULTIESPECTRAL/SALAR DE SURIRE-PUTRE-CHILE",
    "/data/mmedina/DATASET/DATASET_MULTIESPECTRAL/SALAR DE SURIRE-PUTRE-CHILE",
    "/data/mmedina/DATASET_MULTIESPECTRAL/SALAR DE SURIRE-PUTRE-CHILE",
    r"c:\Users\Mathias\Desktop\TESIS2\DATASET\DATASET_MULTIESPECTRAL\SALAR DE SURIRE-PUTRE-CHILE"
]
DATA_DIR = None
for d in rutas_datos_candidatas:
    if os.path.exists(d):
        DATA_DIR = d
        break

if not DATA_DIR:
    raise FileNotFoundError("No se encontró el directorio de Salar de Surire en ninguna de las rutas esperadas.")

print(f"[OK] Directorio de datos: {DATA_DIR}")

OUTPUT_DIR = "/data/mmedina/predicciones_surire"
if not os.path.exists("/data/mmedina"):
    OUTPUT_DIR = r"c:\Users\Mathias\Desktop\TESIS2\PREDICCIONES_SURIRE"
os.makedirs(OUTPUT_DIR, exist_ok=True)
print(f"[OK] Directorio de salida para paneles: {OUTPUT_DIR}")

# 3. Modelos a evaluar con sus rutas de checkpoints
model_candidates = [
    {
        "id": "01_unet_multiespectral",
        "nombre": "U-Net Multiespectral Clásico",
        "familia": "CNN Baseline 6B (350 ep)",
        "tipo": "unet",
        "canales": 6,
        "color": "#0284C7", # Sky blue
        "paths": [
            "/data/mmedina/datos_multiespectral/modelo_multiespectral/checkpoints/best_model.pth",
            "/data/mmedina/paquete_replicas_servidor/unet_multiespectral/resultados/RUN_3/350_epocas/checkpoints/best_model.pth",
            "/data/mmedina/paquete_replicas_servidor/unet_multiespectral/resultados/350_epocas/checkpoints/best_model.pth",
            "/data/mmedina/paquete_replicas_servidor/unet_multiespectral/resultados/RUN_2/350_epocas/checkpoints/best_model.pth",
            r"c:\Users\Mathias\Desktop\TESIS2\modelo_multiespectral\checkpoints\best_model.pth"
        ]
    },
    {
        "id": "02_unetr_transformer",
        "nombre": "UNET-R Transformer Multiespectral",
        "familia": "Transformer Baseline 6B (350 ep)",
        "tipo": "transformer",
        "canales": 6,
        "color": "#7C3AED", # Purple
        "paths": [
            "/data/mmedina/paquete_replicas_servidor/transformer_multiespectral/resultados/RUN_3/350_epocas/checkpoints/best_model.pth",
            "/data/mmedina/paquete_replicas_servidor/transformer_multiespectral/resultados/350_epocas/checkpoints/best_model.pth",
            "/data/mmedina/paquete_replicas_servidor/transformer_multiespectral/resultados/RUN_2/350_epocas/checkpoints/best_model.pth"
        ]
    },
    {
        "id": "03_salar_unet_4bandas_82M",
        "nombre": "Salar-UNet 4-Bandas Large (82M)",
        "familia": "Arquitectura Híbrida 4B (500 ep)",
        "tipo": "salar",
        "canales": 4,
        "color": "#2563EB", # Royal blue
        "paths": [
            "/data/mmedina/paquete_salar_unet/salar_4bandas/resultados/500_epocas_large/checkpoints/best_model.pth",
            "/data/mmedina/paquete_salar_unet/salar_4bandas/resultados/500_epocas_large/checkpoints/checkpoint_epoch_500.pth",
            r"c:\Users\Mathias\Desktop\TESIS2\RESULTADOS_SALAR_4BANDAS_500EP\checkpoints\best_model.pth"
        ]
    },
    {
        "id": "04_salar_unet_multi_82M_SOTA",
        "nombre": "Salar-UNet Multiespectral Large (82M) SOTA",
        "familia": "Arquitectura Híbrida 6B (500 ep)",
        "tipo": "salar",
        "canales": 6,
        "color": "#16A34A", # Green SOTA
        "paths": [
            "/data/mmedina/paquete_salar_unet/salar_multiespectral/resultados/500_epocas_large/checkpoints/best_model.pth",
            "/data/mmedina/paquete_salar_unet/salar_multiespectral/resultados/500_epocas_large/checkpoints/checkpoint_epoch_500.pth"
        ]
    }
]

# 4. Carga robusta de pesos
modelos_listos = []
for mc in model_candidates:
    encontrado = None
    for p in mc["paths"]:
        if os.path.exists(p):
            encontrado = p
            break
    
    if not encontrado:
        print(f"[ADVERTENCIA] No se encontró checkpoint para: {mc['nombre']}")
        continue

    print(f"\n[OK] Cargando modelo: {mc['nombre']}")
    print(f"     Desde: {encontrado}")
    ckpt = torch.load(encontrado, map_location=device, weights_only=True)
    state = ckpt.get('modelo_state_dict', ckpt)
    base_ch = ckpt.get('base_channels', 64)
    in_ch = mc["canales"]

    if mc["tipo"] == "unet":
        m = UNet(in_channels=in_ch, out_channels=1, base_channels=base_ch).to(device)
    elif mc["tipo"] == "transformer":
        t_layers = 4 if any('transformer_encoder.layers.3' in k for k in state.keys()) else 2
        t_heads = 16 if any('transformer_encoder.layers.0.linear1.weight' in k and state[k].shape[0] >= 4096 for k in state.keys()) else 8
        t_ff = 4096 if t_heads == 16 else 2048
        m = UNetTransformer(in_channels=in_ch, out_channels=1, base_channels=base_ch,
                            transformer_layers=t_layers, transformer_heads=t_heads, transformer_ff_dim=t_ff).to(device)
    elif mc["tipo"] == "salar":
        t_layers = 4 if any('transformer_encoder.layers.3' in k for k in state.keys()) else 2
        t_heads = 16 if any('transformer_encoder.layers.0.linear1.weight' in k and state[k].shape[0] >= 4096 for k in state.keys()) else 8
        t_ff = 4096 if t_heads == 16 else 2048
        m = SalarUNet(in_channels=in_ch, out_channels=1, base_channels=base_ch,
                      transformer_layers=t_layers, transformer_heads=t_heads, transformer_ff_dim=t_ff).to(device)

    m.load_state_dict(state)
    m.eval()
    modelos_listos.append({
        "id": mc["id"],
        "nombre": mc["nombre"],
        "familia": mc["familia"],
        "modelo": m,
        "canales": in_ch,
        "color": mc["color"],
        "checkpoint": encontrado
    })

print(f"\n[INFO] Total de modelos disponibles para inferencia: {len(modelos_listos)}")

# 5. Métricas de evaluación
def calcular_metricas(pred, gt):
    p = pred.astype(bool)
    g = gt.astype(bool)
    tp = np.logical_and(p, g).sum()
    fp = np.logical_and(p, np.logical_not(g)).sum()
    fn = np.logical_and(np.logical_not(p), g).sum()
    tn = np.logical_and(np.logical_not(p), np.logical_not(g)).sum()

    inter = tp
    union = tp + fp + fn
    iou = (inter + 1e-7) / (union + 1e-7)
    dice = (2.0 * inter + 1e-7) / (p.sum() + g.sum() + 1e-7)
    precision = (tp + 1e-7) / (tp + fp + 1e-7)
    recall = (tp + 1e-7) / (tp + fn + 1e-7)
    accuracy = (tp + tn + 1e-7) / (tp + tn + fp + fn + 1e-7)

    return {
        "iou": float(iou),
        "dice": float(dice),
        "precision": float(precision),
        "recall": float(recall),
        "accuracy": float(accuracy)
    }

# 6. Parches de evaluación (Foco principal: p00 sin nubes en Putre, Chile)
parches_a_evaluar = [
    {
        "id": "00000_S2B_20251022_p00",
        "descripcion": "22-Oct-2025 | Borde Noroeste del Salar de Surire (0% Nubes, Estiaje Seco)",
        "es_principal": True
    },
    {
        "id": "00001_S2B_20251022_p01",
        "descripcion": "22-Oct-2025 | Transición Bofedal / Costra Salina",
        "es_principal": False
    },
    {
        "id": "00003_S2B_20251022_p03",
        "descripcion": "22-Oct-2025 | Núcleo Central de Evaporita Reluciente (Test Set)",
        "es_principal": False
    }
]

print("\n" + "="*80)
print(" INICIANDO GENERACIÓN DE PANELES INDIVIDUALES — SALAR DE SURIRE (CHILE)")
print("="*80)

resumen_lineas_txt = []
resumen_lineas_txt.append("="*80)
resumen_lineas_txt.append("RESUMEN DE INFERENCIA MULTI-MODELO — SALAR DE SURIRE (PUTRE, CHILE)")
resumen_lineas_txt.append("="*80 + "\n")

for p_info in parches_a_evaluar:
    bname = p_info["id"]
    desc = p_info["descripcion"]
    tif_p = os.path.join(DATA_DIR, "imagen", f"{bname}.tif")
    mask_p = os.path.join(DATA_DIR, "mascara", f"{bname}.png")

    if not os.path.exists(tif_p) or not os.path.exists(mask_p):
        print(f"[!] No se encontró parche {bname}, saltando...")
        continue

    print(f"\n>>> Procesando Parche: {bname} ({desc})")

    # Lectura de datos multiespectrales (Float32)
    with rasterio.open(tif_p) as src:
        data_6b = src.read().astype(np.float32)

    # Lectura de Ground Truth binario
    gt = np.array(Image.open(mask_p)).astype(np.float32)
    if gt.max() > 1.0:
        gt = (gt > 127).astype(np.float32)

    # Extracción de RGB Realce Natural (B04 Rojo, B03 Verde, B02 Azul)
    rgb = np.stack([data_6b[2], data_6b[1], data_6b[0]], axis=-1)
    p1, p99 = np.percentile(rgb, (1, 99))
    if p99 > p1:
        rgb_vis = np.clip((rgb - p1) / (p99 - p1), 0.0, 1.0)
    else:
        rgb_vis = np.clip(rgb * 3.5, 0.0, 1.0)

    parche_resultados = []

    # --------------------------------------------------------------------------
    # A. Inferencia y Generación de Panel Individual (3 Columnas) por Modelo
    # --------------------------------------------------------------------------
    for m_idx, ml in enumerate(modelos_listos, 1):
        m = ml["modelo"]
        c = ml["canales"]

        t_in = torch.from_numpy(data_6b[0:c]).unsqueeze(0).to(device)
        with torch.no_grad():
            logits = m(t_in)
            probs = torch.sigmoid(logits)
            p_bin = (probs[0, 0] > 0.5).cpu().numpy().astype(np.float32)

        mets = calcular_metricas(p_bin, gt)
        parche_resultados.append({
            "modelo_info": ml,
            "pred": p_bin,
            "metricas": mets
        })

        # --- Crear Panel Individual de 3 Columnas ---
        fig, axes = plt.subplots(1, 3, figsize=(15.5, 5.2), dpi=300)
        fig.patch.set_facecolor('#FFFFFF')

        # Columna 1: Imagen Satelital Real (RGB)
        axes[0].imshow(rgb_vis)
        axes[0].set_title("1. Imagen Satelital Real (RGB Color Natural)\nSentinel-2 L2A | Bandas B04-B03-B02 (10m)",
                          fontsize=10.5, fontweight='bold', color='#0F172A', pad=10)
        axes[0].axis('off')

        # Columna 2: Máscara Manual (Ground Truth)
        axes[1].imshow(gt, cmap='gray', vmin=0, vmax=1)
        axes[1].set_title("2. Máscara Manual (Ground Truth)\nAnotación Curada de Costra Salina",
                          fontsize=10.5, fontweight='bold', color='#0F172A', pad=10)
        axes[1].axis('off')

        # Columna 3: Predicción del Modelo
        axes[2].imshow(p_bin, cmap='gray', vmin=0, vmax=1)
        axes[2].set_title(f"3. Predicción del Modelo: {ml['nombre']}\nIoU: {mets['iou']*100:.2f}% | Dice (F1): {mets['dice']*100:.2f}%",
                          fontsize=10.5, fontweight='bold', color=ml['color'], pad=10)
        axes[2].axis('off')

        # Encabezado principal superior
        plt.suptitle(
            f"Evaluación de Inferencia Individual — {ml['nombre']}\n"
            f"Zona de Estudio: Salar de Surire (Putre, Chile) | {desc}",
            fontsize=12, fontweight='bold', color='#0F172A', y=0.98
        )

        # Barra inferior de métricas en caja formal
        info_caja = (
            f"Métricas Oficiales del Parche:   "
            f"IoU (Jaccard): {mets['iou']*100:.2f}%   |   "
            f"Dice (F1-Score): {mets['dice']*100:.2f}%   |   "
            f"Precisión: {mets['precision']*100:.2f}%   |   "
            f"Recall: {mets['recall']*100:.2f}%   |   "
            f"Accuracy: {mets['accuracy']*100:.2f}%"
        )
        fig.text(0.5, 0.04, info_caja, ha='center', fontsize=9.5, fontweight='semibold',
                 color='#1E293B', bbox=dict(boxstyle='round,pad=0.5', facecolor='#F1F5F9', edgecolor='#CBD5E1', lw=1.2))

        plt.subplots_adjust(top=0.82, bottom=0.14, left=0.03, right=0.97, wspace=0.12)

        # Nombre de archivo limpio
        out_individual = os.path.join(OUTPUT_DIR, f"panel_individual_{ml['id']}_{bname}.png")
        plt.savefig(out_individual, bbox_inches='tight')
        plt.close()

        print(f"  [OK] Guardado: {os.path.basename(out_individual)}")
        print(f"       IoU: {mets['iou']*100:6.2f}% | Dice: {mets['dice']*100:6.2f}% | Prec: {mets['precision']*100:6.2f}% | Rec: {mets['recall']*100:6.2f}%")

    # --------------------------------------------------------------------------
    # B. Panel Comparativo Global de 4 Modelos (4 Filas x 4 Columnas)
    # --------------------------------------------------------------------------
    n_mods = len(parche_resultados)
    fig_comp, axs_comp = plt.subplots(n_mods, 4, figsize=(16, 3.8 * n_mods), dpi=250)
    fig_comp.patch.set_facecolor('#FFFFFF')

    for r_i, pres in enumerate(parche_resultados):
        ml_i = pres["modelo_info"]
        m_pred = pres["pred"]
        m_mets = pres["metricas"]

        # Mapa de discrepancia / error:
        # Verde = Verdadero Positivo, Rojo = Falso Positivo, Azul = Falso Negativo, Blanco = Fondo Correcto
        h, w = gt.shape
        err_map = np.ones((h, w, 3), dtype=np.float32) # Blanco base
        tp_m = np.logical_and(m_pred == 1, gt == 1)
        fp_m = np.logical_and(m_pred == 1, gt == 0)
        fn_m = np.logical_and(m_pred == 0, gt == 1)
        
        err_map[tp_m] = [0.13, 0.65, 0.28] # Verde
        err_map[fp_m] = [0.88, 0.15, 0.15] # Rojo
        err_map[fn_m] = [0.15, 0.45, 0.88] # Azul

        # Col 1: RGB
        axs_comp[r_i, 0].imshow(rgb_vis)
        axs_comp[r_i, 0].axis('off')
        if r_i == 0:
            axs_comp[r_i, 0].set_title("1. RGB Real (Sentinel-2)", fontsize=11, fontweight='bold', pad=8)
        axs_comp[r_i, 0].text(-0.08, 0.5, f"{ml_i['nombre']}\n({ml_i['familia']})",
                              transform=axs_comp[r_i, 0].transAxes,
                              fontsize=10, fontweight='bold', color=ml_i['color'],
                              va='center', ha='right', rotation=0)

        # Col 2: GT
        axs_comp[r_i, 1].imshow(gt, cmap='gray', vmin=0, vmax=1)
        axs_comp[r_i, 1].axis('off')
        if r_i == 0:
            axs_comp[r_i, 1].set_title("2. Ground Truth", fontsize=11, fontweight='bold', pad=8)

        # Col 3: Predicción
        axs_comp[r_i, 2].imshow(m_pred, cmap='gray', vmin=0, vmax=1)
        axs_comp[r_i, 2].axis('off')
        axs_comp[r_i, 2].set_title(f"IoU: {m_mets['iou']*100:.2f}% | Dice: {m_mets['dice']*100:.2f}%",
                                  fontsize=10, fontweight='bold', color=ml_i['color'], pad=6)
        if r_i == 0:
            axs_comp[r_i, 2].text(0.5, 1.25, "3. Predicción del Modelo",
                                 transform=axs_comp[r_i, 2].transAxes,
                                 fontsize=11, fontweight='bold', ha='center')

        # Col 4: Mapa de Error
        axs_comp[r_i, 4-1].imshow(err_map)
        axs_comp[r_i, 4-1].axis('off')
        if r_i == 0:
            axs_comp[r_i, 3].set_title("4. Diagnóstico de Error\n(Verde=TP, Rojo=FP, Azul=FN)", fontsize=10, fontweight='bold', pad=8)

    plt.suptitle(
        f"Matriz Comparativa Multi-Modelo — Salar de Surire (Putre, Chile)\n{desc}",
        fontsize=13, fontweight='bold', y=0.99
    )
    plt.tight_layout()
    out_comp = os.path.join(OUTPUT_DIR, f"comparativa_global_4modelos_{bname}.png")
    plt.savefig(out_comp, bbox_inches='tight')
    plt.close()
    print(f"  [OK] Matriz comparativa guardada: {os.path.basename(out_comp)}")

    # Registro en archivo de texto
    resumen_lineas_txt.append(f"\nParche: {bname} — {desc}")
    resumen_lineas_txt.append("-" * 75)
    resumen_lineas_txt.append(f"{'Modelo':<45} | {'IoU (%)':<9} | {'Dice (%)':<9} | {'Prec (%)':<9} | {'Rec (%)':<9}")
    resumen_lineas_txt.append("-" * 75)
    for pres in parche_resultados:
        ml_i = pres["modelo_info"]
        m_mets = pres["metricas"]
        resumen_lineas_txt.append(
            f"{ml_i['nombre']:<45} | {m_mets['iou']*100:>7.2f}% | {m_mets['dice']*100:>7.2f}% | "
            f"{m_mets['precision']*100:>7.2f}% | {m_mets['recall']*100:>7.2f}%"
        )

# Guardar reporte de texto
txt_out = os.path.join(OUTPUT_DIR, "metricas_inferencia_surire.txt")
with open(txt_out, 'w', encoding='utf-8') as f:
    f.write("\n".join(resumen_lineas_txt) + "\n")

print("\n" + "="*80)
print(f"[ÉXITO TOTAL] Todas las predicciones y paneles se han generado en:")
print(f"  >> {OUTPUT_DIR}")
print(f"  >> Reporte numérico: {txt_out}")
print("="*80)
