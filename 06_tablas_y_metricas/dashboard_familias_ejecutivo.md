# TABLERO SINÓPTICO EJECUTIVO: COMPARATIVA DE FAMILIAS DE MODELOS

**Proyecto de Tesis:** Automatización de la delimitación y segmentación semántica multitemporal de salares andinos mediante Sentinel-2 L2A  
**Autor:** Mathias Aldair Medina Vivanco | **Asesor:** Mag. Ferdinand Edgardo Pineda Ancco  
**Laboratorio:** Inteligencia Artificial (IA-PUCP) — Clúster GPU Servidor Phantom

---

| Familia Tecnológica | Canales de Entrada | Arquitectura Representativa | mIoU Techo (Test Set) | Dice Score (F1) | Diagnóstico Clínico y Comportamiento Espacial |
|:---|:---:|:---|:---:|:---:|:---|
| **1. Línea Base Visible (RGB)** | 3 Bandas (B04, B03, B02) | U-Net Convolucional / UNET-R (350 ep) | 73.43% – 75.32% | 84.28% – 84.68% | **Barrera óptica:** Saturación radiométrica en sales blancas y ~25% de falsos positivos en laderas de cerros y arenas claras por similitud espectral en el espectro visible. |
| **2. Línea Base Infrarroja (NIR)** | 4 Bandas (RGB + NIR B8) | U-Net Clásico E07 (350 ep) | 93.65% ± 0.32% | 96.69% ± 0.18% | **Salto cuántico (+20.54% IoU):** Absorción electromagnética radical del agua en 842 nm. Altísima precisión en bordes costeros lacustres, aunque con ligeros falsos positivos en sombras orográficas. |
| **3. Transformer Espacial** | 4 Bandas (RGB + NIR B8) | UNET-R E08 (350 ep) | 91.11% ± 0.25% | 95.35% ± 0.13% | **Autoatención global:** Elimina casi totalmente las confusiones en montañas al comprender el contexto geográfico macroscópico, pero pierde nitidez en canales estrechos y meandros de orilla. |
| **4. Propuesta Propia 4B (Salar-UNet)** | 4 Bandas (RGB + NIR B8) | Salar-UNet Base 48M (500 ep) | 92.68% | 96.20% | **Resolución del dilema cerros vs. orillas:** Combina *Attention Gates* en los *skip connections*, cuello Transformer y supervisión auxiliar de bordes Sobel. Cero falsos positivos en montañas con bordes nítidos. |
| **5. Multiespectral Clásico** | 6 Bandas (RGB+NIR+NDVI+SI) | U-Net / UNET-R Multi (350 ep) | 99.15% – 99.26% | 99.57% – 99.63% | **Fusión biofísica:** Integración de índices precalculados de vegetación (NDVI) y salinidad (SI). Segmentación casi exacta a través de toda la serie temporal 2016-2026. |
| **6. Propuesta Propia SOTA (Salar-UNet Multi)** | 6 Bandas (RGB+NIR+NDVI+SI) | Salar-UNet Large 82M (500 ep) | **99.31% (SOTA)** | **99.65%** | **MÁXIMO SOTA DE LA TESIS:** Test Loss 0.0183, Recall 99.73%, Precisión 99.58% alcanzado en 240.54 min de GPU en clúster Phantom. Delimitación perfecta de salmuera y bofedales. |
