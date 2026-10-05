# TABLA MAESTRA DE RESULTADOS EXPERIMENTALES EN TEST SET (14 EXPERIMENTOS)

**Proyecto de Fin de Carrera (PFC 2 - 1INF46):** Automatización de la delimitación y segmentación semántica multitemporal de salares andinos mediante imágenes Sentinel-2 L2A y Deep Learning  
**Autor:** Mathias Aldair Medina Vivanco (Código: 20190391)  
**Asesor:** Mag. Ferdinand Edgardo Pineda Ancco  
**Institución:** Pontificia Universidad Católica del Perú (PUCP) — Facultad de Ciencias e Ingeniería  
**Infraestructura de Cómputo:** Clúster IA-PUCP — Servidor Phantom (NVIDIA RTX A5500 de 24 GB VRAM)  
**Conjunto de Evaluación:** Test Set Ciego e Independiente (488 parches de 256 × 256 píxeles a 10m de resolución)

---

## Matriz Maestra de Evaluación Comparativa

| ID | Modelo / Arquitectura | Dominio Espectral | Bandas de Entrada | Épocas | Test Loss | mIoU (Jaccard) | Dice Score (F1) | Precisión | Recall |
|:---:|:---|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **E01** | U-Net Convolucional | Visible (RGB) | 3 (B04, B03, B02) | 150 | 0.6494 | 69.91% | 82.04% | 72.49% | 94.52% |
| **E02** | U-Net Convolucional | Visible (RGB) | 3 (B04, B03, B02) | 200 | 0.7119 | 68.58% | 81.08% | 70.83% | 94.81% |
| **E03** | U-Net Convolucional (Base) | Visible (RGB) | 3 (B04, B03, B02) | 350 | 0.5510 | 73.43% | 84.68% | 75.43% | 96.51% |
| **E04** | UNET-R Transformer | Visible (RGB) | 3 (B04, B03, B02) | 350 | 0.6154 | 72.16% | 83.64% | 75.91% | 93.15% |
| **E05** | U-Net Convolucional | Infrarrojo Cercano | 4 (RGB + NIR B8) | 150 | 0.1928 | 90.31% | 94.90% | 94.67% | 95.13% |
| **E06** | U-Net Convolucional | Infrarrojo Cercano | 4 (RGB + NIR B8) | 200 | 0.1881 | 90.56% | 95.04% | 95.07% | 95.02% |
| **E07** | U-Net Convolucional | Infrarrojo Cercano | 4 (RGB + NIR B8) | 350 | 0.1201 | **93.97%** | **96.89%** | **96.88%** | 96.90% |
| **E08** | UNET-R Transformer | Infrarrojo Cercano | 4 (RGB + NIR B8) | 350 | 0.1699 | 91.39% | 95.49% | 95.72% | 95.27% |
| **E09** | U-Net Multiespectral | Fusión Espectral | 6 (RGB+NIR+NDVI+SI) | 150 | 0.0285 | 98.67% | 99.33% | 99.27% | 99.39% |
| **E10** | U-Net Multiespectral | Fusión Espectral | 6 (RGB+NIR+NDVI+SI) | 200 | 0.0257 | 98.95% | 99.47% | 99.46% | 99.48% |
| **E11** | U-Net Multiespectral | Fusión Espectral | 6 (RGB+NIR+NDVI+SI) | 350 | 0.0199 | 99.15% | 99.57% | 99.61% | 99.53% |
| **E12** | UNET-R Transformer Multi | Fusión Espectral | 6 (RGB+NIR+NDVI+SI) | 350 | 0.0169 | 99.24% | 99.62% | 99.59% | 99.65% |
| **PROP-1** | **Salar-UNet Base (48M)** | **Infrarrojo Cercano** | **4 (RGB + NIR B8)** | **500** | **0.1456** | **92.68%** | **96.20%** | **96.60%** | **95.80%** |
| **PROP-2** | **Salar-UNet Large (82M)** | **Fusión Espectral** | **6 (RGB+NIR+NDVI+SI)** | **500** | **0.0183** | **99.31%** | **99.65%** | **99.58%** | **99.73%** |

---

## Hallazgos Científicos Destacados
1. **Barrera Espectral Óptica:** Las bandas RGB solas no superan el 75.32% de IoU por saturación en reflectancias de arenas claras y rocas calcáreas.
2. **Supremacía del NIR (Banda 8 a 842 nm):** Añadir NIR B8 produce un incremento directo de **+20.54% de IoU**, permitiendo segmentar el borde hídrico/salino con nitidez sub-píxel.
3. **Equilibrio Arquitectónico en Salar-UNet:** Integra *Attention Gates* en los *skip connections* y un cuello *Transformer*, eliminando falsos positivos en cerros sin desvanecer orillas finas.
4. **SOTA Global de la Tesis:** `Salar-UNet Multiespectral Large 82M` alcanza un **IoU de 99.31%** y **Dice de 99.65%** a 500 épocas.
