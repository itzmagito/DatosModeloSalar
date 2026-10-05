# TABLA CONSOLIDADA DE RESULTADOS: 3 RÉPLICAS INDEPENDIENTES (MEDIA ± DESVIACIÓN ESTÁNDAR)

**Proyecto de Tesis:** Automatización de la detección de salares andinos con Sentinel-2
**Autor:** Mathias Aldair Medina Vivanco | **Asesor:** Mag. Ferdinand Edgardo Pineda Ancco
**Condición Experimental:** 3 Réplicas independientes (Run 1: Semilla base, Run 2: Semilla 101, Run 3: Semilla 202)

| ID | Dominio Espectral | Arquitectura | Épocas | Réplicas | IoU Promedio (Jaccard) | Dice Score (F1) | Test Loss | Precisión | Recall |
|:---:|:---|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| E01 | RGB | U-Net Clásico | 150 | 3/3 | **71.81 ± 1.65%** | 81.88 ± 0.18% | 0.6606 ± 0.0107 | 72.15% | 95.45% |
| E02 | RGB | U-Net Clásico | 200 | 3/3 | **71.44 ± 2.49%** | 81.51 ± 0.38% | 0.6835 ± 0.0246 | 71.41% | 95.79% |
| E03 | RGB | U-Net Clásico | 350 | 3/3 | **75.32 ± 1.64%** | 84.28 ± 0.35% | 0.5621 ± 0.0099 | 74.92% | 96.83% |
| E04 | RGB | UNET-R (Transformer) | 350 | 3/3 | **73.63 ± 1.30%** | 83.69 ± 0.05% | 0.6153 ± 0.0061 | 76.01% | 93.45% |
| E05 | 4-BANDAS | U-Net Clásico | 150 | 3/3 | **90.19 ± 0.23%** | 94.77 ± 0.14% | 0.1975 ± 0.0050 | 94.66% | 94.90% |
| E06 | 4-BANDAS | U-Net Clásico | 200 | 3/3 | **90.27 ± 0.32%** | 94.82 ± 0.21% | 0.1959 ± 0.0074 | 94.79% | 94.85% |
| E07 | 4-BANDAS | U-Net Clásico | 350 | 3/3 | **93.65 ± 0.32%** | 96.69 ± 0.18% | 0.1273 ± 0.0065 | 96.72% | 96.67% |
| E08 | 4-BANDAS | UNET-R (Transformer) | 350 | 3/3 | **91.11 ± 0.25%** | 95.35 ± 0.13% | 0.1740 ± 0.0036 | 95.49% | 95.21% |
| E09 | MULTIESPECTRAL | U-Net Clásico | 150 | 3/3 | **98.66 ± 0.02%** | 99.32 ± 0.01% | 0.0292 ± 0.0006 | 99.35% | 99.30% |
| E10 | MULTIESPECTRAL | U-Net Clásico | 200 | 3/3 | **98.79 ± 0.15%** | 99.39 ± 0.07% | 0.0281 ± 0.0021 | 99.48% | 99.30% |
| E11 | MULTIESPECTRAL | U-Net Clásico | 350 | 3/3 | **99.23 ± 0.07%** | 99.61 ± 0.04% | 0.0177 ± 0.0019 | 99.63% | 99.60% |
| E12 | MULTIESPECTRAL | UNET-R (Transformer) | 350 | 3/3 | **99.26 ± 0.02%** | 99.63 ± 0.01% | 0.0169 ± 0.0003 | 99.64% | 99.62% |
