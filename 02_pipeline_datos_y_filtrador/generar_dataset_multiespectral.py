"""
==============================================================================
GENERADOR DE DATASET MULTIESPECTRAL (6 CANALES) - TESIS SALARES
==============================================================================
Genera parches multicanal GeoTIFF (.tif) de 6 bandas a partir de los datos
crudos Sentinel-2 (.SAFE en DATOS/), reutilizando las coordenadas exactas de
los 3,255 parches de DATASET PROCESADO/.

Variables de salida (Float32, 256x256):
  Banda 1: B02 (Azul - 10m)
  Banda 2: B03 (Verde - 10m)
  Banda 3: B04 (Rojo - 10m)
  Banda 4: B08 (NIR - 10m)
  Banda 5: NDVI (Índice de Vegetación / Bofedales)
  Banda 6: SI (Índice de Salinidad / Costras Salinas)

Estructura de salida:
  DATASET/DATASET_MULTIESPECTRAL/
    <salar>/
      imagen/<id>.tif   (6 bandas GeoTIFF float32, compresión DEFLATE)
      mascara/<id>.png  (Máscara binaria existente)
==============================================================================
"""

import os
import glob
import re
import shutil
import time
import rasterio
from rasterio.windows import from_bounds
from rasterio.enums import Resampling
import numpy as np

BASE_DIR = r"c:\Users\Mathias\Desktop\TESIS2"
PROCESADO_DIR = os.path.join(BASE_DIR, "DATASET", "DATASET PROCESADO")
BASE_DATOS_DIR = os.path.join(BASE_DIR, "DATASET")
OUTPUT_DIR = os.path.join(BASE_DIR, "DATASET", "DATASET_MULTIESPECTRAL")


def buscar_jp2(datos_dir, fecha, banda, res):
    """Busca el archivo JP2 correspondiente a la fecha y banda en DATOS/."""
    fecha_guion = f"{fecha[:4]}-{fecha[4:6]}-{fecha[6:]}"
    patron = f"*_{banda}_{res}.jp2"

    encontrados = glob.glob(os.path.join(datos_dir, f"*{fecha}*", "**", patron), recursive=True)
    if not encontrados:
        encontrados = glob.glob(os.path.join(datos_dir, f"*{fecha_guion}*", "**", patron), recursive=True)

    if not encontrados:
        raise FileNotFoundError(f"No se encontró {banda}_{res} para fecha {fecha} en {datos_dir}")
    return encontrados[0]


def procesar_todo():
    print("=" * 75)
    print(" GENERADOR DE DATASET MULTIESPECTRAL (6 CANALES)")
    print(f" Origen parches: {PROCESADO_DIR}")
    print(f" Destino:        {OUTPUT_DIR}")
    print(" Variables:      B02, B03, B04, B08, NDVI, SI")
    print("=" * 75)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    todos_salares = sorted([
        d for d in os.listdir(PROCESADO_DIR)
        if os.path.isdir(os.path.join(PROCESADO_DIR, d))
    ])

    total_parches_global = 0
    t0_global = time.time()

    for salar_idx, salar in enumerate(todos_salares, 1):
        salar_proc_img = os.path.join(PROCESADO_DIR, salar, "imagen")
        salar_proc_mask = os.path.join(PROCESADO_DIR, salar, "mascara")
        datos_salar_dir = os.path.join(BASE_DATOS_DIR, salar, "DATOS")

        if not os.path.exists(salar_proc_img):
            continue

        out_img_dir = os.path.join(OUTPUT_DIR, salar, "imagen")
        out_mask_dir = os.path.join(OUTPUT_DIR, salar, "mascara")
        os.makedirs(out_img_dir, exist_ok=True)
        os.makedirs(out_mask_dir, exist_ok=True)

        tifs = sorted(glob.glob(os.path.join(salar_proc_img, "*.tif")))
        if not tifs:
            continue

        print(f"\n[{salar_idx}/{len(todos_salares)}] Salar: {salar} ({len(tifs)} parches)")

        # Agrupar parches por fecha para optimizar apertura de archivos JP2
        parches_por_fecha = {}
        for p_tif in tifs:
            fname = os.path.basename(p_tif)
            m = re.search(r'_(S2[ABC])_(\d{8})_', fname)
            if m:
                fecha = m.group(2)
            else:
                fecha = "unknown"
            if fecha not in parches_por_fecha:
                parches_por_fecha[fecha] = []
            parches_por_fecha[fecha].append(p_tif)

        # Procesar por fecha
        for fecha, lista_parches in parches_por_fecha.items():
            try:
                jp2_b02 = buscar_jp2(datos_salar_dir, fecha, "B02", "10m")
                jp2_b03 = buscar_jp2(datos_salar_dir, fecha, "B03", "10m")
                jp2_b04 = buscar_jp2(datos_salar_dir, fecha, "B04", "10m")
                jp2_b08 = buscar_jp2(datos_salar_dir, fecha, "B08", "10m")
                jp2_b11 = buscar_jp2(datos_salar_dir, fecha, "B11", "20m")
                jp2_b12 = buscar_jp2(datos_salar_dir, fecha, "B12", "20m")
            except Exception as e:
                print(f"  [ERROR] Buscando JP2 para fecha {fecha}: {e}")
                continue

            # Abrir los 6 archivos JP2 una sola vez para todos los parches de esta fecha
            with rasterio.open(jp2_b02) as src_b02, \
                 rasterio.open(jp2_b03) as src_b03, \
                 rasterio.open(jp2_b04) as src_b04, \
                 rasterio.open(jp2_b08) as src_b08, \
                 rasterio.open(jp2_b11) as src_b11, \
                 rasterio.open(jp2_b12) as src_b12:

                for p_tif in lista_parches:
                    fname = os.path.basename(p_tif)
                    nombre_base = os.path.splitext(fname)[0]

                    # Leer metadatos del parche original
                    with rasterio.open(p_tif) as ref:
                        bounds = ref.bounds
                        crs = ref.crs
                        transform = ref.transform

                    # Extraer ventanas para cada banda
                    win_10m = from_bounds(bounds.left, bounds.bottom, bounds.right, bounds.top, src_b02.transform)
                    win_20m = from_bounds(bounds.left, bounds.bottom, bounds.right, bounds.top, src_b11.transform)

                    b02 = src_b02.read(1, window=win_10m, out_shape=(256, 256), resampling=Resampling.bilinear).astype('float32') / 10000.0
                    b03 = src_b03.read(1, window=win_10m, out_shape=(256, 256), resampling=Resampling.bilinear).astype('float32') / 10000.0
                    b04 = src_b04.read(1, window=win_10m, out_shape=(256, 256), resampling=Resampling.bilinear).astype('float32') / 10000.0
                    b08 = src_b08.read(1, window=win_10m, out_shape=(256, 256), resampling=Resampling.bilinear).astype('float32') / 10000.0

                    b11 = src_b11.read(1, window=win_20m, out_shape=(256, 256), resampling=Resampling.bilinear).astype('float32') / 10000.0
                    b12 = src_b12.read(1, window=win_20m, out_shape=(256, 256), resampling=Resampling.bilinear).astype('float32') / 10000.0

                    # Calcular NDVI y SI
                    ndvi = np.nan_to_num((b08 - b04) / (b08 + b04 + 1e-6), nan=0.0)
                    si = np.nan_to_num((b11 - b12) / (b11 + b12 + 1e-6), nan=0.0)

                    # Apilar en tensor de 6 canales (6, 256, 256)
                    stack = np.stack([b02, b03, b04, b08, ndvi, si], axis=0)

                    # Guardar GeoTIFF multicanal con compresión DEFLATE
                    out_tif = os.path.join(out_img_dir, f"{nombre_base}.tif")
                    profile = {
                        'driver': 'GTiff',
                        'dtype': 'float32',
                        'width': 256,
                        'height': 256,
                        'count': 6,
                        'crs': crs,
                        'transform': transform,
                        'compress': 'deflate'
                    }
                    with rasterio.open(out_tif, 'w', **profile) as dst:
                        for b_idx in range(6):
                            dst.write(stack[b_idx], b_idx + 1)

                    # Copiar máscara correspondiente (PNG)
                    mask_orig = os.path.join(salar_proc_mask, f"{nombre_base}.png")
                    mask_dest = os.path.join(out_mask_dir, f"{nombre_base}.png")
                    if os.path.exists(mask_orig):
                        shutil.copy2(mask_orig, mask_dest)

                    total_parches_global += 1

            print(f"  [OK] Fecha {fecha}: {len(lista_parches)} parches procesados a 6 canales.")

    dt_global = time.time() - t0_global
    print("\n" + "=" * 75)
    print(f" [OK] DATASET MULTIESPECTRAL CREADO EXITOSAMENTE")
    print(f" Total parches generados: {total_parches_global:,}")
    print(f" Tiempo total: {dt_global/60:.2f} minutos")
    print(f" Carpeta destino: {OUTPUT_DIR}")
    print("=" * 75)


if __name__ == "__main__":
    procesar_todo()
