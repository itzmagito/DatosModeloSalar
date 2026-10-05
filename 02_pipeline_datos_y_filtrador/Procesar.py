"""
==============================================================================
 GENERADOR DE DATASET - SALAR DE SURIRE-PUTRE-CHILE
 
 Formulas utilizadas:
   - Indice de Salinidad (SI): SI = (B11 - B12) / (B11 + B12)
     Donde B11 y B12 son bandas SWIR de Sentinel-2 a 20m de resolucion
   - Umbral de segmentacion: 0.0771 (validado visualmente)
   - Mascara binaria: SI > 0.0771 -> 255 (blanco=sal), 0 (negro=no-sal)
 
 Estructura esperada:
   SALAR DE NEGRITOS PIURA/
     REFERENCIA/      <- 1 .SAFE para elegir coordenadas
     DATOS/           <- .SAFEs organizados por fecha (YYYY-MM-DD/)
     Procesar.py      <- este script
     
   DATASET PROCESADO/ <- SALIDA CENTRALIZADA
     SALAR DE NEGRITOS PIURA/
       imagen/ + mascara/ (con PNG y TIF)
 
 Salida por cada parche:
   - imagen/XXXXX_S2A_20250604_p00.png   (RGB color real)
   - imagen/XXXXX_S2A_20250604_p00.tif   (RGB con georreferencia)
   - mascara/XXXXX_S2A_20250604_p00.png  (mascara binaria)
   - mascara/XXXXX_S2A_20250604_p00.tif  (mascara con georreferencia)
==============================================================================
"""

import os
import glob
import re
import numpy as np
import rasterio
from rasterio.transform import from_bounds
from PIL import Image

# =============================================================================
# CONFIGURACION
# =============================================================================
UMBRAL_SI = 0.0771
TAMANO_PARCHE = 256
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

RUTA_DATOS = os.path.join(SCRIPT_DIR, "DATOS")
RUTA_COORDENADAS = os.path.join(SCRIPT_DIR, "coordenadas_salar.txt")
# Carpeta de salida del dataset (Centralizada)
RUTA_DATASET = os.path.join(SCRIPT_DIR, "..", "DATASET PROCESADO", "SALAR DE SURIRE-PUTRE-CHILE")
RUTA_IMAGENES = os.path.join(RUTA_DATASET, "imagen")
RUTA_MASCARAS = os.path.join(RUTA_DATASET, "mascara")

# =============================================================================
# FUNCIONES
# =============================================================================

def cargar_coordenadas(ruta_txt):
    coordenadas = []
    with open(ruta_txt, 'r') as f:
        for linea in f:
            linea = linea.strip()
            if linea:
                x, y = linea.split(',')
                coordenadas.append((int(x), int(y)))
    print(f"   Coordenadas cargadas: {len(coordenadas)} parches definidos")
    return coordenadas


def encontrar_escenas(ruta_datos):
    patron = os.path.join(ruta_datos, "**", "R20m")
    carpetas_r20m = glob.glob(patron, recursive=True)

    escenas_validas = []
    for r20m in carpetas_r20m:
        tci = glob.glob(os.path.join(r20m, "*_TCI_20m.jp2"))
        b11 = glob.glob(os.path.join(r20m, "*_B11_20m.jp2"))
        b12 = glob.glob(os.path.join(r20m, "*_B12_20m.jp2"))

        if tci and b11 and b12:
            escenas_validas.append({
                'tci': tci[0],
                'b11': b11[0],
                'b12': b12[0],
                'r20m': r20m
            })

    escenas_validas.sort(key=lambda e: e['r20m'])
    print(f"   Escenas Sentinel-2 encontradas: {len(escenas_validas)}")
    return escenas_validas


def extraer_nombre_escena(ruta_r20m):
    partes = ruta_r20m.replace("\\", "/").split("/")
    safe_name = ""
    for p in partes:
        if p.endswith(".SAFE"):
            safe_name = p
            break
    if safe_name:
        tokens = safe_name.split("_")
        satelite = tokens[0]
        fecha = tokens[2][:8]
        return f"{satelite}_{fecha}"
    return "escena"


def procesar_escena(escena, coordenadas, contador_global):
    nombre = extraer_nombre_escena(escena['r20m'])
    print(f"\n   Procesando: {nombre}")

    with rasterio.open(escena['tci']) as src:
        rgb = np.moveaxis(src.read([1, 2, 3]), 0, -1)
        alto, ancho, _ = rgb.shape
        transform_global = src.transform
        crs_global = src.crs

    with rasterio.open(escena['b11']) as src:
        b11 = src.read(1).astype('float32')
    with rasterio.open(escena['b12']) as src:
        b12 = src.read(1).astype('float32')

    with np.errstate(divide='ignore', invalid='ignore'):
        si = (b11 - b12) / (b11 + b12)
        si = np.nan_to_num(si, nan=0.0)

    mascara = (si > UMBRAL_SI).astype(np.uint8) * 255

    parches_guardados = 0

    for i, (x, y) in enumerate(coordenadas):
        if (x + TAMANO_PARCHE > ancho) or (y + TAMANO_PARCHE > alto):
            print(f"      [!] Parche ({x},{y}) fuera de limites ({ancho}x{alto}), saltando...")
            continue
        if x < 0 or y < 0:
            print(f"      [!] Parche ({x},{y}) coordenada negativa, saltando...")
            continue

        parche_rgb = rgb[y:y+TAMANO_PARCHE, x:x+TAMANO_PARCHE]
        parche_mascara = mascara[y:y+TAMANO_PARCHE, x:x+TAMANO_PARCHE]

        nombre_archivo = f"{contador_global:05d}_{nombre}_p{i:02d}"

        # PNG
        img_pil = Image.fromarray(parche_rgb)
        img_pil.save(os.path.join(RUTA_IMAGENES, f"{nombre_archivo}.png"))

        mask_pil = Image.fromarray(parche_mascara, mode='L')
        mask_pil.save(os.path.join(RUTA_MASCARAS, f"{nombre_archivo}.png"))

        # TIF georeferenciado
        col_off = transform_global.c + x * transform_global.a
        row_off = transform_global.f + y * transform_global.e
        parche_transform = rasterio.transform.from_origin(
            col_off, row_off, abs(transform_global.a), abs(transform_global.e)
        )

        tif_profile = {
            'driver': 'GTiff', 'dtype': 'uint8',
            'width': TAMANO_PARCHE, 'height': TAMANO_PARCHE,
            'count': 3, 'crs': crs_global, 'transform': parche_transform
        }
        with rasterio.open(os.path.join(RUTA_IMAGENES, f"{nombre_archivo}.tif"), 'w', **tif_profile) as dst:
            for band in range(3):
                dst.write(parche_rgb[:, :, band], band + 1)

        mask_profile = tif_profile.copy()
        mask_profile['count'] = 1
        with rasterio.open(os.path.join(RUTA_MASCARAS, f"{nombre_archivo}.tif"), 'w', **mask_profile) as dst:
            dst.write(parche_mascara, 1)

        contador_global += 1
        parches_guardados += 1

    print(f"      [OK] {parches_guardados} parches guardados (PNG + TIF)")
    return parches_guardados, contador_global


# =============================================================================
# PROGRAMA PRINCIPAL
# =============================================================================

def main():
    print("=" * 70)
    print(" GENERADOR DE DATASET - SALAR DE SURIRE-PUTRE-CHILE")
    print(f" Umbral SI: {UMBRAL_SI} | Parche: {TAMANO_PARCHE}x{TAMANO_PARCHE} px")
    print(" Formula: SI = (B11 - B12) / (B11 + B12)")
    print(" Mascara: SI > 0.0771 -> blanco (sal) | negro (no-sal)")
    print(" Salida: PNG + TIF georeferenciado")
    print("=" * 70)

    os.makedirs(RUTA_IMAGENES, exist_ok=True)
    os.makedirs(RUTA_MASCARAS, exist_ok=True)
    print(f"\n[OK] Carpeta dataset: {RUTA_DATASET}")

    ruta_coord = os.path.normpath(RUTA_COORDENADAS)
    if not os.path.exists(ruta_coord):
        print(f"\n[ERROR] No se encontro {ruta_coord}")
        print("  Ejecuta primero SacarCoordenadas.py para definir las zonas.")
        return

    coordenadas = cargar_coordenadas(ruta_coord)
    if not coordenadas:
        print("\n[ERROR] El archivo de coordenadas esta vacio.")
        return

    escenas = encontrar_escenas(RUTA_DATOS)
    if not escenas:
        print("\n[ERROR] No se encontraron escenas en DATOS/")
        return

    total_parches = 0
    contador = 0

    for idx, escena in enumerate(escenas, 1):
        print(f"\n[{idx}/{len(escenas)}]", end="")
        guardados, contador = procesar_escena(escena, coordenadas, contador)
        total_parches += guardados

    total_archivos = total_parches * 4
    print("\n" + "=" * 70)
    print(f" [OK] DATASET GENERADO EXITOSAMENTE")
    print(f"   Escenas procesadas:    {len(escenas)}")
    print(f"   Parches por escena:    {len(coordenadas)}")
    print(f"   Total pares:           {total_parches}")
    print(f"   Total archivos:        {total_archivos}")
    print(f"   Carpeta imagen/:       {RUTA_IMAGENES}")
    print(f"   Carpeta mascara/:      {RUTA_MASCARAS}")
    print("=" * 70)


if __name__ == "__main__":
    main()
