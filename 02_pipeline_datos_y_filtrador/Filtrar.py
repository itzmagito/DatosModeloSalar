"""
==============================================================================
 FILTRADOR DE DATASET - Herramienta de Control de Calidad
 
 Permite revisar visualmente cada imagen del dataset procesado y decidir
 si se acepta o rechaza. Las decisiones se guardan en logs para que luego
 solo se usen las coordenadas/fechas aceptadas al generar el dataset final.
 
 Controles:
   A = ACEPTAR (conservar esta imagen/fecha)
   D = RECHAZAR (descartar esta imagen/fecha)
   Q = SALIR del filtrado actual
 
 Salida:
   FILTRADOR/resultados/NOMBRE_SALAR_data_con_valor.txt <- fechas+coords aceptadas
   FILTRADOR/historial/NOMBRE_SALAR_historial.txt   <- log completo accept/reject
==============================================================================
"""
import os
import sys
import glob
import re
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
import numpy as np
from datetime import datetime

# =============================================================================
# CONFIGURACION
# =============================================================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.dirname(SCRIPT_DIR)  # DATASET/
PROCESADO_DIR = os.path.join(DATASET_DIR, "DATASET PROCESADO")
RESULTADOS_DIR = os.path.join(SCRIPT_DIR, "resultados")
HISTORIAL_DIR = os.path.join(SCRIPT_DIR, "historial")

os.makedirs(RESULTADOS_DIR, exist_ok=True)
os.makedirs(HISTORIAL_DIR, exist_ok=True)


def listar_salares():
    """Lista los salares disponibles en DATASET PROCESADO"""
    if not os.path.exists(PROCESADO_DIR):
        print("[ERROR] No existe la carpeta DATASET PROCESADO.")
        print("  Ejecuta primero Procesar.py en cada salar.")
        return []
    
    salares = sorted([
        d for d in os.listdir(PROCESADO_DIR)
        if os.path.isdir(os.path.join(PROCESADO_DIR, d))
    ])
    return salares


def obtener_imagenes(salar):
    """Obtiene todas las imagenes PNG del salar, agrupadas por fecha"""
    img_dir = os.path.join(PROCESADO_DIR, salar, "imagen")
    mask_dir = os.path.join(PROCESADO_DIR, salar, "mascara")
    
    if not os.path.exists(img_dir):
        print(f"[ERROR] No existe carpeta imagen/ para {salar}")
        return {}
    
    # Agrupar PNGs por fecha
    pngs = sorted(glob.glob(os.path.join(img_dir, "*.png")))
    
    por_fecha = {}
    for png_path in pngs:
        nombre = os.path.basename(png_path)
        # Formato: XXXXX_S2X_YYYYMMDD_pNN.png
        match = re.search(r'_S2[ABC]_(\d{8})_p(\d+)', nombre)
        if match:
            fecha = match.group(1)
            parche = match.group(2)
            if fecha not in por_fecha:
                por_fecha[fecha] = []
            
            # Buscar mascara correspondiente
            mask_name = nombre.replace("imagen", "mascara")
            mask_path = os.path.join(mask_dir, nombre)
            
            por_fecha[fecha].append({
                'imagen': png_path,
                'mascara': mask_path if os.path.exists(mask_path) else None,
                'parche': parche,
                'nombre': nombre,
                'fecha': fecha
            })
    
    return por_fecha


def filtrar_salar(salar):
    """Proceso interactivo de filtrado para un salar"""
    por_fecha = obtener_imagenes(salar)
    
    if not por_fecha:
        print(f"  [!] No se encontraron imagenes para {salar}")
        return
    
    fechas = sorted(por_fecha.keys())
    total_fechas = len(fechas)
    
    print(f"\n  Imagenes encontradas: {sum(len(v) for v in por_fecha.values())} en {total_fechas} fechas")
    print(f"  Controles: A=Aceptar | D=Rechazar | Q=Salir")
    print(f"  {'='*50}")
    
    aceptadas = []
    rechazadas = []
    decision = [None]  # Mutable para el callback
    
    for idx_fecha, fecha in enumerate(fechas):
        items = por_fecha[fecha]
        fecha_fmt = f"{fecha[:4]}-{fecha[4:6]}-{fecha[6:]}"
        
        # Mostrar todos los parches de esta fecha en una grilla
        n_items = len(items)
        cols = min(n_items, 4)
        rows = (n_items + cols - 1) // cols
        
        # 2 filas: imagen arriba, mascara abajo
        fig, axes = plt.subplots(2, cols, figsize=(4*cols, 8))
        if cols == 1:
            axes = axes.reshape(2, 1)
        
        fig.canvas.manager.set_window_title(
            f"FILTRADOR - {salar} | Fecha {idx_fecha+1}/{total_fechas}: {fecha_fmt}"
        )
        
        fig.suptitle(
            f"{salar}\nFecha: {fecha_fmt}  ({idx_fecha+1}/{total_fechas})  |  "
            f"{n_items} parche(s)\n\n[A] = ACEPTAR   |   [D] = RECHAZAR   |   [Q] = SALIR",
            fontsize=12, fontweight='bold'
        )
        
        for i in range(cols):
            if i < n_items:
                # Imagen real
                img = Image.open(items[i]['imagen'])
                axes[0, i].imshow(np.array(img))
                axes[0, i].set_title(f"p{items[i]['parche']} - RGB", fontsize=9)
                axes[0, i].axis('off')
                
                # Mascara
                if items[i]['mascara'] and os.path.exists(items[i]['mascara']):
                    mask = Image.open(items[i]['mascara'])
                    axes[1, i].imshow(np.array(mask), cmap='gray')
                    axes[1, i].set_title(f"p{items[i]['parche']} - Mascara SI", fontsize=9)
                else:
                    axes[1, i].text(0.5, 0.5, 'Sin mascara', ha='center', va='center')
                axes[1, i].axis('off')
            else:
                axes[0, i].axis('off')
                axes[1, i].axis('off')
        
        plt.tight_layout()
        
        # Esperar decision del usuario
        decision[0] = None
        
        def on_key(event):
            k = event.key.lower()
            if k == 'a':
                decision[0] = 'ACEPTADO'
                plt.close(fig)
            elif k == 'd':
                decision[0] = 'RECHAZADO'
                plt.close(fig)
            elif k == 'q':
                decision[0] = 'SALIR'
                plt.close(fig)
        
        fig.canvas.mpl_connect('key_press_event', on_key)
        plt.show()
        
        if decision[0] == 'SALIR':
            print("  [!] Filtrado interrumpido por el usuario.")
            break
        
        # Registrar decision
        coords_str = "; ".join([
            f"p{it['parche']}" for it in items
        ])
        
        if decision[0] == 'ACEPTADO':
            aceptadas.append({
                'fecha': fecha,
                'fecha_fmt': fecha_fmt,
                'parches': items,
                'coords': coords_str
            })
            print(f"  [{idx_fecha+1}/{total_fechas}] {fecha_fmt} -> ACEPTADO ({n_items} parches)")
        elif decision[0] == 'RECHAZADO':
            rechazadas.append({
                'fecha': fecha,
                'fecha_fmt': fecha_fmt,
                'parches': items,
                'coords': coords_str
            })
            print(f"  [{idx_fecha+1}/{total_fechas}] {fecha_fmt} -> RECHAZADO ({n_items} parches)")
    
    # Guardar resultados
    salar_clean = salar.replace(" ", "_")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # 1. Archivo de fechas aceptadas (para uso futuro)
    filtrado_path = os.path.join(RESULTADOS_DIR, f"{salar_clean}_data_con_valor.txt")
    with open(filtrado_path, "w", encoding="utf-8") as f:
        f.write(f"# Filtrado de: {salar}\n")
        f.write(f"# Fecha de filtrado: {timestamp}\n")
        f.write(f"# Total aceptadas: {len(aceptadas)}\n")
        f.write(f"# Total rechazadas: {len(rechazadas)}\n")
        f.write(f"# Formato: FECHA,PARCHES\n")
        f.write(f"#{'='*50}\n")
        for item in aceptadas:
            f.write(f"{item['fecha']},{item['coords']}\n")
    
    # 2. Historial completo (trazabilidad)
    historial_path = os.path.join(HISTORIAL_DIR, f"{salar_clean}_historial.txt")
    with open(historial_path, "w", encoding="utf-8") as f:
        f.write(f"{'='*60}\n")
        f.write(f" HISTORIAL DE FILTRADO - {salar}\n")
        f.write(f" Fecha: {timestamp}\n")
        f.write(f"{'='*60}\n\n")
        
        f.write(f"--- ACEPTADAS ({len(aceptadas)}) ---\n")
        for item in aceptadas:
            f.write(f"  ACEPTADO | {item['fecha_fmt']} | {item['coords']}\n")
        
        f.write(f"\n--- RECHAZADAS ({len(rechazadas)}) ---\n")
        for item in rechazadas:
            f.write(f"  RECHAZADO | {item['fecha_fmt']} | {item['coords']}\n")
        
        f.write(f"\n{'='*60}\n")
        f.write(f" Resumen: {len(aceptadas)} aceptadas, {len(rechazadas)} rechazadas\n")
        f.write(f"{'='*60}\n")
    
    print(f"\n  Resultados guardados:")
    print(f"    Aceptadas:  {len(aceptadas)}")
    print(f"    Rechazadas: {len(rechazadas)}")
    print(f"    Filtrado:   {filtrado_path}")
    print(f"    Historial:  {historial_path}")


# =============================================================================
# PROGRAMA PRINCIPAL
# =============================================================================
def main():
    print("=" * 60)
    print(" FILTRADOR DE DATASET - Control de Calidad Visual")
    print(" Controles: A=Aceptar | D=Rechazar | Q=Salir")
    print("=" * 60)
    
    salares = listar_salares()
    
    if not salares:
        return
    
    while True:
        print(f"\n Salares disponibles ({len(salares)}):")
        print("-" * 40)
        for i, s in enumerate(salares, 1):
            # Verificar si ya fue filtrado
            salar_clean = s.replace(" ", "_")
            ya_filtrado = os.path.exists(
                os.path.join(RESULTADOS_DIR, f"{salar_clean}_data_con_valor.txt")
            )
            estado = " [YA FILTRADO]" if ya_filtrado else ""
            print(f"  {i:2d}. {s}{estado}")
        
        print(f"   0. SALIR")
        print()
        
        try:
            opcion = input(" Selecciona un salar (numero): ").strip()
            if opcion == '0' or opcion.lower() == 'q':
                print("\n Filtrado finalizado.")
                break
            
            idx = int(opcion) - 1
            if 0 <= idx < len(salares):
                salar = salares[idx]
                print(f"\n Filtrando: {salar}")
                filtrar_salar(salar)
            else:
                print(" [!] Numero invalido")
        except ValueError:
            print(" [!] Ingresa un numero valido")
        except KeyboardInterrupt:
            print("\n\n Filtrado finalizado.")
            break


if __name__ == "__main__":
    main()
