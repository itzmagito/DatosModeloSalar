"""
mover_rechazados.py
Mueve los parches RECHAZADOS (según filtrado) de DATASET PROCESADO a D:\TESIS2DATOS como backup.
Solo quedan en DATASET PROCESADO los parches ACEPTADOS.
"""
import os
import re
import shutil
import sys

sys.stdout.reconfigure(encoding='utf-8')

# ====== RUTAS ======
DATASET_DIR = r"c:\Users\Mathias\Desktop\TESIS2\DATASET"
PROCESADO_DIR = os.path.join(DATASET_DIR, "DATASET PROCESADO")
RESULTADOS_DIR = os.path.join(DATASET_DIR, "FILTRADOR", "resultados")
BACKUP_BASE = r"D:\TESIS2DATOS"

DRY_RUN = False  # True = solo muestra qué haría, False = ejecuta de verdad


def cargar_aceptados(salar):
    """Lee _data_con_valor.txt y retorna set de (fecha, parche) aceptados"""
    salar_clean = salar.replace(" ", "_")
    res_path = os.path.join(RESULTADOS_DIR, f"{salar_clean}_data_con_valor.txt")
    
    accepted = set()
    if not os.path.exists(res_path):
        return accepted
    
    with open(res_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line.startswith('#') or not line:
                continue
            parts = line.split(',', 1)
            if len(parts) == 2:
                fecha = parts[0].strip()
                parches = [p.strip() for p in parts[1].split(';')]
                for p in parches:
                    accepted.add((fecha, p))
    return accepted


def mover_rechazados():
    """Mueve archivos rechazados de DATASET PROCESADO a D: TESIS2DATOS backup"""
    
    total_movidos = 0
    total_conservados = 0
    
    salares = sorted([d for d in os.listdir(PROCESADO_DIR) 
                      if os.path.isdir(os.path.join(PROCESADO_DIR, d))])
    
    for salar in salares:
        accepted = cargar_aceptados(salar)
        if not accepted:
            print(f"⚠️  {salar}: Sin archivo de filtrado, SALTANDO")
            continue
        
        salar_proc = os.path.join(PROCESADO_DIR, salar)
        movidos_salar = 0
        conservados_salar = 0
        
        for subcarpeta in ['imagen', 'mascara']:
            src_dir = os.path.join(salar_proc, subcarpeta)
            if not os.path.exists(src_dir):
                continue
            
            # Destino backup
            dst_dir = os.path.join(BACKUP_BASE, salar, "RECHAZADOS", subcarpeta)
            
            archivos = sorted(os.listdir(src_dir))
            
            for archivo in archivos:
                # Parse nombre: 00000_S2A_20251225_p00.png o .tif
                m = re.match(r'(\d+)_S\d[A-C]_(\d{8})_(p\d+)\.(png|tif)', archivo)
                if not m:
                    continue
                
                fecha = m.group(2)
                parche = m.group(3)
                
                if (fecha, parche) in accepted:
                    conservados_salar += 1
                else:
                    # RECHAZADO -> mover a backup
                    src_path = os.path.join(src_dir, archivo)
                    dst_path = os.path.join(dst_dir, archivo)
                    
                    if DRY_RUN:
                        pass  # solo contar
                    else:
                        os.makedirs(dst_dir, exist_ok=True)
                        shutil.move(src_path, dst_path)
                    
                    movidos_salar += 1
        
        total_movidos += movidos_salar
        total_conservados += conservados_salar
        
        accion = "se moverian" if DRY_RUN else "movidos"
        print(f"{'OK' if movidos_salar == 0 else 'MOVIDO'} {salar}: "
              f"{conservados_salar} archivos conservados (aceptados), "
              f"{movidos_salar} {accion} a D: (rechazados)")
    
    print()
    print("=" * 60)
    modo = "[DRY RUN - sin cambios reales]" if DRY_RUN else "[EJECUTADO]"
    print(f"  RESUMEN FINAL {modo}")
    print(f"  Archivos conservados (aceptados): {total_conservados}")
    print(f"  Archivos movidos a D: (rechazados): {total_movidos}")
    print("=" * 60)


if __name__ == "__main__":
    print("=" * 60)
    print("  MOVER RECHAZADOS A BACKUP D:\\TESIS2DATOS")
    print("  Modo: {}".format("DRY RUN (simulacion)" if DRY_RUN else "EJECUCION REAL"))
    print("=" * 60)
    print()
    mover_rechazados()
