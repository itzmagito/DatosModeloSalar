"""
==============================================================================
 SELECTOR DE COORDENADAS - SALAR DE SURIRE-PUTRE-CHILE
 VERSION LIVIANA - Carga imagen reducida para interaccion rapida
 
 Uso: Ejecuta este script. La imagen se muestra reducida (rapido),
      pero las coordenadas se guardan en resolucion real (20m).
 
 Controles:
   - Dibuja un rectangulo en la vista IZQUIERDA para crear cuadricula
   - Dibuja en la vista DERECHA o Ctrl+Clic para seleccionar varios
   - Flechas: mover todo el bloque
   - W/A/S/D: clonar parches seleccionados en esa direccion
   - Suprimir/Backspace: borrar parches seleccionados
   - G: GUARDAR coordenadas a coordenadas_salar.txt
==============================================================================
"""
import rasterio
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.widgets import RectangleSelector
import glob
import numpy as np
import math
import os

# === CONFIGURACION ===
FACTOR = 4  # Factor de reduccion (4 = imagen 4x mas chica, 16x mas rapida)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
carpeta_ref = os.path.join(SCRIPT_DIR, "REFERENCIA")
safes = glob.glob(os.path.join(carpeta_ref, "*.SAFE"))

if not safes:
    print("[ERROR] No se encontro ninguna carpeta .SAFE en REFERENCIA/")
    exit(1)

carpeta_safe = safes[0]
print(f"Usando referencia: {os.path.basename(carpeta_safe)}")

ruta_r20m = glob.glob(os.path.join(carpeta_safe, "**", "R20m"), recursive=True)
if not ruta_r20m:
    print("[ERROR] No se encontro carpeta R20m")
    exit(1)

ruta_rgb = glob.glob(os.path.join(ruta_r20m[0], "*_TCI_20m.jp2"))
if not ruta_rgb:
    print("[ERROR] No se encontro TCI_20m.jp2")
    exit(1)

print(f"Cargando imagen reducida (1/{FACTOR})...")

# Leer imagen REDUCIDA para display rapido
with rasterio.open(ruta_rgb[0]) as src:
    alto_real = src.height
    ancho_real = src.width
    alto_red = alto_real // FACTOR
    ancho_red = ancho_real // FACTOR
    rgb = np.moveaxis(
        src.read([1, 2, 3], out_shape=(3, alto_red, ancho_red)),
        0, -1
    )

print(f"  Real: {ancho_real}x{alto_real} -> Display: {ancho_red}x{alto_red}")
print("Cargando Editor...")

# Tamano de parche EN COORDENADAS REDUCIDAS
TAMANO_REAL = 256
TAMANO = TAMANO_REAL // FACTOR  # Tamano visual del parche

fig, (ax_main, ax_zoom) = plt.subplots(1, 2, figsize=(16, 8))
fig.canvas.manager.set_window_title("Selector de Coordenadas (Liviano)")
ax_main.imshow(rgb, aspect='equal')
ax_main.set_title(f"1. Dibuja area (1/{FACTOR} res)")
ax_zoom.imshow(rgb, aspect='equal')
ax_zoom.set_title("2. Selecciona/Borra -> G para guardar")

MARGEN_ZOOM = 300 // FACTOR
parches = []  # Coordenadas en escala REDUCIDA
estado = {'seleccionados': set()}
RUTA_COORDENADAS = os.path.join(SCRIPT_DIR, "coordenadas_salar.txt")


def dibujar_cuadricula():
    for p in list(ax_main.patches):
        p.remove()
    for p in list(ax_zoom.patches):
        p.remove()

    for i, p_coord in enumerate(parches):
        es_activo = (i in estado['seleccionados'])
        color = 'cyan' if es_activo else 'red'
        grosor = 2.5 if es_activo else 1.0
        x, y = p_coord
        ax_main.add_patch(patches.Rectangle(
            (x, y), TAMANO, TAMANO, ec=color, fc='none', lw=grosor, ls='--'))
        ax_zoom.add_patch(patches.Rectangle(
            (x, y), TAMANO, TAMANO, ec=color, fc='none', lw=grosor+0.5, ls='-'))

    if parches:
        xs = [p[0] for p in parches]
        ys = [p[1] for p in parches]
        ax_zoom.set_xlim(
            max(0, min(xs) - MARGEN_ZOOM),
            min(ancho_red, max(xs) + TAMANO + MARGEN_ZOOM))
        ax_zoom.set_ylim(
            min(alto_red, max(ys) + TAMANO + MARGEN_ZOOM),
            max(0, min(ys) - MARGEN_ZOOM))

    fig.canvas.draw_idle()


def al_seleccionar_crear(eclick, erelease):
    x1 = min(eclick.xdata, erelease.xdata)
    y1 = min(eclick.ydata, erelease.ydata)
    x2 = max(eclick.xdata, erelease.xdata)
    y2 = max(eclick.ydata, erelease.ydata)

    cols = max(1, math.ceil((x2 - x1) / TAMANO))
    filas = max(1, math.ceil((y2 - y1) / TAMANO))

    grid_w, grid_h = cols * TAMANO, filas * TAMANO
    if x1 + grid_w > ancho_red: x1 = ancho_red - grid_w
    if y1 + grid_h > alto_red: y1 = alto_red - grid_h
    x1, y1 = max(0, x1), max(0, y1)

    parches.clear()
    for f in range(filas):
        for c in range(cols):
            parches.append([x1 + c * TAMANO, y1 + f * TAMANO])

    estado['seleccionados'].clear()
    dibujar_cuadricula()


def al_seleccionar_multiples(eclick, erelease):
    if not parches: return
    x1 = min(eclick.xdata, erelease.xdata)
    y1 = min(eclick.ydata, erelease.ydata)
    x2 = max(eclick.xdata, erelease.xdata)
    y2 = max(eclick.ydata, erelease.ydata)

    if getattr(eclick, 'key', None) != 'control':
        estado['seleccionados'].clear()

    for i, (px, py) in enumerate(parches):
        cx, cy = px + TAMANO/2, py + TAMANO/2
        if x1 <= cx <= x2 and y1 <= cy <= y2:
            estado['seleccionados'].add(i)

    dibujar_cuadricula()


def al_hacer_clic(event):
    if fig.canvas.manager.toolbar.mode != '' or event.button != 1:
        return
    if event.inaxes not in (ax_main, ax_zoom):
        return

    cx, cy = event.xdata, event.ydata
    encontrado = -1
    for i in range(len(parches)-1, -1, -1):
        px, py = parches[i]
        if px <= cx <= px + TAMANO and py <= cy <= py + TAMANO:
            encontrado = i
            break

    if encontrado != -1:
        if event.key == 'control':
            if encontrado in estado['seleccionados']:
                estado['seleccionados'].remove(encontrado)
            else:
                estado['seleccionados'].add(encontrado)
        else:
            estado['seleccionados'] = {encontrado}
    else:
        if event.key != 'control':
            estado['seleccionados'].clear()

    dibujar_cuadricula()


def al_presionar_tecla(event):
    k = event.key.lower()

    if k == 'g':
        if not parches: return
        # Convertir coordenadas reducidas -> reales (* FACTOR)
        with open(RUTA_COORDENADAS, "a") as f:
            for p in parches:
                x_real = int(p[0] * FACTOR)
                y_real = int(p[1] * FACTOR)
                f.write(f"{x_real},{y_real}\n")
        print(f" -> {len(parches)} parches guardados (coords reales x{FACTOR})!")
        for p in ax_main.patches + ax_zoom.patches:
            p.set_edgecolor('lime')
        fig.canvas.draw_idle()
        return

    if k in ('up', 'down', 'left', 'right'):
        paso = max(1, 32 // FACTOR)
        dx = -paso if k == 'left' else paso if k == 'right' else 0
        dy = -paso if k == 'up' else paso if k == 'down' else 0

        puede = all(
            0 <= p[0]+dx <= ancho_red-TAMANO and
            0 <= p[1]+dy <= alto_red-TAMANO
            for p in parches)
        if puede:
            for p in parches:
                p[0] += dx
                p[1] += dy
            dibujar_cuadricula()
        return

    if not estado['seleccionados']:
        return

    if k in ('delete', 'backspace'):
        for idx in sorted(list(estado['seleccionados']), reverse=True):
            parches.pop(idx)
        estado['seleccionados'].clear()

    elif k in ('w', 'a', 's', 'd'):
        nuevos = []
        for idx in estado['seleccionados']:
            nx, ny = parches[idx][0], parches[idx][1]
            if k == 'w': ny -= TAMANO
            elif k == 's': ny += TAMANO
            elif k == 'a': nx -= TAMANO
            elif k == 'd': nx += TAMANO
            if [nx, ny] not in parches and [nx, ny] not in nuevos:
                nuevos.append([nx, ny])

        if nuevos:
            dx2, dy2 = 0, 0
            for nx, ny in nuevos:
                if nx < 0: dx2 = min(dx2, nx)
                elif nx + TAMANO > ancho_red: dx2 = max(dx2, (nx+TAMANO)-ancho_red)
                if ny < 0: dy2 = min(dy2, ny)
                elif ny + TAMANO > alto_red: dy2 = max(dy2, (ny+TAMANO)-alto_red)
            if dx2 != 0 or dy2 != 0:
                for p in parches + nuevos:
                    p[0] -= dx2
                    p[1] -= dy2
            parches.extend(nuevos)
            estado['seleccionados'] = set(range(len(parches)-len(nuevos), len(parches)))

    dibujar_cuadricula()


selector_main = RectangleSelector(ax_main, al_seleccionar_crear,
    useblit=True, button=[1], interactive=True)
selector_zoom = RectangleSelector(ax_zoom, al_seleccionar_multiples,
    useblit=True, button=[1], interactive=True)

fig.canvas.mpl_connect('button_press_event', al_hacer_clic)
fig.canvas.mpl_connect('key_press_event', al_presionar_tecla)
plt.tight_layout()
plt.show()
