import os
import sys
import glob
import re
import json
import tkinter as tk
from tkinter import ttk, messagebox
from PIL import Image, ImageTk, ImageDraw
from datetime import datetime
import numpy as np
import xml.etree.ElementTree as ET

try:
    import rasterio
    from rasterio.windows import Window
except ImportError:
    rasterio = None

# =============================================================================
# CONFIGURACION
# =============================================================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.dirname(SCRIPT_DIR)  # DATASET/
PROCESADO_DIR = os.path.join(DATASET_DIR, "DATASET PROCESADO")
RESULTADOS_DIR = os.path.join(SCRIPT_DIR, "resultados")
HISTORIAL_DIR = os.path.join(SCRIPT_DIR, "historial")
CACHE_DIR = os.path.join(SCRIPT_DIR, "cache")

os.makedirs(RESULTADOS_DIR, exist_ok=True)
os.makedirs(HISTORIAL_DIR, exist_ok=True)
os.makedirs(CACHE_DIR, exist_ok=True)


class FiltradorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("FILTRADOR DE DATASET - TESIS 2")
        self.root.geometry("1400x800")
        self.root.configure(bg="#1e1e1e")
        try:
            self.root.state('zoomed') # Maximizar en Windows
        except:
            pass
        
        self.salares = self.listar_salares()
        self.salar_actual = None
        self.lista_parches = []  
        self.idx_actual = 0
        self.coordenadas_map = {} # ID de parche -> (X, Y)
        
        self.vista_modo = "doble"  # "doble" (Doble Comparativa), "zoom" (Solo Fecha Actual), "general" (Mapa Salar)
        self.pad_cuadrantes = 2.0  # Cuadrantes de margen alrededor del parche (~2 cuadrantes como margen)
        self.current_ctx_base = None  # Cache en memoria de imagen PIL de la fecha actual
        self.current_meta = None
        self.current_ctx_fecha = None
        
        self.ref_ctx_base = None  # Cache en memoria de imagen PIL de REFERENCIA limpia
        self.ref_meta = None
        
        self.aceptadas = []
        self.rechazadas = []
        
        self.current_img_tk = None
        self.current_act_tk = None
        self.current_ref_tk = None
        self.current_single_tk = None
        self.cache_nubes_parche = {}  # Cache (scl_path, x, y) -> pct_nubes
        
        # Detección de resolución de pantalla para escalar las imágenes dinámicamente
        screen_h = self.root.winfo_screenheight()
        if screen_h >= 1200:  # Monitores 2K / 1440p (como 2560x1440)
            self.patch_disp_size = (760, 760)
            self.doble_disp_size = (720, 410)
            self.single_disp_size = (760, 760)
        elif screen_h >= 950:  # Monitores Full HD 1080p
            self.patch_disp_size = (620, 620)
            self.doble_disp_size = (560, 330)
            self.single_disp_size = (620, 620)
        else:  # Pantallas compactas (768p)
            self.patch_disp_size = (480, 480)
            self.doble_disp_size = (440, 250)
            self.single_disp_size = (480, 480)
        
        self.setup_ui()
        
    def listar_salares(self):
        if not os.path.exists(PROCESADO_DIR):
            return []
        return sorted([d for d in os.listdir(PROCESADO_DIR) if os.path.isdir(os.path.join(PROCESADO_DIR, d))])
        
    def setup_ui(self):
        style = ttk.Style()
        style.theme_use('clam')
        
        # Top Frame (Selector)
        top_frame = tk.Frame(self.root, bg="#2d2d2d", pady=10)
        top_frame.pack(fill=tk.X)
        
        lbl_salar = tk.Label(top_frame, text="Salar:", font=('Segoe UI', 12, 'bold'), bg="#2d2d2d", fg="white")
        lbl_salar.pack(side=tk.LEFT, padx=15)
        
        self.combo_salar = ttk.Combobox(top_frame, values=self.salares, state="readonly", width=40, font=('Segoe UI', 11))
        self.combo_salar.pack(side=tk.LEFT, padx=5)
        self.combo_salar.bind("<<ComboboxSelected>>", self.on_salar_selected)
        
        self.lbl_estado = tk.Label(top_frame, text="Selecciona un salar para comenzar...", font=('Segoe UI', 14, 'bold'), bg="#2d2d2d", fg="#38bdf8")
        self.lbl_estado.pack(side=tk.LEFT, padx=30)
        
        # Middle Frame (Split panels)
        self.main_container = tk.Frame(self.root, bg="#1e1e1e")
        self.main_container.pack(fill=tk.BOTH, expand=True, padx=20, pady=10)
        
        # Left Panel (Patch Image)
        self.left_panel = tk.Frame(self.main_container, bg="#1e1e1e")
        self.left_panel.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        tk.Label(self.left_panel, text="PARCHE ACTUAL (256x256 real)", font=('Segoe UI', 12, 'bold'), bg="#1e1e1e", fg="#94a3b8").pack(pady=3)
        
        self.lbl_filename = tk.Label(self.left_panel, text="", font=('Consolas', 11), bg="#1e1e1e", fg="#4ade80")
        self.lbl_filename.pack()
        
        self.lbl_patch_clouds = tk.Label(self.left_panel, text="", font=('Segoe UI', 11, 'bold'), bg="#1e1e1e", fg="#38bdf8")
        self.lbl_patch_clouds.pack(pady=(2, 2))
        
        self.lbl_imagen = tk.Label(self.left_panel, bg="#1e1e1e")
        self.lbl_imagen.pack(expand=True)
        
        # Right Panel (Context Map)
        self.right_panel = tk.Frame(self.main_container, bg="#1e1e1e")
        self.right_panel.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        
        ctx_top_bar = tk.Frame(self.right_panel, bg="#1e1e1e")
        ctx_top_bar.pack(fill=tk.X, pady=2)
        
        tk.Label(ctx_top_bar, text="MAPAS DE CONTEXTO", font=('Segoe UI', 12, 'bold'), bg="#1e1e1e", fg="#94a3b8").pack(side=tk.LEFT, padx=5)
        
        # Controles de Zoom interactivo (+ / -)
        zoom_frame = tk.Frame(ctx_top_bar, bg="#1e1e1e")
        zoom_frame.pack(side=tk.LEFT, padx=15)
        
        btn_zoom_out = tk.Button(
            zoom_frame, text="➖ [-]", font=('Segoe UI', 9, 'bold'), bg="#334155", fg="white", 
            activebackground="#475569", activeforeground="white", relief=tk.FLAT, padx=6, pady=1, cursor="hand2",
            command=lambda: self.cambiar_zoom(0.5)
        )
        btn_zoom_out.pack(side=tk.LEFT, padx=2)
        
        self.lbl_zoom_info = tk.Label(zoom_frame, text=f"Zoom: {self.pad_cuadrantes}x cuadrantes", font=('Segoe UI', 10, 'bold'), bg="#1e1e1e", fg="#38bdf8")
        self.lbl_zoom_info.pack(side=tk.LEFT, padx=5)
        
        btn_zoom_in = tk.Button(
            zoom_frame, text="➕ [+]", font=('Segoe UI', 9, 'bold'), bg="#334155", fg="white", 
            activebackground="#475569", activeforeground="white", relief=tk.FLAT, padx=6, pady=1, cursor="hand2",
            command=lambda: self.cambiar_zoom(-0.5)
        )
        btn_zoom_in.pack(side=tk.LEFT, padx=2)
        
        self.btn_toggle_vista = tk.Button(
            ctx_top_bar, 
            text="👁️ Vista: Doble Comparativa [V]", 
            font=('Segoe UI', 9, 'bold'), 
            bg="#0f766e", 
            fg="white", 
            activebackground="#115e59",
            activeforeground="white",
            relief=tk.FLAT,
            padx=10, 
            pady=3,
            cursor="hand2",
            command=self.toggle_vista
        )
        self.btn_toggle_vista.pack(side=tk.RIGHT, padx=5)
        
        # 1. Contenedor para modo DOBLE (Arriba: Fecha Actual / Abajo: Referencia Limpia)
        self.frame_doble = tk.Frame(self.right_panel, bg="#1e1e1e")
        self.frame_doble.pack(fill=tk.BOTH, expand=True)
        
        self.lbl_act_title = tk.Label(self.frame_doble, text="", font=('Consolas', 10, 'bold'), bg="#1e1e1e", fg="#facc15", wraplength=700)
        self.lbl_act_title.pack(pady=(2, 0))
        self.lbl_act_img = tk.Label(self.frame_doble, bg="#1e1e1e")
        self.lbl_act_img.pack(pady=2, expand=True)
        
        self.lbl_ref_title = tk.Label(self.frame_doble, text="", font=('Consolas', 10, 'bold'), bg="#1e1e1e", fg="#4ade80", wraplength=700)
        self.lbl_ref_title.pack(pady=(3, 0))
        self.lbl_ref_img = tk.Label(self.frame_doble, bg="#1e1e1e")
        self.lbl_ref_img.pack(pady=2, expand=True)
        
        # 2. Contenedor para modo SINGLE (Zoom Grande o Mapa Salar)
        self.frame_single = tk.Frame(self.right_panel, bg="#1e1e1e")
        
        self.lbl_single_title = tk.Label(self.frame_single, text="", font=('Consolas', 10, 'bold'), bg="#1e1e1e", fg="#facc15", wraplength=700)
        self.lbl_single_title.pack(pady=2)
        self.lbl_single_img = tk.Label(self.frame_single, bg="#1e1e1e")
        self.lbl_single_img.pack(expand=True)
        
        # Bottom Frame (Buttons)
        bottom_frame = tk.Frame(self.root, bg="#2d2d2d", pady=15)
        bottom_frame.pack(fill=tk.X, side=tk.BOTTOM)
        
        self.btn_undo = tk.Button(bottom_frame, text="↩ DESHACER (Z)", font=('Segoe UI', 14, 'bold'), bg='#f57c00', fg='white', activebackground='#e65100', activeforeground='white', command=self.deshacer, state=tk.DISABLED, height=2, width=15, relief=tk.FLAT)
        self.btn_undo.pack(side=tk.LEFT, padx=20)
        
        self.btn_accept = tk.Button(bottom_frame, text="✅ ACEPTAR (A)", font=('Segoe UI', 16, 'bold'), bg='#2e7d32', fg='white', activebackground='#1b5e20', activeforeground='white', command=lambda: self.registrar_decision('ACEPTADO'), state=tk.DISABLED, height=2, width=20, relief=tk.FLAT)
        self.btn_accept.pack(side=tk.LEFT, padx=20, expand=True)
        
        self.btn_reject = tk.Button(bottom_frame, text="❌ RECHAZAR (D)", font=('Segoe UI', 16, 'bold'), bg='#c62828', fg='white', activebackground='#b71c1c', activeforeground='white', command=lambda: self.registrar_decision('RECHAZADO'), state=tk.DISABLED, height=2, width=20, relief=tk.FLAT)
        self.btn_reject.pack(side=tk.LEFT, padx=20, expand=True)
        
        # Bind keyboard shortcuts
        self.root.bind('<a>', lambda e: self.btn_accept.invoke() if self.btn_accept['state'] == tk.NORMAL else None)
        self.root.bind('<A>', lambda e: self.btn_accept.invoke() if self.btn_accept['state'] == tk.NORMAL else None)
        self.root.bind('<d>', lambda e: self.btn_reject.invoke() if self.btn_reject['state'] == tk.NORMAL else None)
        self.root.bind('<D>', lambda e: self.btn_reject.invoke() if self.btn_reject['state'] == tk.NORMAL else None)
        self.root.bind('<z>', lambda e: self.btn_undo.invoke() if self.btn_undo['state'] == tk.NORMAL else None)
        self.root.bind('<Z>', lambda e: self.btn_undo.invoke() if self.btn_undo['state'] == tk.NORMAL else None)
        self.root.bind('<v>', lambda e: self.toggle_vista())
        self.root.bind('<V>', lambda e: self.toggle_vista())
        
        # Zoom shortcuts (+/- y rueda del raton)
        self.root.bind('+', lambda e: self.cambiar_zoom(-0.5))
        self.root.bind('=', lambda e: self.cambiar_zoom(-0.5))
        self.root.bind('<plus>', lambda e: self.cambiar_zoom(-0.5))
        self.root.bind('<KP_Add>', lambda e: self.cambiar_zoom(-0.5))
        self.root.bind('-', lambda e: self.cambiar_zoom(0.5))
        self.root.bind('<minus>', lambda e: self.cambiar_zoom(0.5))
        self.root.bind('<KP_Subtract>', lambda e: self.cambiar_zoom(0.5))
        self.root.bind('<Up>', lambda e: self.cambiar_zoom(-0.5))
        self.root.bind('<Down>', lambda e: self.cambiar_zoom(0.5))
        self.root.bind('<MouseWheel>', lambda e: self.cambiar_zoom(-0.5 if e.delta > 0 else 0.5))

    def cambiar_zoom(self, delta):
        nueva = round(self.pad_cuadrantes + delta, 1)
        if 0.5 <= nueva <= 8.0:
            self.pad_cuadrantes = nueva
            if hasattr(self, 'lbl_zoom_info'):
                self.lbl_zoom_info.config(text=f"Zoom: {self.pad_cuadrantes}x cuadrantes")
            self.actualizar_vista_contexto()

    def toggle_vista(self):
        if self.vista_modo == "doble":
            self.vista_modo = "zoom"
            self.btn_toggle_vista.config(text="🔍 Vista: Zoom Fecha Actual [V]", bg="#b45309", activebackground="#92400e")
            self.frame_doble.pack_forget()
            self.frame_single.pack(fill=tk.BOTH, expand=True)
        elif self.vista_modo == "zoom":
            self.vista_modo = "general"
            self.btn_toggle_vista.config(text="🗺️ Vista: Mapa Salar [V]", bg="#1e3a8a", activebackground="#1e40af")
            self.frame_doble.pack_forget()
            self.frame_single.pack(fill=tk.BOTH, expand=True)
        else:
            self.vista_modo = "doble"
            self.btn_toggle_vista.config(text="👁️ Vista: Doble Comparativa [V]", bg="#0f766e", activebackground="#115e59")
            self.frame_single.pack_forget()
            self.frame_doble.pack(fill=tk.BOTH, expand=True)
        self.actualizar_vista_contexto()

    def cargar_coordenadas(self, salar):
        """Carga ÚNICAMENTE el mapa de coordenadas espaciales originales (coordenadas_salar.txt)"""
        coords_path = os.path.join(DATASET_DIR, salar, "coordenadas_salar.txt")
        self.coordenadas_map = {}
        idx = 0
        if os.path.exists(coords_path):
            with open(coords_path, "r") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        parts = line.split(',')
                        if len(parts) >= 2:
                            self.coordenadas_map[idx] = (int(parts[0]), int(parts[1]))
                            idx += 1

    def obtener_porcentaje_nubes(self, safe_path):
        """Lee el porcentaje de nubosidad oficial de Sentinel-2 desde MTD_MSIL2A.xml"""
        if not safe_path or not os.path.exists(safe_path):
            return None
        xmls = glob.glob(os.path.join(safe_path, "**/MTD_*.xml"), recursive=True)
        for x in xmls:
            try:
                tree = ET.parse(x)
                for elem in tree.getroot().iter():
                    if 'cloud_coverage_assessment' in elem.tag.lower():
                        return float(elem.text)
            except Exception:
                pass
        return None

    def obtener_nubes_parche(self, scl_path, x, y, size=256):
        """Calcula el porcentaje de píxeles con nubes o sombras en el parche (256x256)"""
        if not rasterio or not scl_path or not os.path.exists(scl_path):
            return None
        cache_key = (scl_path, x, y)
        if cache_key in self.cache_nubes_parche:
            return self.cache_nubes_parche[cache_key]
        try:
            w = Window(x, y, size, size)
            with rasterio.open(scl_path) as src:
                scl = src.read(1, window=w)
            nubes_mask = np.isin(scl, [3, 8, 9, 10])
            pct = float(np.mean(nubes_mask) * 100)
            self.cache_nubes_parche[cache_key] = pct
            return pct
        except Exception:
            return None

    def encontrar_escena_mas_limpia(self, salar):
        """Busca entre todas las carpetas .SAFE (REFERENCIA y DATOS) la que tenga menor porcentaje de nubes"""
        salar_dir = os.path.join(DATASET_DIR, salar)
        safes_candidatos = glob.glob(os.path.join(salar_dir, "**/*.SAFE"), recursive=True)
        
        real_safes = set()
        for sf in safes_candidatos:
            parts = sf.split(os.sep)
            for i, p in enumerate(parts):
                if p.endswith('.SAFE') and not p.lower().startswith('manifest'):
                    real_safes.add(os.sep.join(parts[:i+1]))
                    break
                    
        if not real_safes:
            return None, None, None, ""
            
        candidatos = []
        for sf in real_safes:
            tcis = glob.glob(os.path.join(sf, "**", "*_TCI_20m.jp2"), recursive=True)
            if tcis:
                nubes = self.obtener_porcentaje_nubes(sf)
                val_sort = nubes if nubes is not None else 999.0
                
                m = re.search(r'(\d{4})(\d{2})(\d{2})T', os.path.basename(sf))
                fecha_fmt = f"{m.group(1)}-{m.group(2)}-{m.group(3)}" if m else ""
                
                candidatos.append((val_sort, sf, tcis[0], nubes, fecha_fmt))
                
        if not candidatos:
            return None, None, None, ""
            
        candidatos.sort(key=lambda x: x[0])
        ganador = candidatos[0]
        return ganador[1], ganador[2], ganador[3], ganador[4]

    def obtener_nombre_safe(self, salar, fecha):
        """Busca el nombre de la carpeta .SAFE original para esta fecha"""
        datos_dir = os.path.join(DATASET_DIR, salar, "DATOS")
        fecha_fmt = f"{fecha[:4]}-{fecha[4:6]}-{fecha[6:]}"
        fecha_dir = os.path.join(datos_dir, fecha_fmt)
        
        safe_dirs = glob.glob(os.path.join(fecha_dir, "*.SAFE"))
        if not safe_dirs:
            safe_dirs = glob.glob(os.path.join(fecha_dir, "*", "*.SAFE"))
        if not safe_dirs:
            ref_dirs = [
                d for d in glob.glob(os.path.join(DATASET_DIR, salar, "REFERENCIA", "*.SAFE"))
                if os.path.isdir(d) and not os.path.basename(d).lower().startswith("manifest")
            ]
            for r in ref_dirs:
                if fecha in r: return os.path.basename(r)
            return "Carpeta .SAFE no encontrada"
        return os.path.basename(safe_dirs[0])

    def obtener_contexto(self, salar, fecha):
        """Genera o carga el mapa recortado R20m del salar completo como base de contexto"""
        if not rasterio or not self.coordenadas_map:
            return None, None
            
        # Si ya lo tenemos en memoria RAM para esta fecha, retornarlo de inmediato
        if self.current_ctx_fecha == fecha and self.current_ctx_base is not None and self.current_meta is not None:
            return self.current_ctx_base, self.current_meta
            
        salar_clean = salar.replace(" ", "_")
        cache_path = os.path.join(CACHE_DIR, f"{salar_clean}_{fecha}_R20m_crop.png")
        meta_path = os.path.join(CACHE_DIR, f"{salar_clean}_{fecha}_R20m_meta.json")
        
        # Verificar caché en disco
        if os.path.exists(cache_path) and os.path.exists(meta_path):
            try:
                with open(meta_path, 'r') as f:
                    meta = json.load(f)
                    
                # Asegurar que scl_path y nubes existan
                if 'scl_path' not in meta or not meta.get('scl_path'):
                    tci_p = meta.get('tci_path', '')
                    cand_scl = tci_p.replace('_TCI_20m.jp2', '_SCL_20m.jp2')
                    if os.path.exists(cand_scl):
                        meta['scl_path'] = cand_scl
                if 'nubes' not in meta or meta.get('nubes') is None:
                    tci_p = meta.get('tci_path', '')
                    parts = tci_p.split(os.sep)
                    for i, p in enumerate(parts):
                        if p.endswith('.SAFE'):
                            meta['nubes'] = self.obtener_porcentaje_nubes(os.sep.join(parts[:i+1]))
                            break
                    with open(meta_path, 'w') as f:
                        json.dump(meta, f)
                        
                img = Image.open(cache_path)
                self.current_ctx_base = img
                self.current_meta = meta
                self.current_ctx_fecha = fecha
                return img, meta
            except Exception:
                pass
                
        self.lbl_estado.config(text=f"Extrayendo contexto Sentinel-2 R20m para {fecha}...")
        self.root.update()
        
        datos_dir = os.path.join(DATASET_DIR, salar, "DATOS")
        fecha_fmt = f"{fecha[:4]}-{fecha[4:6]}-{fecha[6:]}"
        fecha_dir = os.path.join(datos_dir, fecha_fmt)
        
        safe_dirs = glob.glob(os.path.join(fecha_dir, "*.SAFE"))
        if not safe_dirs:
            safe_dirs = glob.glob(os.path.join(fecha_dir, "*", "*.SAFE"))
        if not safe_dirs:
            ref_dirs = [
                d for d in glob.glob(os.path.join(DATASET_DIR, salar, "REFERENCIA", "*.SAFE"))
                if os.path.isdir(d) and not os.path.basename(d).lower().startswith("manifest")
            ]
            for r in ref_dirs:
                if fecha in r:
                    safe_dirs = [r]
                    break
                    
        if not safe_dirs:
            return None, None
            
        safe_dir = safe_dirs[0]
        tcis = glob.glob(os.path.join(safe_dir, "**", "*_TCI_20m.jp2"), recursive=True)
        if not tcis:
            return None, None
        tci = tcis[0]
        
        scl_files = glob.glob(os.path.join(safe_dir, "**", "*_SCL_20m.jp2"), recursive=True)
        scl_path = scl_files[0] if scl_files else None
        
        nubes_act = self.obtener_porcentaje_nubes(safe_dir)
        
        min_x = min([x for (x, y) in self.coordenadas_map.values()])
        max_x = max([x for (x, y) in self.coordenadas_map.values()])
        min_y = min([y for (x, y) in self.coordenadas_map.values()])
        max_y = max([y for (x, y) in self.coordenadas_map.values()])
        
        MARGIN = 1024
        crop_x0 = max(0, min_x - MARGIN)
        crop_y0 = max(0, min_y - MARGIN)
        crop_x1 = min(5490, max_x + 256 + MARGIN)
        crop_y1 = min(5490, max_y + 256 + MARGIN)
        
        width = crop_x1 - crop_x0
        height = crop_y1 - crop_y0
        
        try:
            window = Window(crop_x0, crop_y0, width, height)
            with rasterio.open(tci) as src:
                rgb = src.read([1, 2, 3], window=window)
                rgb = np.moveaxis(rgb, 0, -1)
                
            img = Image.fromarray(rgb)
            img.save(cache_path)
            
            meta = {
                'offset_x': crop_x0, 
                'offset_y': crop_y0, 
                'tci_path': tci, 
                'scl_path': scl_path,
                'safe_name': os.path.basename(safe_dir),
                'nubes': nubes_act
            }
            with open(meta_path, 'w') as f:
                json.dump(meta, f)
                
            self.current_ctx_base = img
            self.current_meta = meta
            self.current_ctx_fecha = fecha
            return img, meta
        except Exception as e:
            print(f"Error extrayendo ventana R20m: {e}")
            return None, None

    def obtener_contexto_ref(self, salar):
        """Genera o carga el mapa recortado R20m de la escena con MENOR NUBOSIDAD de todo el salar"""
        if not rasterio or not self.coordenadas_map:
            return None, None
            
        if self.ref_ctx_base is not None and self.ref_meta is not None:
            return self.ref_ctx_base, self.ref_meta
            
        # Buscar automáticamente la escena más limpia (menor porcentaje de nubes)
        ref_dir, tci, nubes_ref, fecha_fmt = self.encontrar_escena_mas_limpia(salar)
        if not ref_dir or not tci:
            return None, None
            
        salar_clean = salar.replace(" ", "_")
        safe_name = os.path.basename(ref_dir)
        m_fecha = re.search(r'(\d{8})T', safe_name)
        fecha_tag = m_fecha.group(1) if m_fecha else "LIMPIA"
        
        cache_path = os.path.join(CACHE_DIR, f"{salar_clean}_{fecha_tag}_REF_LIMPIA_R20m_crop.png")
        meta_path = os.path.join(CACHE_DIR, f"{salar_clean}_{fecha_tag}_REF_LIMPIA_R20m_meta.json")
        
        if os.path.exists(cache_path) and os.path.exists(meta_path):
            try:
                with open(meta_path, 'r') as f:
                    meta = json.load(f)
                if 'scl_path' not in meta or not meta.get('scl_path'):
                    tci_p = meta.get('tci_path', '')
                    cand_scl = tci_p.replace('_TCI_20m.jp2', '_SCL_20m.jp2')
                    if os.path.exists(cand_scl):
                        meta['scl_path'] = cand_scl
                        try:
                            with open(meta_path, 'w') as f:
                                json.dump(meta, f)
                        except Exception:
                            pass
                img = Image.open(cache_path)
                self.ref_ctx_base = img
                self.ref_meta = meta
                return img, meta
            except Exception:
                pass
                
        n_str = f" ({nubes_ref:.1f}% nubes)" if nubes_ref is not None else ""
        self.lbl_estado.config(text=f"Extrayendo referencia más limpia para {salar}{n_str}...")
        self.root.update()
        
        scl_files = glob.glob(os.path.join(ref_dir, "**", "*_SCL_20m.jp2"), recursive=True)
        ref_scl_path = scl_files[0] if scl_files else None
        
        min_x = min([x for (x, y) in self.coordenadas_map.values()])
        max_x = max([x for (x, y) in self.coordenadas_map.values()])
        min_y = min([y for (x, y) in self.coordenadas_map.values()])
        max_y = max([y for (x, y) in self.coordenadas_map.values()])
        
        MARGIN = 1024
        crop_x0 = max(0, min_x - MARGIN)
        crop_y0 = max(0, min_y - MARGIN)
        crop_x1 = min(5490, max_x + 256 + MARGIN)
        crop_y1 = min(5490, max_y + 256 + MARGIN)
        
        width = crop_x1 - crop_x0
        height = crop_y1 - crop_y0
        
        try:
            window = Window(crop_x0, crop_y0, width, height)
            with rasterio.open(tci) as src:
                rgb = src.read([1, 2, 3], window=window)
                rgb = np.moveaxis(rgb, 0, -1)
                
            img = Image.fromarray(rgb)
            img.save(cache_path)
            
            meta = {
                'offset_x': crop_x0,
                'offset_y': crop_y0,
                'tci_path': tci,
                'scl_path': ref_scl_path,
                'safe_name': safe_name,
                'nubes': nubes_ref,
                'fecha_fmt': fecha_fmt
            }
            with open(meta_path, 'w') as f:
                json.dump(meta, f)
                
            self.ref_ctx_base = img
            self.ref_meta = meta
            return img, meta
        except Exception as e:
            print(f"Error extrayendo referencia limpia: {e}")
            return None, None

    def on_salar_selected(self, event):
        salar = self.combo_salar.get()
        if not salar: return
        
        salar_clean = salar.replace(" ", "_")
        ya_filtrado = os.path.exists(os.path.join(RESULTADOS_DIR, f"{salar_clean}_data_con_valor.txt"))
        progreso_path = os.path.join(RESULTADOS_DIR, f"{salar_clean}_progreso.json")
        
        if ya_filtrado:
            resp = messagebox.askyesno("Ya filtrado", f"El salar {salar} ya fue filtrado.\n¿Deseas empezar desde cero y sobrescribir los resultados anteriores?")
            if not resp:
                self.combo_salar.set('')
                return
            else:
                if os.path.exists(progreso_path): os.remove(progreso_path)
        
        self.salar_actual = salar
        self.current_ctx_base = None
        self.current_meta = None
        self.current_ctx_fecha = None
        self.ref_ctx_base = None
        self.ref_meta = None
        self.cargar_coordenadas(salar)
        self.cargar_imagenes(salar)
        
        self.aceptadas = []
        self.rechazadas = []
        self.idx_actual = 0
        
        if os.path.exists(progreso_path):
            try:
                with open(progreso_path, 'r', encoding='utf-8') as f: data = json.load(f)
                self.aceptadas = data.get('aceptadas', [])
                self.rechazadas = data.get('rechazadas', [])
                self.idx_actual = data.get('idx_actual', 0)
            except Exception:
                self.idx_actual = 0
        
        if self.lista_parches:
            if self.idx_actual >= len(self.lista_parches):
                self.finalizar_salar()
            else:
                self.mostrar_imagen_actual()
                self.btn_accept.config(state=tk.NORMAL)
                self.btn_reject.config(state=tk.NORMAL)
                self.actualizar_btn_deshacer()
        else:
            messagebox.showinfo("Vacío", f"No se encontraron imágenes procesadas para {salar}")
            self.lbl_estado.config(text="")
            self.btn_accept.config(state=tk.DISABLED)
            self.btn_reject.config(state=tk.DISABLED)
            self.btn_undo.config(state=tk.DISABLED)

    def cargar_imagenes(self, salar):
        img_dir = os.path.join(PROCESADO_DIR, salar, "imagen")
        pngs = sorted(glob.glob(os.path.join(img_dir, "*.png")))
        self.lista_parches = []
        
        for png_path in pngs:
            nombre = os.path.basename(png_path)
            match = re.search(r'_p(\d+)', nombre)
            if match:
                # Extraemos fecha del nombre (XXXXX_S2X_YYYYMMDD_pXX.png)
                match_fecha = re.search(r'_S2[ABC]_(\d{8})_p', nombre)
                fecha = match_fecha.group(1) if match_fecha else "00000000"
                
                parche = match.group(1)
                self.lista_parches.append({
                    'imagen': png_path,
                    'parche': parche,
                    'nombre': nombre,
                    'fecha': fecha,
                    'fecha_fmt': f"{fecha[:4]}-{fecha[4:6]}-{fecha[6:]}"
                })

    def actualizar_btn_deshacer(self):
        if self.idx_actual > 0: self.btn_undo.config(state=tk.NORMAL)
        else: self.btn_undo.config(state=tk.DISABLED)

    def deshacer(self):
        if self.idx_actual > 0:
            self.idx_actual -= 1
            item_a_deshacer = self.lista_parches[self.idx_actual]
            
            if self.aceptadas and self.aceptadas[-1]['nombre'] == item_a_deshacer['nombre']:
                self.aceptadas.pop()
            elif self.rechazadas and self.rechazadas[-1]['nombre'] == item_a_deshacer['nombre']:
                self.rechazadas.pop()
                
            self.guardar_progreso()
            self.mostrar_imagen_actual()
            self.actualizar_btn_deshacer()

    def mostrar_imagen_actual(self):
        if self.idx_actual >= len(self.lista_parches):
            self.finalizar_salar()
            return
            
        item = self.lista_parches[self.idx_actual]
        estado_texto = f"Fecha: {item['fecha_fmt']}  |  Parche: p{item['parche']}  |  Avance: {self.idx_actual + 1} / {len(self.lista_parches)}"
        self.lbl_estado.config(text=estado_texto)
        
        # Mostrar el nombre del archivo del parche
        self.lbl_filename.config(text=item['nombre'])
        
        self.root.update()
        
        # 1. MOSTRAR PARCHE (IZQUIERDA)
        try:
            img = Image.open(item['imagen'])
            img = img.resize(self.patch_disp_size, Image.Resampling.NEAREST)
            self.current_img_tk = ImageTk.PhotoImage(img)
            self.lbl_imagen.config(image=self.current_img_tk)
        except Exception as e:
            self.lbl_imagen.config(image='', text=f"Error parche:\n{e}", fg="red")
            
        # 2. CARGAR BASES DE CONTEXTO Y RENDERIZAR (DERECHA)
        self.obtener_contexto(self.salar_actual, item['fecha'])
        self.obtener_contexto_ref(self.salar_actual)
        self.actualizar_vista_contexto()

    def actualizar_vista_contexto(self):
        """Renderiza el panel derecho según el modo de vista (Doble Comparativa, Zoom o General)"""
        if self.current_ctx_base is None or self.current_meta is None or self.idx_actual >= len(self.lista_parches):
            return
            
        item = self.lista_parches[self.idx_actual]
        p_idx = int(item['parche'])
        
        if p_idx not in self.coordenadas_map:
            return
            
        try:
            x, y = self.coordenadas_map[p_idx]
            offset_x = self.current_meta['offset_x']
            offset_y = self.current_meta['offset_y']
            
            px = x - offset_x
            py = y - offset_y
            size = 256
            
            # Calcular nubosidad exacta en el parche actual (256x256) usando la banda SCL
            scl_path = self.current_meta.get('scl_path')
            pct_parche = self.obtener_nubes_parche(scl_path, x, y) if scl_path else None
            
            # Calcular nubosidad exacta en la referencia limpia para este mismo parche
            pct_ref_parche = None
            if self.ref_meta:
                ref_scl = self.ref_meta.get('scl_path')
                if not ref_scl:
                    tci_ref_p = self.ref_meta.get('tci_path', '')
                    cand_scl = tci_ref_p.replace('_TCI_20m.jp2', '_SCL_20m.jp2')
                    if os.path.exists(cand_scl):
                        ref_scl = cand_scl
                        self.ref_meta['scl_path'] = cand_scl
                if ref_scl:
                    pct_ref_parche = self.obtener_nubes_parche(ref_scl, x, y)
            
            ref_badge_info = f"   [Ref: {pct_ref_parche:.1f}%]" if pct_ref_parche is not None else ""
            badge_parche = ""
            if pct_parche is not None:
                if round(pct_parche, 1) == 0.0:
                    self.lbl_patch_clouds.config(text=f"🟢 NUBES EN PARCHE: 0.0% (DESPEJADO){ref_badge_info}", fg="#4ade80")
                    badge_parche = " | 🟢 Parche: 0.0% nubes"
                elif pct_parche < 5.0:
                    self.lbl_patch_clouds.config(text=f"🟡 NUBES EN PARCHE: {pct_parche:.1f}% (LEVE){ref_badge_info}", fg="#facc15")
                    badge_parche = f" | 🟡 Parche: {pct_parche:.1f}% nubes"
                else:
                    self.lbl_patch_clouds.config(text=f"⚠️ ALERTA: {pct_parche:.1f}% NUBES EN ESTE PARCHE{ref_badge_info}", fg="#ef4444")
                    badge_parche = f" | ⚠️ ALERTA: {pct_parche:.1f}% nubes"
            else:
                self.lbl_patch_clouds.config(text="", fg="#94a3b8")
            
            safe_act = self.current_meta.get('safe_name', '')
            
            if self.vista_modo == "doble":
                self.frame_single.pack_forget()
                self.frame_doble.pack(fill=tk.BOTH, expand=True)
                
                disp_w, disp_h = self.doble_disp_size
                PAD_Y = int(self.pad_cuadrantes * 256)
                crop_h = size + 2 * PAD_Y
                crop_w = int(crop_h * (disp_w / disp_h))
                
                # 1. IMAGEN FECHA ACTUAL (ARRIBA)
                nubes_act = self.current_meta.get('nubes')
                nubes_act_str = f" | ☁️ Escena: {nubes_act:.1f}%" if nubes_act is not None else ""
                self.lbl_act_title.config(text=f"📅 FECHA ACTUAL: {item['fecha_fmt']}{nubes_act_str}{badge_parche} | {safe_act}")
                
                cx = px + size // 2
                cy = py + size // 2
                sub_x0 = max(0, cx - crop_w // 2)
                sub_y0 = max(0, cy - crop_h // 2)
                sub_x1 = min(self.current_ctx_base.width, sub_x0 + crop_w)
                sub_y1 = min(self.current_ctx_base.height, sub_y0 + crop_h)
                if sub_x1 - sub_x0 < crop_w:
                    sub_x0 = max(0, sub_x1 - crop_w)
                if sub_y1 - sub_y0 < crop_h:
                    sub_y0 = max(0, sub_y1 - crop_h)
                
                ctx_crop_act = self.current_ctx_base.crop((sub_x0, sub_y0, sub_x1, sub_y1)).copy()
                d_act = ImageDraw.Draw(ctx_crop_act)
                
                for i, (c_x, c_y) in self.coordenadas_map.items():
                    cpx = c_x - offset_x - sub_x0
                    cpy = c_y - offset_y - sub_y0
                    if -size <= cpx <= ctx_crop_act.width and -size <= cpy <= ctx_crop_act.height:
                        if i != p_idx:
                            d_act.rectangle([cpx, cpy, cpx + size, cpy + size], outline="#06b6d4", width=2)
                            
                hpx = px - sub_x0
                hpy = py - sub_y0
                d_act.rectangle([hpx - 6, hpy - 6, hpx + size + 6, hpy + size + 6], outline="#facc15", width=5)
                d_act.rectangle([hpx, hpy, hpx + size, hpy + size], outline="#ef4444", width=7)
                
                img_act_disp = ctx_crop_act.resize((disp_w, disp_h), Image.Resampling.LANCZOS)
                self.current_act_tk = ImageTk.PhotoImage(img_act_disp)
                self.lbl_act_img.config(image=self.current_act_tk)
                
                # 2. IMAGEN REFERENCIA LIMPIA (ABAJO)
                if self.ref_ctx_base and self.ref_meta:
                    safe_ref = self.ref_meta.get('safe_name', '')
                    nubes_ref = self.ref_meta.get('nubes')
                    nubes_ref_str = f" | ☁️ Escena: {nubes_ref:.1f}%" if nubes_ref is not None else ""
                    fecha_ref = self.ref_meta.get('fecha_fmt', '')
                    fecha_ref_str = f": {fecha_ref}" if fecha_ref else ""
                    
                    badge_ref_parche = ""
                    ref_title_color = "#4ade80"
                    if pct_ref_parche is not None:
                        if round(pct_ref_parche, 1) == 0.0:
                            badge_ref_parche = " | 🟢 Parche: 0.0% nubes"
                            ref_title_color = "#4ade80"
                        elif pct_ref_parche < 5.0:
                            badge_ref_parche = f" | 🟡 Parche: {pct_ref_parche:.1f}% nubes"
                            ref_title_color = "#facc15"
                        else:
                            badge_ref_parche = f" | ⚠️ ALERTA: {pct_ref_parche:.1f}% nubes"
                            ref_title_color = "#ef4444"
                            
                    self.lbl_ref_title.config(
                        text=f"☀️ REFERENCIA MÁS LIMPIA{fecha_ref_str}{nubes_ref_str}{badge_ref_parche} | {safe_ref}",
                        fg=ref_title_color
                    )
                    
                    ref_offset_x = self.ref_meta['offset_x']
                    ref_offset_y = self.ref_meta['offset_y']
                    ref_px = x - ref_offset_x
                    ref_py = y - ref_offset_y
                    
                    ref_cx = ref_px + size // 2
                    ref_cy = ref_py + size // 2
                    ref_sub_x0 = max(0, ref_cx - crop_w // 2)
                    ref_sub_y0 = max(0, ref_cy - crop_h // 2)
                    ref_sub_x1 = min(self.ref_ctx_base.width, ref_sub_x0 + crop_w)
                    ref_sub_y1 = min(self.ref_ctx_base.height, ref_sub_y0 + crop_h)
                    if ref_sub_x1 - ref_sub_x0 < crop_w:
                        ref_sub_x0 = max(0, ref_sub_x1 - crop_w)
                    if ref_sub_y1 - ref_sub_y0 < crop_h:
                        ref_sub_y0 = max(0, ref_sub_y1 - crop_h)
                    
                    ctx_crop_ref = self.ref_ctx_base.crop((ref_sub_x0, ref_sub_y0, ref_sub_x1, ref_sub_y1)).copy()
                    d_ref = ImageDraw.Draw(ctx_crop_ref)
                    
                    for i, (c_x, c_y) in self.coordenadas_map.items():
                        cpx = c_x - ref_offset_x - ref_sub_x0
                        cpy = c_y - ref_offset_y - ref_sub_y0
                        if -size <= cpx <= ctx_crop_ref.width and -size <= cpy <= ctx_crop_ref.height:
                            if i != p_idx:
                                d_ref.rectangle([cpx, cpy, cpx + size, cpy + size], outline="#06b6d4", width=2)
                                
                    ref_hpx = ref_px - ref_sub_x0
                    ref_hpy = ref_py - ref_sub_y0
                    d_ref.rectangle([ref_hpx - 6, ref_hpy - 6, ref_hpx + size + 6, ref_hpy + size + 6], outline="#facc15", width=5)
                    d_ref.rectangle([ref_hpx, ref_hpy, ref_hpx + size, ref_hpy + size], outline="#22c55e", width=7)
                    
                    img_ref_disp = ctx_crop_ref.resize((disp_w, disp_h), Image.Resampling.LANCZOS)
                    self.current_ref_tk = ImageTk.PhotoImage(img_ref_disp)
                    self.lbl_ref_img.config(image=self.current_ref_tk)
                else:
                    self.lbl_ref_title.config(text="☀️ REFERENCIA LIMPIA: No encontrada")
                    self.lbl_ref_img.config(image='')
                    
            elif self.vista_modo == "zoom":
                self.frame_doble.pack_forget()
                self.frame_single.pack(fill=tk.BOTH, expand=True)
                
                nubes_act = self.current_meta.get('nubes')
                nubes_act_str = f" | ☁️ Escena: {nubes_act:.1f}%" if nubes_act is not None else ""
                self.lbl_single_title.config(text=f"📅 FECHA ACTUAL: {item['fecha_fmt']}{nubes_act_str}{badge_parche} | {safe_act}", fg="#facc15")
                
                PAD = int(self.pad_cuadrantes * 256)
                sub_x0 = max(0, px - PAD)
                sub_y0 = max(0, py - PAD)
                sub_x1 = min(self.current_ctx_base.width, px + size + PAD)
                sub_y1 = min(self.current_ctx_base.height, py + size + PAD)
                
                ctx_crop = self.current_ctx_base.crop((sub_x0, sub_y0, sub_x1, sub_y1)).copy()
                draw = ImageDraw.Draw(ctx_crop)
                
                for i, (cx, cy) in self.coordenadas_map.items():
                    cpx = cx - offset_x - sub_x0
                    cpy = cy - offset_y - sub_y0
                    if -size <= cpx <= ctx_crop.width and -size <= cpy <= ctx_crop.height:
                        if i != p_idx:
                            draw.rectangle([cpx, cpy, cpx + size, cpy + size], outline="#06b6d4", width=2)
                            
                hpx = px - sub_x0
                hpy = py - sub_y0
                draw.rectangle([hpx - 6, hpy - 6, hpx + size + 6, hpy + size + 6], outline="#facc15", width=6)
                draw.rectangle([hpx, hpy, hpx + size, hpy + size], outline="#ef4444", width=8)
                
                img_disp = ctx_crop.resize(self.single_disp_size, Image.Resampling.LANCZOS)
                self.current_single_tk = ImageTk.PhotoImage(img_disp)
                self.lbl_single_img.config(image=self.current_single_tk)
                
            else: # general
                self.frame_doble.pack_forget()
                self.frame_single.pack(fill=tk.BOTH, expand=True)
                
                self.lbl_single_title.config(text=f"🗺️ MAPA GENERAL DEL SALAR | {safe_act}", fg="#38bdf8")
                
                ctx_full = self.current_ctx_base.copy()
                draw = ImageDraw.Draw(ctx_full)
                
                for i, (cx, cy) in self.coordenadas_map.items():
                    cpx = cx - offset_x
                    cpy = cy - offset_y
                    if i != p_idx:
                        draw.rectangle([cpx, cpy, cpx + size, cpy + size], outline="#06b6d4", width=2)
                        
                draw.rectangle([px - 10, py - 10, px + size + 10, py + size + 10], outline="#facc15", width=6)
                draw.rectangle([px, py, px + size, py + size], outline="#ef4444", width=8)
                
                img_disp = ctx_full.resize(self.single_disp_size, Image.Resampling.LANCZOS)
                self.current_single_tk = ImageTk.PhotoImage(img_disp)
                self.lbl_single_img.config(image=self.current_single_tk)
                
        except Exception as e:
            print(f"Error renderizando contexto: {e}")

    def registrar_decision(self, decision):
        item = self.lista_parches[self.idx_actual]
        
        if decision == 'ACEPTADO': self.aceptadas.append(item)
        else: self.rechazadas.append(item)
            
        self.idx_actual += 1
        self.guardar_progreso()
        self.actualizar_btn_deshacer()
        self.mostrar_imagen_actual()

    def guardar_progreso(self):
        if not self.salar_actual: return
        salar_clean = self.salar_actual.replace(" ", "_")
        progreso_path = os.path.join(RESULTADOS_DIR, f"{salar_clean}_progreso.json")
        data = {'idx_actual': self.idx_actual, 'aceptadas': self.aceptadas, 'rechazadas': self.rechazadas}
        with open(progreso_path, 'w', encoding='utf-8') as f: json.dump(data, f)

    def agrupar_por_fecha(self, lista):
        agrupado = {}
        for item in lista:
            if item['fecha'] not in agrupado: agrupado[item['fecha']] = []
            agrupado[item['fecha']].append(item)
        return agrupado

    def finalizar_salar(self):
        salar_clean = self.salar_actual.replace(" ", "_")
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        aceptadas_por_fecha = self.agrupar_por_fecha(self.aceptadas)
        rechazadas_por_fecha = self.agrupar_por_fecha(self.rechazadas)
        todas_las_fechas = sorted(list(set(list(aceptadas_por_fecha.keys()) + list(rechazadas_por_fecha.keys()))))
        
        filtrado_path = os.path.join(RESULTADOS_DIR, f"{salar_clean}_data_con_valor.txt")
        with open(filtrado_path, "w", encoding="utf-8") as f:
            f.write(f"# Filtrado de: {self.salar_actual}\n")
            f.write(f"# Fecha de filtrado: {timestamp}\n")
            f.write(f"# Total parches aceptados: {len(self.aceptadas)}\n")
            f.write(f"# Formato: FECHA,PARCHES\n")
            f.write(f"#{'='*50}\n")
            for fecha in sorted(aceptadas_por_fecha.keys()):
                parches = aceptadas_por_fecha[fecha]
                coords_str = "; ".join([f"p{p['parche']}" for p in parches])
                f.write(f"{fecha},{coords_str}\n")
        
        historial_path = os.path.join(HISTORIAL_DIR, f"{salar_clean}_historial.txt")
        with open(historial_path, "w", encoding="utf-8") as f:
            f.write(f"{'='*60}\n")
            f.write(f" HISTORIAL DETALLADO DE FILTRADO - {self.salar_actual}\n")
            f.write(f" Fecha: {timestamp}\n")
            f.write(f"{'='*60}\n\n")
            for fecha in todas_las_fechas:
                fecha_fmt = f"{fecha[:4]}-{fecha[4:6]}-{fecha[6:]}"
                aceptados_aqui = aceptadas_por_fecha.get(fecha, [])
                rechazados_aqui = rechazadas_por_fecha.get(fecha, [])
                f.write(f"[{fecha_fmt}]\n")
                if aceptados_aqui:
                    coords = "; ".join([f"p{p['parche']}" for p in aceptados_aqui])
                    f.write(f"  ACEPTADOS ({len(aceptados_aqui)}): {coords}\n")
                if rechazados_aqui:
                    coords = "; ".join([f"p{p['parche']}" for p in rechazados_aqui])
                    f.write(f"  RECHAZADOS ({len(rechazados_aqui)}): {coords}\n")
                f.write("\n")
            f.write(f"{'='*60}\n")
            f.write(f" Resumen Final: {len(self.aceptadas)} parches aceptados, {len(self.rechazadas)} parches rechazados\n")
            f.write(f"{'='*60}\n")
            
        progreso_path = os.path.join(RESULTADOS_DIR, f"{salar_clean}_progreso.json")
        if os.path.exists(progreso_path): os.remove(progreso_path)
        
        messagebox.showinfo("Completado", f"El salar {self.salar_actual} ha sido filtrado.\nResultados guardados.")
        
        self.combo_salar.set('')
        self.salar_actual = None
        self.lbl_estado.config(text="Selecciona otro salar para continuar...")
        self.btn_accept.config(state=tk.DISABLED)
        self.btn_reject.config(state=tk.DISABLED)
        self.btn_undo.config(state=tk.DISABLED)
        self.lbl_imagen.config(image='')
        self.lbl_act_img.config(image='')
        self.lbl_ref_img.config(image='')
        self.lbl_single_img.config(image='')
        self.lbl_filename.config(text='')
        self.lbl_patch_clouds.config(text='')

if __name__ == "__main__":
    root = tk.Tk()
    app = FiltradorApp(root)
    root.mainloop()
