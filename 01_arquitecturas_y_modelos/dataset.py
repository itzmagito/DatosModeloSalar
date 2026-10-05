"""
==============================================================================
DATASET MULTIESPECTRAL (6 CANALES) PARA SALARES
==============================================================================
Carga GeoTIFF de 6 bandas (B02, B03, B04, B08, NDVI, SI) en float32.
Augmentación geométrica ortogonal C4 (0°, 90°, 180°, 270°).
==============================================================================
"""

import os
import glob
import json
import numpy as np
import rasterio
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
COMPARTIDO_DIR = os.path.join(os.path.dirname(SCRIPT_DIR), "codigo_compartido")
SPLITS_FILE = os.path.join(COMPARTIDO_DIR, "splits.json")


def recolectar_pares(multi_dir, salares=None):
    """Escanea DATASET_MULTIESPECTRAL/ y retorna lista de tuplas."""
    pares = []
    if not os.path.exists(multi_dir):
        print(f"[ERROR] No existe {multi_dir}")
        return pares

    todos_salares = sorted([
        d for d in os.listdir(multi_dir)
        if os.path.isdir(os.path.join(multi_dir, d))
    ])

    if salares:
        todos_salares = [s for s in todos_salares if s in salares]

    for salar in todos_salares:
        img_dir = os.path.join(multi_dir, salar, "imagen")
        mask_dir = os.path.join(multi_dir, salar, "mascara")

        if not os.path.exists(img_dir) or not os.path.exists(mask_dir):
            continue

        tifs = sorted(glob.glob(os.path.join(img_dir, "*.tif")))
        for p_img in tifs:
            fname = os.path.basename(p_img)
            nombre_base = os.path.splitext(fname)[0]
            p_mask = os.path.join(mask_dir, f"{nombre_base}.png")
            if not os.path.exists(p_mask):
                p_mask = os.path.join(mask_dir, f"{nombre_base}.tif")

            if os.path.exists(p_mask):
                pares.append((p_img, p_mask, salar, fname))

    return pares


def cargar_splits(multi_dir):
    """Carga la partición exacta desde splits.json."""
    todos = recolectar_pares(multi_dir)
    mapa = {os.path.splitext(p[3])[0]: p for p in todos}

    if not os.path.exists(SPLITS_FILE):
        raise FileNotFoundError(f"No se encontró {SPLITS_FILE}")

    with open(SPLITS_FILE, 'r', encoding='utf-8') as f:
        splits = json.load(f)

    def resolver(lista):
        res = []
        for n in lista:
            base = os.path.splitext(n)[0]
            if base in mapa:
                res.append(mapa[base])
        return res

    train_p = resolver(splits['train'])
    val_p = resolver(splits['val'])
    test_p = resolver(splits['test'])

    print(f"Splits Multiespectral cargados: Train={len(train_p)}, Val={len(val_p)}, Test={len(test_p)}")
    return train_p, val_p, test_p


class SalarMultiespectralDataset(Dataset):
    """Dataset PyTorch para parches de 6 canales."""
    def __init__(self, lista_pares, augment=False):
        self.pares = lista_pares
        self.augment = augment

    def __len__(self):
        return len(self.pares)

    def __getitem__(self, idx):
        ruta_img, ruta_mask, salar, fname = self.pares[idx]

        # 1. Cargar imagen multiespectral (6 bandas)
        with rasterio.open(ruta_img) as src:
            img_arr = src.read().astype(np.float32)  # (6, 256, 256)

        # 2. Cargar máscara binaria
        if ruta_mask.endswith('.tif'):
            with rasterio.open(ruta_mask) as src:
                mask_arr = (src.read(1) > 127).astype(np.float32)
        else:
            mask = Image.open(ruta_mask).convert('L')
            mask_arr = (np.array(mask, dtype=np.float32) > 127.0).astype(np.float32)

        # 3. Tensores
        img_tensor = torch.from_numpy(img_arr)
        mask_tensor = torch.from_numpy(mask_arr).unsqueeze(0)

        # 4. Augmentación geométrica ortogonal C4
        if self.augment:
            k_rot = int(torch.randint(0, 4, (1,)).item())
            if k_rot > 0:
                img_tensor = torch.rot90(img_tensor, k=k_rot, dims=[1, 2])
                mask_tensor = torch.rot90(mask_tensor, k=k_rot, dims=[1, 2])

        return {
            'image': img_tensor,
            'mask': mask_tensor,
            'name': fname,
            'salar': salar
        }


def obtener_dataloaders(batch_size=16, num_workers=0, multi_dir=None):
    """Crea (train_loader, val_loader, test_loader) para 6 canales."""
    if multi_dir is None:
        raise ValueError("Debe especificar multi_dir")

    train_p, val_p, test_p = cargar_splits(multi_dir)

    ds_train = SalarMultiespectralDataset(train_p, augment=True)
    ds_val = SalarMultiespectralDataset(val_p, augment=False)
    ds_test = SalarMultiespectralDataset(test_p, augment=False)

    loader_train = DataLoader(ds_train, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    loader_val = DataLoader(ds_val, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    loader_test = DataLoader(ds_test, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

    return loader_train, loader_val, loader_test
