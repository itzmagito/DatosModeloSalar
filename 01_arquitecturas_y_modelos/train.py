"""
==============================================================================
ENTRENAMIENTO GENÉRICO U-NET / U-NET TRANSFORMER - TESIS SALARES
==============================================================================
Script de entrenamiento unificado que funciona para todas las variantes:
  - U-Net RGB (3 canales)
  - U-Net 4 Bandas RGB+NIR (4 canales)
  - U-Net Multiespectral (6 canales)
  - U-Net Transformer RGB/4B/6B

Características:
- Precisión Mixta (AMP FP16) para máxima velocidad en GPUs NVIDIA.
- Selección dinámica de la mejor GPU libre en servidores compartidos.
- Límite estricto de hilos CPU (4 hilos) para respetar a otros usuarios.
- Scheduler cíclico CosineAnnealingWarmRestarts para convergencia profunda.
- Evaluación automática en Test Set al finalizar.
- Inferencia automática (genera imágenes comparativas) al finalizar.
==============================================================================
"""

import os
import sys
import time
import csv
import argparse
import numpy as np

if sys.stdout and hasattr(sys.stdout, 'reconfigure'):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.amp import autocast, GradScaler

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
EXPERIMENTOS_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
COMPARTIDO_DIR = os.path.join(EXPERIMENTOS_DIR, "codigo_compartido")
sys.path.insert(0, os.getcwd())
sys.path.insert(0, COMPARTIDO_DIR)
sys.path.insert(0, SCRIPT_DIR)

from loss import BCEDiceLoss
from metricas import calcular_metricas


def graficar_curvas(historial, ruta_guardado):
    """Genera gráfico de evolución de pérdida y métricas (Train vs Val)."""
    epocas = [h['epoca'] for h in historial]
    train_loss = [h['train_loss'] for h in historial]
    val_loss = [h['val_loss'] for h in historial]
    train_iou = [h['train_iou'] for h in historial]
    val_iou = [h['val_iou'] for h in historial]
    train_dice = [h['train_dice'] for h in historial]
    val_dice = [h['val_dice'] for h in historial]

    fig, axes = plt.subplots(1, 3, figsize=(18, 5), dpi=150)
    fig.patch.set_facecolor('#FFFFFF')

    axes[0].plot(epocas, train_loss, label='Train Loss', color='#2563EB', linewidth=2)
    axes[0].plot(epocas, val_loss, label='Val Loss', color='#DC2626', linewidth=2, linestyle='--')
    axes[0].set_title('Pérdida (BCE + Dice Loss)', fontsize=12, fontweight='bold')
    axes[0].set_xlabel('Época')
    axes[0].set_ylabel('Loss')
    axes[0].grid(True, linestyle=':', alpha=0.6)
    axes[0].legend()

    axes[1].plot(epocas, train_iou, label='Train IoU', color='#2563EB', linewidth=2)
    axes[1].plot(epocas, val_iou, label='Val IoU', color='#16A34A', linewidth=2, linestyle='--')
    axes[1].set_title('Métrica IoU (Jaccard Index)', fontsize=12, fontweight='bold')
    axes[1].set_xlabel('Época')
    axes[1].set_ylabel('IoU')
    axes[1].grid(True, linestyle=':', alpha=0.6)
    axes[1].legend()

    axes[2].plot(epocas, train_dice, label='Train Dice', color='#2563EB', linewidth=2)
    axes[2].plot(epocas, val_dice, label='Val Dice', color='#0D9488', linewidth=2, linestyle='--')
    axes[2].set_title('Coeficiente Dice (F1-Score)', fontsize=12, fontweight='bold')
    axes[2].set_xlabel('Época')
    axes[2].set_ylabel('Dice')
    axes[2].grid(True, linestyle=':', alpha=0.6)
    axes[2].legend()

    plt.tight_layout()
    plt.savefig(ruta_guardado, bbox_inches='tight')
    plt.close()


def entrenar_una_epoca(modelo, loader, optimizer, criterion, scaler, device, use_amp=True):
    modelo.train()
    total_loss = 0.0
    metricas_acum = {'iou': 0.0, 'dice': 0.0, 'precision': 0.0, 'recall': 0.0, 'accuracy': 0.0}
    num_batches = len(loader)

    for i, batch in enumerate(loader):
        imgs = batch['image'].to(device, non_blocking=True)
        masks = batch['mask'].to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        if use_amp:
            with autocast('cuda'):
                logits = modelo(imgs)
                loss = criterion(logits, masks)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(modelo.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            logits = modelo(imgs)
            loss = criterion(logits, masks)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(modelo.parameters(), max_norm=1.0)
            optimizer.step()

        total_loss += loss.item()
        with torch.no_grad():
            m = calcular_metricas(logits, masks)
            for k in metricas_acum:
                metricas_acum[k] += m[k]

        if (i + 1) % 20 == 0 or (i + 1) == num_batches:
            print(f"    Paso [{i+1:>3}/{num_batches}] | Loss: {loss.item():.4f} | IoU: {m['iou']:.4f} | Dice: {m['dice']:.4f}", end='\r')

    print()
    res = {k: v / num_batches for k, v in metricas_acum.items()}
    res['loss'] = total_loss / num_batches
    return res


def validar_modelo(modelo, loader, criterion, device, use_amp=True):
    modelo.eval()
    total_loss = 0.0
    metricas_acum = {'iou': 0.0, 'dice': 0.0, 'precision': 0.0, 'recall': 0.0, 'accuracy': 0.0}
    num_batches = len(loader)

    with torch.no_grad():
        for batch in loader:
            imgs = batch['image'].to(device, non_blocking=True)
            masks = batch['mask'].to(device, non_blocking=True)

            if use_amp:
                with autocast('cuda'):
                    logits = modelo(imgs)
                    loss = criterion(logits, masks)
            else:
                logits = modelo(imgs)
                loss = criterion(logits, masks)

            total_loss += loss.item()
            m = calcular_metricas(logits, masks)
            for k in metricas_acum:
                metricas_acum[k] += m[k]

    res = {k: v / num_batches for k, v in metricas_acum.items()}
    res['loss'] = total_loss / num_batches
    return res


def seleccionar_mejor_gpu():
    if not torch.cuda.is_available():
        return torch.device('cpu')

    n_gpus = torch.cuda.device_count()
    if n_gpus == 1:
        return torch.device('cuda:0')

    print(f"  [Auto-GPU] Escaneando {n_gpus} GPUs disponibles:")
    mejor_idx = 0
    max_libre = -1
    for i in range(n_gpus):
        libre_b, total_b = torch.cuda.mem_get_info(i)
        libre_gb = libre_b / (1024 ** 3)
        total_gb = total_b / (1024 ** 3)
        print(f"    GPU {i} ({torch.cuda.get_device_name(i)}): {libre_gb:.1f} GB libres de {total_gb:.1f} GB")
        if libre_b > max_libre:
            max_libre = libre_b
            mejor_idx = i

    print(f"  [Auto-GPU] Seleccionada GPU {mejor_idx} ({max_libre / (1024**3):.1f} GB libres).")
    return torch.device(f'cuda:{mejor_idx}')


def fijar_semilla(seed):
    """Garantiza reproducibilidad determinística entre réplicas independientes."""
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    print(f"  [Reproducibilidad] Semilla aleatoria fijada en: {seed}")


def main():
    parser = argparse.ArgumentParser(description="Entrenamiento Genérico U-Net / U-Net Transformer")

    # Modelo
    parser.add_argument("--modelo-tipo", type=str, required=True,
                        choices=["unet", "transformer", "attention", "residual", "salar"],
                        help="Tipo de arquitectura: unet, transformer, attention, residual o salar")
    parser.add_argument("--in-channels", type=int, required=True,
                        help="Número de canales de entrada (3=RGB, 4=RGB+NIR, 6=Multi)")
    parser.add_argument("--dataset-tipo", type=str, required=True,
                        choices=["rgb", "4bandas", "multiespectral"],
                        help="Tipo de dataset a cargar")

    # Entrenamiento
    parser.add_argument("--epochs", type=int, default=350, help="Número de épocas")
    parser.add_argument("--batch-size", type=int, default=32, help="Tamaño de lote")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate inicial")
    parser.add_argument("--base-channels", type=int, default=64, help="Canales base U-Net")
    parser.add_argument("--seed", type=int, default=42, help="Semilla aleatoria para reproducibilidad")
    parser.add_argument("--no-amp", action="store_true", help="Desactivar AMP")
    parser.add_argument("--test-run", action="store_true", help="Prueba rápida de 1 época")

    # Transformer específico
    parser.add_argument("--transformer-heads", type=int, default=8)
    parser.add_argument("--transformer-layers", type=int, default=2)
    parser.add_argument("--transformer-ff-dim", type=int, default=2048)
    parser.add_argument("--transformer-dropout", type=float, default=0.1)

    # Servidor
    parser.add_argument("--gpu", type=int, default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--save-freq", type=int, default=50)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=4)

    # Datos
    parser.add_argument("--data-dir", type=str, required=True,
                        help="Ruta al directorio de datos (DATASET PROCESADO o DATASET_MULTIESPECTRAL)")

    # Salida
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Directorio de salida para checkpoints y resultados")

    args = parser.parse_args()

    # Fijar semilla aleatoria
    fijar_semilla(args.seed)

    # Configuración de hilos CPU
    if args.threads > 0:
        torch.set_num_threads(args.threads)
        os.environ["OMP_NUM_THREADS"] = str(args.threads)
        os.environ["MKL_NUM_THREADS"] = str(args.threads)

    # GPU
    if args.gpu is not None and torch.cuda.is_available():
        device = torch.device(f'cuda:{args.gpu}')
        print(f"  [GPU] Usando GPU {args.gpu} especificada manualmente.")
    else:
        device = seleccionar_mejor_gpu()

    use_amp = not args.no_amp and device.type == 'cuda'

    # Nombre del modelo para logs
    tipo_bandas = {3: "RGB (3 bandas)", 4: "4 Bandas (RGB+NIR)", 6: "Multiespectral (6 bandas)"}
    nombres_modelo = {
        'unet': 'U-Net',
        'transformer': 'U-Net Transformer',
        'attention': 'Attention U-Net',
        'residual': 'ResUNet',
        'salar': 'Salar-UNet'
    }
    nombre_modelo = f"{nombres_modelo.get(args.modelo_tipo, args.modelo_tipo)} {tipo_bandas.get(args.in_channels, f'{args.in_channels} canales')}"

    epochs_to_run = 1 if args.test_run else args.epochs

    print("=" * 75)
    print(f" ENTRENAMIENTO {nombre_modelo.upper()}")
    print(f" Dispositivo: {device} ({torch.cuda.get_device_name(device) if device.type == 'cuda' else 'CPU'})")
    print(f" Hilos CPU: {torch.get_num_threads()} | AMP: {'Sí' if use_amp else 'No'}")
    print(f" Épocas: {epochs_to_run} | Batch: {args.batch_size} | LR: {args.lr}")
    if args.modelo_tipo == 'transformer':
        print(f" Transformer: heads={args.transformer_heads}, layers={args.transformer_layers}, ff={args.transformer_ff_dim}")
    print("=" * 75)

    # Directorio de salida
    if args.output_dir:
        output_dir = args.output_dir
    else:
        output_dir = os.path.join(SCRIPT_DIR, "resultados", f"{epochs_to_run}_epocas")
    os.makedirs(output_dir, exist_ok=True)
    checkpoints_dir = os.path.join(output_dir, "checkpoints")
    os.makedirs(checkpoints_dir, exist_ok=True)

    # 1. Dataset
    print(f"\n[1/4] Cargando dataset {args.dataset_tipo}...")
    if args.dataset_tipo == "rgb":
        from dataset import obtener_dataloaders
        train_loader, val_loader, test_loader = obtener_dataloaders(
            batch_size=args.batch_size, num_workers=args.num_workers, procesado_dir=args.data_dir)
    elif args.dataset_tipo == "4bandas":
        from dataset import obtener_dataloaders
        train_loader, val_loader, test_loader = obtener_dataloaders(
            batch_size=args.batch_size, num_workers=args.num_workers, multi_dir=args.data_dir)
    elif args.dataset_tipo == "multiespectral":
        from dataset import obtener_dataloaders
        train_loader, val_loader, test_loader = obtener_dataloaders(
            batch_size=args.batch_size, num_workers=args.num_workers, multi_dir=args.data_dir)

    print(f"  Muestras: Train={len(train_loader.dataset)}, Val={len(val_loader.dataset)}, Test={len(test_loader.dataset)}")
    print(f"  Batches por época: {len(train_loader)}")

    # 2. Modelo
    print(f"\n[2/4] Inicializando {nombre_modelo}...")
    if args.modelo_tipo == "unet":
        from unet import UNet
        modelo = UNet(in_channels=args.in_channels, out_channels=1, base_channels=args.base_channels).to(device)
    elif args.modelo_tipo == "transformer":
        from unet_transformer import UNetTransformer
        modelo = UNetTransformer(
            in_channels=args.in_channels, out_channels=1, base_channels=args.base_channels,
            transformer_heads=args.transformer_heads, transformer_layers=args.transformer_layers,
            transformer_ff_dim=args.transformer_ff_dim, transformer_dropout=args.transformer_dropout
        ).to(device)
    elif args.modelo_tipo == "attention":
        from unet_attention import AttentionUNet
        modelo = AttentionUNet(
            in_channels=args.in_channels, out_channels=1, base_channels=args.base_channels
        ).to(device)
    elif args.modelo_tipo == "residual":
        from unet_residual import ResUNet
        modelo = ResUNet(
            in_channels=args.in_channels, out_channels=1, base_channels=args.base_channels
        ).to(device)
    elif args.modelo_tipo == "salar":
        from unet_salar import SalarUNet
        modelo = SalarUNet(
            in_channels=args.in_channels, out_channels=1, base_channels=args.base_channels,
            transformer_heads=args.transformer_heads, transformer_layers=args.transformer_layers,
            transformer_ff_dim=args.transformer_ff_dim, transformer_dropout=args.transformer_dropout
        ).to(device)

    print(f"  Parámetros entrenables: {modelo.contar_parametros():,}")
    if hasattr(modelo, 'contar_parametros_transformer'):
        print(f"  Parámetros Transformer: {modelo.contar_parametros_transformer():,}")

    # 3. Optimizador y Pérdida
    criterion = BCEDiceLoss(alpha=1.0, beta=1.0)
    optimizer = torch.optim.AdamW(modelo.parameters(), lr=args.lr, weight_decay=1e-4)

    if epochs_to_run >= 50:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=50, T_mult=2, eta_min=1e-6)
    else:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs_to_run, eta_min=1e-6)

    scaler = GradScaler('cuda') if use_amp else None

    # 4. Bucle de entrenamiento
    print(f"\n[3/4] Iniciando entrenamiento ({epochs_to_run} épocas)...")
    start_epoch = 1
    mejor_val_iou = 0.0
    historial = []
    csv_file = os.path.join(output_dir, "historial_entrenamiento.csv")
    curvas_img = os.path.join(output_dir, "curvas_entrenamiento.png")

    last_path = os.path.join(checkpoints_dir, "last_model.pth")
    if args.resume and os.path.exists(last_path):
        print(f"  [Resume] Cargando desde {last_path}...")
        ckpt = torch.load(last_path, map_location=device, weights_only=True)
        modelo.load_state_dict(ckpt['modelo_state_dict'])
        if 'optimizer_state_dict' in ckpt:
            try:
                optimizer.load_state_dict(ckpt['optimizer_state_dict'])
            except Exception:
                pass
        start_epoch = ckpt['epoca'] + 1
        mejor_val_iou = ckpt.get('val_iou', 0.0)
        print(f"  [Resume] Desde época {start_epoch} (Mejor Val IoU: {mejor_val_iou:.4f})")

    tiempo_inicio = time.time()

    for epoca in range(start_epoch, epochs_to_run + 1):
        t0 = time.time()
        print(f"\n--- Época [{epoca:>3}/{epochs_to_run}] (LR: {optimizer.param_groups[0]['lr']:.2e}) ---")

        res_train = entrenar_una_epoca(modelo, train_loader, optimizer, criterion, scaler, device, use_amp)
        res_val = validar_modelo(modelo, val_loader, criterion, device, use_amp)
        scheduler.step()

        dt = time.time() - t0
        print(f"  Resultados Época {epoca} ({dt:.1f}s):")
        print(f"    Train -> Loss: {res_train['loss']:.4f} | IoU: {res_train['iou']:.4f} | Dice: {res_train['dice']:.4f}")
        print(f"    Val   -> Loss: {res_val['loss']:.4f} | IoU: {res_val['iou']:.4f} | Dice: {res_val['dice']:.4f} | Prec: {res_val['precision']:.4f} | Rec: {res_val['recall']:.4f}")

        # Guardar mejor modelo
        if res_val['iou'] > mejor_val_iou:
            mejor_val_iou = res_val['iou']
            best_path = os.path.join(checkpoints_dir, "best_model.pth")
            torch.save({
                'epoca': epoca,
                'modelo_state_dict': modelo.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_iou': res_val['iou'],
                'val_dice': res_val['dice'],
                'val_loss': res_val['loss'],
                'in_channels': args.in_channels,
                'base_channels': args.base_channels,
                'modelo_tipo': args.modelo_tipo,
                'dataset_tipo': args.dataset_tipo
            }, best_path)
            print(f"    [*] Nuevo mejor modelo! (Val IoU: {mejor_val_iou:.4f})")

        # Guardar último modelo
        torch.save({
            'epoca': epoca,
            'modelo_state_dict': modelo.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'val_iou': res_val['iou'],
            'val_dice': res_val['dice'],
            'val_loss': res_val['loss'],
            'in_channels': args.in_channels,
            'base_channels': args.base_channels,
            'modelo_tipo': args.modelo_tipo,
            'dataset_tipo': args.dataset_tipo
        }, last_path)

        # Checkpoints periódicos
        if args.save_freq > 0 and epoca % args.save_freq == 0 and epoca < epochs_to_run:
            periodic_path = os.path.join(checkpoints_dir, f"checkpoint_epoch_{epoca}.pth")
            torch.save({
                'epoca': epoca,
                'modelo_state_dict': modelo.state_dict(),
                'val_iou': res_val['iou'],
                'val_dice': res_val['dice'],
                'in_channels': args.in_channels,
                'base_channels': args.base_channels,
                'modelo_tipo': args.modelo_tipo
            }, periodic_path)
            print(f"    [Checkpoint] Guardado: checkpoint_epoch_{epoca}.pth")

        registro = {
            'epoca': epoca,
            'train_loss': res_train['loss'],
            'train_iou': res_train['iou'],
            'train_dice': res_train['dice'],
            'val_loss': res_val['loss'],
            'val_iou': res_val['iou'],
            'val_dice': res_val['dice'],
            'val_precision': res_val['precision'],
            'val_recall': res_val['recall'],
            'val_accuracy': res_val['accuracy'],
            'lr': optimizer.param_groups[0]['lr'],
            'duracion_seg': round(dt, 2)
        }
        historial.append(registro)

        with open(csv_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=registro.keys())
            writer.writeheader()
            writer.writerows(historial)

        if len(historial) >= 2:
            graficar_curvas(historial, curvas_img)

    duracion_total = time.time() - tiempo_inicio
    print("\n" + "=" * 75)
    print(f" [OK] ENTRENAMIENTO {nombre_modelo.upper()} COMPLETADO")
    print(f" Duración total: {duracion_total/60:.2f} minutos")
    print(f" Mejor Val IoU: {mejor_val_iou:.4f}")
    print(f" Resultados en: {output_dir}")
    print("=" * 75)

    # 5. Evaluación en Test Set
    if not args.test_run and os.path.exists(os.path.join(checkpoints_dir, "best_model.pth")):
        print(f"\n[4/4] Evaluando {nombre_modelo} sobre TEST SET...")
        checkpoint = torch.load(os.path.join(checkpoints_dir, "best_model.pth"), weights_only=True)
        modelo.load_state_dict(checkpoint['modelo_state_dict'])
        res_test = validar_modelo(modelo, test_loader, criterion, device, use_amp)
        print(f"  RESULTADOS EN TEST SET ({nombre_modelo}):")
        print(f"    Loss:      {res_test['loss']:.4f}")
        print(f"    IoU:       {res_test['iou']:.4f}")
        print(f"    Dice (F1): {res_test['dice']:.4f}")
        print(f"    Precisión: {res_test['precision']:.4f}")
        print(f"    Recall:    {res_test['recall']:.4f}")
        print(f"    Accuracy:  {res_test['accuracy']:.4f}")

        # Guardar métricas de test en un archivo
        test_file = os.path.join(output_dir, "resultados_test.txt")
        with open(test_file, 'w', encoding='utf-8') as f:
            f.write(f"Modelo: {nombre_modelo}\n")
            f.write(f"Semilla: {args.seed}\n")
            f.write(f"Épocas: {epochs_to_run}\n")
            f.write(f"Duración: {duracion_total/60:.2f} minutos\n")
            f.write(f"Mejor Val IoU: {mejor_val_iou:.4f}\n")
            f.write(f"\n=== TEST SET ===\n")
            f.write(f"Loss:      {res_test['loss']:.4f}\n")
            f.write(f"IoU:       {res_test['iou']:.4f}\n")
            f.write(f"Dice (F1): {res_test['dice']:.4f}\n")
            f.write(f"Precisión: {res_test['precision']:.4f}\n")
            f.write(f"Recall:    {res_test['recall']:.4f}\n")
            f.write(f"Accuracy:  {res_test['accuracy']:.4f}\n")

    print("=" * 75)


if __name__ == "__main__":
    main()
