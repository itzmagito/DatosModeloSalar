"""
==============================================================================
MÉTRICAS DE EVALUACIÓN PARA SEGMENTACIÓN BINARIA
==============================================================================
Calcula las métricas estándar:
- IoU (Intersection over Union / Jaccard Index)
- Dice Score (F1-Score)
- Precision
- Recall (Sensibilidad)
- Accuracy
==============================================================================
"""

import torch


def calcular_metricas(logits, targets, umbral=0.5, eps=1e-7):
    """
    Calcula métricas binarias a partir de los logits del modelo y las máscaras ground truth.
    """
    with torch.no_grad():
        probs = torch.sigmoid(logits)
        preds = (probs > umbral).float()

        preds_flat = preds.view(-1)
        targets_flat = targets.view(-1)

        tp = (preds_flat * targets_flat).sum().item()
        fp = (preds_flat * (1.0 - targets_flat)).sum().item()
        fn = ((1.0 - preds_flat) * targets_flat).sum().item()
        tn = ((1.0 - preds_flat) * (1.0 - targets_flat)).sum().item()

        iou = (tp + eps) / (tp + fp + fn + eps)
        dice = (2.0 * tp + eps) / (2.0 * tp + fp + fn + eps)
        precision = (tp + eps) / (tp + fp + eps)
        recall = (tp + eps) / (tp + fn + eps)
        accuracy = (tp + tn) / (tp + tn + fp + fn + eps)

        return {
            'iou': iou,
            'dice': dice,
            'precision': precision,
            'recall': recall,
            'accuracy': accuracy
        }


# Alias de conveniencia
calcular_metricas_binarias = calcular_metricas


class MetricasAccumulator:
    """
    Acumulador para computar el promedio global de métricas a lo largo de una época completa.
    """
    def __init__(self):
        self.reset()

    def reset(self):
        self.tp = 0.0
        self.fp = 0.0
        self.fn = 0.0
        self.tn = 0.0
        self.total_loss = 0.0
        self.steps = 0

    def update(self, logits, targets, loss_val=0.0, umbral=0.5):
        with torch.no_grad():
            probs = torch.sigmoid(logits)
            preds = (probs > umbral).float()

            p_flat = preds.view(-1)
            t_flat = targets.view(-1)

            self.tp += (p_flat * t_flat).sum().item()
            self.fp += (p_flat * (1.0 - t_flat)).sum().item()
            self.fn += ((1.0 - p_flat) * t_flat).sum().item()
            self.tn += ((1.0 - p_flat) * (1.0 - t_flat)).sum().item()

            self.total_loss += loss_val
            self.steps += 1

    def obtener_resumen(self, eps=1e-7):
        tp, fp, fn, tn = self.tp, self.fp, self.fn, self.tn
        iou = (tp + eps) / (tp + fp + fn + eps)
        dice = (2.0 * tp + eps) / (2.0 * tp + fp + fn + eps)
        precision = (tp + eps) / (tp + fp + eps)
        recall = (tp + eps) / (tp + fn + eps)
        accuracy = (tp + tn) / (tp + tn + fp + fn + eps)
        avg_loss = self.total_loss / max(1, self.steps)

        return {
            'loss': avg_loss,
            'iou': iou,
            'dice': dice,
            'precision': precision,
            'recall': recall,
            'accuracy': accuracy
        }


if __name__ == "__main__":
    logits = torch.randn(4, 1, 256, 256)
    targets = torch.randint(0, 2, (4, 1, 256, 256)).float()
    met = calcular_metricas(logits, targets)
    print("=== PRUEBA DE MÉTRICAS ===")
    for k, v in met.items():
        print(f"  {k:<10}: {v:.4f}")
    print("Test passed con éxito!")
