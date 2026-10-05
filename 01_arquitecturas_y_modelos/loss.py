"""
==============================================================================
FUNCIONES DE PÉRDIDA PARA SEGMENTACIÓN SEMÁNTICA BINARIA
==============================================================================
Combina Binary Cross Entropy (BCE) con Dice Loss:
- BCE penaliza errores individuales a nivel de pixel y garantiza estabilidad
  numérica y gradientes suaves.
- Dice Loss optimiza directamente la métrica de solapamiento regional y combate
  el desbalance de clases (superficie salina vs no salina).
==============================================================================
"""

import torch
import torch.nn as nn


class DiceLoss(nn.Module):
    """
    Dice Loss para segmentación binaria.
    Entrada: logits sin aplicar Sigmoid.
    """
    def __init__(self, smooth=1.0):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, logits, targets):
        probs = torch.sigmoid(logits)
        probs_flat = probs.view(-1)
        targets_flat = targets.view(-1)
        intersection = (probs_flat * targets_flat).sum()
        dice = (2.0 * intersection + self.smooth) / (probs_flat.sum() + targets_flat.sum() + self.smooth)
        return 1.0 - dice


class BCEDiceLoss(nn.Module):
    """
    Pérdida híbrida: alfa * BCEWithLogitsLoss + beta * DiceLoss
    """
    def __init__(self, alpha=1.0, beta=1.0, smooth=1.0):
        super(BCEDiceLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.bce = nn.BCEWithLogitsLoss()
        self.dice = DiceLoss(smooth=smooth)

    def forward(self, logits, targets):
        loss_bce = self.bce(logits, targets)
        loss_dice = self.dice(logits, targets)
        return self.alpha * loss_bce + self.beta * loss_dice


if __name__ == "__main__":
    criterio = BCEDiceLoss()
    logits = torch.randn(4, 1, 256, 256)
    targets = torch.randint(0, 2, (4, 1, 256, 256)).float()
    loss = criterio(logits, targets)
    print("=== PRUEBA BCEDiceLoss ===")
    print(f"Loss calculado: {loss.item():.4f}")
    assert loss.item() > 0, "Error en cálculo de pérdida"
    print("Test passed con éxito!")
