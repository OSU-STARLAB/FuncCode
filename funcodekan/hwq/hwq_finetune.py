import torch
import torch.nn as nn
from .quantizers import fake_quant_codebook

class CodebookFakeQuantWrapper(nn.Module):
    """
    Wrap a clustered PolyKAN-like model and fake-quantize its clustered codebooks during fine-tuning.

    Expected model attributes:
        model.clustered_weights: nn.ParameterList of [K, coeff_dim] tensors.
    """
    def __init__(self, model: nn.Module, bits: int = 4, granularity: str = "per_vector"):
        super().__init__()
        self.model = model
        self.bits = bits
        self.granularity = granularity

    def forward(self, *args, **kwargs):
        originals = []
        try:
            if hasattr(self.model, "clustered_weights"):
                for p in self.model.clustered_weights:
                    originals.append(p.data.clone())
                    p.data.copy_(fake_quant_codebook(p, self.bits, self.granularity).data)
            return self.model(*args, **kwargs)
        finally:
            if hasattr(self.model, "clustered_weights"):
                for p, old in zip(self.model.clustered_weights, originals):
                    p.data.copy_(old)

def hardware_aware_codebook_regularizer(model: nn.Module, target: str = "fpga_bram") -> torch.Tensor:
    """
    Lightweight regularizer encouraging hardware-friendly codebooks.

    Current terms:
    - small centroid magnitude,
    - smooth adjacent coefficient changes, useful for spline/RBF/Gram coefficient vectors.
    """
    reg = None
    if not hasattr(model, "clustered_weights"):
        return torch.tensor(0.0)

    for cb in model.clustered_weights:
        mag = cb.pow(2).mean()
        smooth = (cb[:, 1:] - cb[:, :-1]).pow(2).mean() if cb.shape[-1] > 1 else torch.zeros_like(mag)
        term = 1e-4 * mag + 1e-4 * smooth
        reg = term if reg is None else reg + term

    return reg
