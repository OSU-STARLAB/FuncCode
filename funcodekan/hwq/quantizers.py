import torch
from dataclasses import dataclass
from typing import Optional, Tuple
from .configs import HWQConfig

@dataclass
class QuantizedTensor:
    qweight: torch.Tensor
    scale: torch.Tensor
    zero_point: Optional[torch.Tensor]
    bits: int
    granularity: str
    mode: str
    original_shape: Tuple[int, ...]

def _safe_scale(x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    return torch.clamp(x, min=eps)

def symmetric_quantize(x: torch.Tensor, bits: int, granularity: str) -> QuantizedTensor:
    """
    Symmetric signed quantization.

    For KAN codebooks shaped [K, D]:
    - per_tensor: scale shape [1]
    - per_vector: scale shape [K, 1]
    - per_channel: scale shape [1, D]
    """
    assert 2 <= bits <= 8, "This implementation supports 2-8 bit signed quantization."
    qmax = 2 ** (bits - 1) - 1
    qmin = -2 ** (bits - 1)

    if granularity == "per_tensor":
        max_abs = x.abs().amax().view(1)
    elif granularity == "per_vector":
        max_abs = x.abs().amax(dim=1, keepdim=True)
    elif granularity == "per_channel":
        max_abs = x.abs().amax(dim=0, keepdim=True)
    else:
        raise ValueError(f"Unknown granularity: {granularity}")

    scale = _safe_scale(max_abs / qmax)
    q = torch.round(x / scale).clamp(qmin, qmax).to(torch.int8)

    return QuantizedTensor(q, scale, None, bits, granularity, "symmetric", tuple(x.shape))

def asymmetric_quantize(x: torch.Tensor, bits: int, granularity: str) -> QuantizedTensor:
    """
    Unsigned asymmetric affine quantization.
    """
    assert 2 <= bits <= 8, "This implementation supports 2-8 bit quantization."
    qmin, qmax = 0, 2 ** bits - 1

    if granularity == "per_tensor":
        xmin = x.amin().view(1)
        xmax = x.amax().view(1)
    elif granularity == "per_vector":
        xmin = x.amin(dim=1, keepdim=True)
        xmax = x.amax(dim=1, keepdim=True)
    elif granularity == "per_channel":
        xmin = x.amin(dim=0, keepdim=True)
        xmax = x.amax(dim=0, keepdim=True)
    else:
        raise ValueError(f"Unknown granularity: {granularity}")

    scale = _safe_scale((xmax - xmin) / float(qmax - qmin))
    zero_point = torch.round(qmin - xmin / scale).clamp(qmin, qmax).to(torch.uint8)
    q = torch.round(x / scale + zero_point).clamp(qmin, qmax).to(torch.uint8)

    return QuantizedTensor(q, scale, zero_point, bits, granularity, "asymmetric", tuple(x.shape))

def dequantize(qt: QuantizedTensor) -> torch.Tensor:
    if qt.mode == "symmetric":
        return qt.qweight.float() * qt.scale
    if qt.mode == "asymmetric":
        return (qt.qweight.float() - qt.zero_point.float()) * qt.scale
    raise ValueError(f"Unknown quantization mode: {qt.mode}")

def quantize_codebook(codebook: torch.Tensor, cfg: HWQConfig) -> QuantizedTensor:
    """
    Quantize a KAN compressed codebook/prototype tensor.

    Expected input:
        codebook: [num_clusters, coeffs_per_edge]
    """
    if codebook.dim() != 2:
        codebook = codebook.reshape(codebook.shape[0], -1)

    if cfg.quant_mode == "symmetric":
        return symmetric_quantize(codebook, cfg.codebook_bits, cfg.quant_granularity)
    if cfg.quant_mode == "asymmetric":
        return asymmetric_quantize(codebook, cfg.codebook_bits, cfg.quant_granularity)
    raise ValueError(f"Unknown quantization mode: {cfg.quant_mode}")

class FakeQuantCodebookSTE(torch.autograd.Function):
    """
    Straight-through estimator for codebook quantization-aware fine-tuning.
    """
    @staticmethod
    def forward(ctx, x: torch.Tensor, bits: int, granularity: str):
        qt = symmetric_quantize(x, bits, granularity)
        return dequantize(qt)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None, None

def fake_quant_codebook(x: torch.Tensor, bits: int = 4, granularity: str = "per_vector") -> torch.Tensor:
    return FakeQuantCodebookSTE.apply(x, bits, granularity)
