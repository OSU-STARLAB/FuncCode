from dataclasses import dataclass
from typing import Literal, Optional

TargetHardware = Literal["cpu", "gpu", "fpga_bram", "asic_sram"]
Granularity = Literal["per_tensor", "per_vector", "per_channel"]
QuantMode = Literal["symmetric", "asymmetric"]

@dataclass
class HWQConfig:
    """
    Hardware-aware quantization configuration for compressed KAN codebooks.
    """
    codebook_bits: int = 4
    index_bits: Optional[int] = None
    quant_granularity: Granularity = "per_vector"
    quant_mode: QuantMode = "symmetric"
    target: TargetHardware = "fpga_bram"
    align_bytes: int = 1
    keep_fp32_scales: bool = True
    huffman_indices: bool = False
