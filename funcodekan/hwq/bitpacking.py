import math
import torch

def required_index_bits(num_clusters: int) -> int:
    if num_clusters <= 1:
        return 1
    return int(math.ceil(math.log2(num_clusters)))

def pack_unsigned(values: torch.Tensor, bits: int) -> torch.ByteTensor:
    """
    Pack a 1D unsigned tensor into bytes.
    Supports 1 to 8 bits.
    """
    assert values.dim() == 1, "values must be 1D."
    assert 1 <= bits <= 8
    values = values.detach().cpu().to(torch.long)
    max_val = 2 ** bits
    assert torch.all((values >= 0) & (values < max_val)), f"values must be in [0, {max_val})."

    total_bits = values.numel() * bits
    out_bytes = (total_bits + 7) // 8
    out = torch.zeros(out_bytes, dtype=torch.uint8)

    bit_pos = 0
    for v in values.tolist():
        for b in range(bits - 1, -1, -1):
            if (v >> b) & 1:
                byte_idx = bit_pos // 8
                offset = 7 - (bit_pos % 8)
                out[byte_idx] |= (1 << offset)
            bit_pos += 1
    return out

def unpack_unsigned(packed: torch.ByteTensor, bits: int, length: int) -> torch.Tensor:
    assert 1 <= bits <= 8
    packed = packed.detach().cpu().to(torch.uint8)
    values = torch.zeros(length, dtype=torch.long)

    bit_pos = 0
    for i in range(length):
        v = 0
        for _ in range(bits):
            byte_idx = bit_pos // 8
            offset = 7 - (bit_pos % 8)
            bit = (int(packed[byte_idx]) >> offset) & 1
            v = (v << 1) | bit
            bit_pos += 1
        values[i] = v
    return values

def signed_to_unsigned(q: torch.Tensor, bits: int) -> torch.Tensor:
    offset = 2 ** (bits - 1)
    return (q.to(torch.long) + offset).clamp(0, 2 ** bits - 1)

def unsigned_to_signed(q: torch.Tensor, bits: int) -> torch.Tensor:
    offset = 2 ** (bits - 1)
    return (q.to(torch.long) - offset).to(torch.int8)
