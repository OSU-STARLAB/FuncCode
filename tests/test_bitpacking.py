import torch
from funcodekan.hwq.bitpacking import pack_unsigned, unpack_unsigned

def test_pack_roundtrip():
    for bits in range(1, 9):
        vals = torch.arange(0, min(2 ** bits, 17), dtype=torch.long)
        packed = pack_unsigned(vals, bits)
        recovered = unpack_unsigned(packed, bits, vals.numel())
        assert torch.equal(vals, recovered)
