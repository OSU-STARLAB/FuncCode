import torch
from funcodekan.hwq.configs import HWQConfig
from funcodekan.hwq.compressed_state import CompressedKANState
from funcodekan.hwq.export import export_hardware_quantized_state, read_metadata_from_export

def test_export_toy():
    state = {
        "clustered_weights.0": torch.randn(16, 9),
        "cluster_ids.0": torch.randint(0, 16, (128,)),
    }
    compressed = CompressedKANState.from_state_dict(state)
    cfg = HWQConfig(codebook_bits=4)
    out = export_hardware_quantized_state(compressed, cfg)
    meta = read_metadata_from_export(out)
    assert "storage_summary" in meta
    assert meta["layers"]["0"]["index_bits"] == 4
