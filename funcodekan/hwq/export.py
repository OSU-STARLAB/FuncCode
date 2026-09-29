import json
import torch
from typing import Dict, Any
from .configs import HWQConfig
from .compressed_state import CompressedKANState
from .quantizers import quantize_codebook
from .bitpacking import pack_unsigned, signed_to_unsigned, required_index_bits
from .storage import estimate_layer_storage, summarize_reports

def export_hardware_quantized_state(
    compressed: CompressedKANState,
    cfg: HWQConfig,
) -> Dict[str, Any]:
    """
    Convert a clustered KAN state into a hardware-aware quantized export package.
    """
    export = {}
    reports = []
    metadata = {
        "format": "FuncCodeKAN-HWQ-v1",
        "codebook_bits": cfg.codebook_bits,
        "quant_granularity": cfg.quant_granularity,
        "quant_mode": cfg.quant_mode,
        "target": cfg.target,
        "layers": {},
    }

    for layer_idx in compressed.layer_indices():
        codebook = compressed.codebooks[layer_idx].detach().cpu()
        ids = compressed.cluster_ids[layer_idx].detach().cpu().to(torch.long).reshape(-1)

        qt = quantize_codebook(codebook, cfg)
        qflat = qt.qweight.reshape(-1)

        if qt.mode == "symmetric":
            qflat_unsigned = signed_to_unsigned(qflat, cfg.codebook_bits)
        else:
            qflat_unsigned = qflat.to(torch.long)

        packed_q = pack_unsigned(qflat_unsigned, cfg.codebook_bits)

        num_clusters = codebook.reshape(codebook.shape[0], -1).shape[0]
        idx_bits = cfg.index_bits if cfg.index_bits is not None else required_index_bits(num_clusters)
        packed_ids = pack_unsigned(ids, idx_bits)

        export[f"q_codebooks.{layer_idx}"] = packed_q
        export[f"cluster_ids_packed.{layer_idx}"] = packed_ids
        export[f"codebook_scales.{layer_idx}"] = qt.scale.detach().cpu()

        if qt.zero_point is not None:
            export[f"codebook_zero_points.{layer_idx}"] = qt.zero_point.detach().cpu()

        report = estimate_layer_storage(
            layer_idx=layer_idx,
            codebook=codebook,
            cluster_ids=ids,
            codebook_bits=cfg.codebook_bits,
            index_bits=idx_bits,
            granularity=cfg.quant_granularity,
            mode=cfg.quant_mode,
            scale_bits=32 if cfg.keep_fp32_scales else 16,
        )
        reports.append(report)

        metadata["layers"][str(layer_idx)] = {
            "original_codebook_shape": list(codebook.shape),
            "q_codebook_num_values": int(qflat.numel()),
            "q_codebook_packed_num_bytes": int(packed_q.numel()),
            "cluster_ids_num_values": int(ids.numel()),
            "cluster_ids_packed_num_bytes": int(packed_ids.numel()),
            "index_bits": idx_bits,
            "scale_shape": list(qt.scale.shape),
            "zero_point_shape": None if qt.zero_point is None else list(qt.zero_point.shape),
        }

    metadata["storage_summary"] = summarize_reports(reports)

    metadata_bytes = json.dumps(metadata, indent=2).encode("utf-8")
    export["metadata_json"] = torch.tensor(list(metadata_bytes), dtype=torch.uint8)

    for k, v in compressed.other_state.items():
        export[f"other_state.{k}"] = v.detach().cpu()

    return export

def read_metadata_from_export(export_state: Dict[str, torch.Tensor]) -> Dict[str, Any]:
    data = bytes(export_state["metadata_json"].cpu().tolist())
    return json.loads(data.decode("utf-8"))
