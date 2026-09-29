import argparse
import torch
from .configs import HWQConfig
from .compressed_state import CompressedKANState
from .export import export_hardware_quantized_state, read_metadata_from_export

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--codebook-bits", type=int, default=4)
    parser.add_argument("--index-bits", type=str, default="auto")
    parser.add_argument("--quant-granularity", type=str, default="per_vector",
                        choices=["per_tensor", "per_vector", "per_channel"])
    parser.add_argument("--quant-mode", type=str, default="symmetric",
                        choices=["symmetric", "asymmetric"])
    parser.add_argument("--target", type=str, default="fpga_bram",
                        choices=["cpu", "gpu", "fpga_bram", "asic_sram"])
    args = parser.parse_args()

    state = torch.load(args.checkpoint, map_location="cpu")
    if "state_dict" in state:
        state = state["state_dict"]

    compressed = CompressedKANState.from_state_dict(state)

    cfg = HWQConfig(
        codebook_bits=args.codebook_bits,
        index_bits=None if args.index_bits == "auto" else int(args.index_bits),
        quant_granularity=args.quant_granularity,
        quant_mode=args.quant_mode,
        target=args.target,
    )

    export = export_hardware_quantized_state(compressed, cfg)
    torch.save(export, args.output)

    meta = read_metadata_from_export(export)
    print("Saved:", args.output)
    print("Compression ratio vs FP32 edge coefficients:",
          meta["storage_summary"]["compression_ratio_vs_fp32_edges"])
    print("Total storage KiB:", meta["storage_summary"]["total_storage_kib"])

if __name__ == "__main__":
    main()
