"""Golden export for the GRAM designs (G1/G2/G3) -> hw/golden/gram_*.

Reuses the artifact machinery of export.py (c_array, manifests, golden
emission, stratified selection) and adds the GRAM-specific parameter
blocks: LN constants (gamma/beta/eps in the frozen fixed-point formats),
the 257-entry interpolated SiLU table, pre-LN requant constants and the
post-SiLU activation requant.

CLI:  python -m funcodekan.hw.gram_export --design g1|g2|g3
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from funcodekan.data.mnist import get_mnist_loaders
from funcodekan.models.variants import DirectKANVariant
from funcodekan.utils.training import set_seed

from . import gram_fixed_point as gfp
from .export import (_finish_export, _fmt_f32, _fmt_int, _guard, c_array,
                     stratified_indices)
from .gram_qat import QuantGramBranchKAN, QuantGramKAN, rebuild_gram_branch
from .gram_spec import (BETA_FRAC, GAMMA_FRAC, LOGIT_SHIFT, NORM_FRAC,
                        PRE_LN_FRAC, SILU_TAB_FRAC, SILU_TABLE, ln_eps_int)
from .verify import _collect_test_set, run_l1_fp32, run_l1_quant, run_l4
from .export import pack_base_words, _fmt_hex32


def pack_basis_words16(basis_q: np.ndarray) -> np.ndarray:
    """[out,in,4] INT4 codes -> uint16 word per edge (nibble k at bits
    4k+3:4k). Exactly 4 bits/coefficient."""
    q = basis_q.reshape(-1, 4).astype(np.int64) & 0xF
    words = np.zeros(len(q), dtype=np.uint64)
    for k in range(4):
        words |= q[:, k].astype(np.uint64) << (4 * k)
    return words.astype(np.uint16)


def _fmt_hex16(v) -> str:
    return f"0x{int(v) & 0xFFFF:04x}u"


def _dims_block(layers) -> str:
    lines = [f"#define N_LAYERS {len(layers)}"]
    for l, layer in enumerate(layers):
        lines.append(f"#define L{l}_IN {layer.in_features}")
        lines.append(f"#define L{l}_OUT {layer.out_features}")
    lines.append("#define GRAM_BASIS_DIM 4")
    return "\n".join(lines) + "\n\n"


def _silu_table_block() -> str:
    return c_array("silu_tab", "lut_t", SILU_TABLE)


def _gram_layer_block(l: int, ql: gfp.GramQuantLayer) -> str:
    assert np.abs(ql.gamma_q).max() < 2 ** 15, "gamma exceeds int16 Q4.12"
    assert np.abs(ql.beta_q).max() < 2 ** 31, "beta exceeds int32 Q.24"
    parts = [
        c_array(f"l{l}_lut_b", "lut_t", ql.lut_b, dims=(16, 4)),
        c_array(f"l{l}_lut_s", "lut_t", ql.lut_s),
        f"static const mult_t l{l}_mult_basis = {ql.mult_basis};\n"
        f"static const mult_t l{l}_mult_base = {ql.mult_base};\n"
        f"static const int l{l}_shift = {ql.shift};\n"
        f"static const mult_t l{l}_act_mult = {ql.act_mult};\n"
        f"static const int l{l}_act_shift = {ql.act_shift};\n"
        f"static const long long l{l}_eps_int = "
        f"{ln_eps_int(len(ql.gamma_q))}LL;\n",
        c_array(f"l{l}_gamma_q", "lut_t", ql.gamma_q),
        c_array(f"l{l}_beta_q", "int", ql.beta_q),
    ]
    return "".join(parts)


def emit_params_g2(model: gfp.GramFixedPointModel, layers) -> str:
    body = _dims_block(layers) + _silu_table_block()
    for l, ql in enumerate(model.layers):
        out_f, in_f = ql.base_q.shape
        groups = (out_f + 7) // 8
        body += f"#define L{l}_BASE_GROUPS {groups}\n"
        body += _gram_layer_block(l, ql)
        body += c_array(f"l{l}_basis_pk", "unsigned short",
                        pack_basis_words16(ql.basis_q), fmt=_fmt_hex16,
                        per_line=8)
        body += c_array(f"l{l}_base_pk", "unsigned int",
                        pack_base_words(ql.base_q), fmt=_fmt_hex32,
                        per_line=8)
        body += (f"HW_STATIC_ASSERT(sizeof(l{l}_basis_pk)/"
                 f"sizeof(l{l}_basis_pk[0]) == L{l}_OUT*L{l}_IN, "
                 f"\"l{l}_basis_pk size\");\n")
    return _guard("params_g2", body)


def emit_params_g3(model: gfp.GramFixedPointModel, layers) -> str:
    body = _dims_block(layers) + _silu_table_block()
    for l, ql in enumerate(model.layers):
        ks = ql.basis_codebook_q.shape[0]
        kb = ql.base_codebook_q.shape[0]
        body += f"#define L{l}_KS {ks}\n#define L{l}_KB {kb}\n"
        body += _gram_layer_block(l, ql)
        body += c_array(f"l{l}_basis_codebook_q", "w4_t",
                        ql.basis_codebook_q, dims=(ks, 4))
        body += c_array(f"l{l}_base_codebook_q", "w4_t", ql.base_codebook_q)
        body += c_array(f"l{l}_basis_ids", "sidxrom_t", ql.basis_ids)
        body += c_array(f"l{l}_base_ids", "bidxrom_t", ql.base_ids)
        body += (f"HW_STATIC_ASSERT(sizeof(l{l}_basis_ids)/"
                 f"sizeof(l{l}_basis_ids[0]) == L{l}_OUT*L{l}_IN, "
                 f"\"l{l}_basis_ids size\");\n")
    return _guard("params_g3", body)


def emit_params_g1(model: gfp.GramFp32Model, layers) -> str:
    body = _dims_block(layers)
    body += "#define COEFF_DIM 5\n"
    for l, fl in enumerate(model.layers):
        out_f, in_f, cd = fl.weight.shape
        body += c_array(f"l{l}_weight", "float", fl.weight, fmt=_fmt_f32,
                        per_line=8, dims=(out_f * in_f, cd))
        body += c_array(f"l{l}_gamma", "float", fl.gamma, fmt=_fmt_f32)
        body += c_array(f"l{l}_beta", "float", fl.beta, fmt=_fmt_f32)
        body += (f"HW_STATIC_ASSERT(sizeof(l{l}_weight)/sizeof(float) == "
                 f"L{l}_OUT*L{l}_IN*COEFF_DIM, \"l{l}_weight size\");\n")
    return _guard("params_g1", body)


def _export_gram_quant(design, wrapper, fp_model, test_loader, out_dir,
                       dense_acc, params_h,
                       log_path="runs_hw/verification_log.json"):
    out_dir = Path(out_dir)
    l1 = run_l1_quant(design, wrapper, fp_model, test_loader, log_path)
    l4 = (run_l4(design, l1["emulator_acc"], dense_acc, log_path)
          if dense_acc is not None else None)
    x, y = _collect_test_set(test_loader)
    idx = stratified_indices(y)
    gin = fp_model.quantize_input(x[idx]).astype(np.int8)
    gout = fp_model.forward_codes(gin.astype(np.int64)).astype(np.int32)
    return _finish_export(
        design, out_dir, params_h, gin, gout, y[idx].astype(np.uint8), idx,
        {"l1": l1, "l4": l4, "family": "gram",
         "input_step": float(fp_model.input_step),
         "act_steps": wrapper.act_steps(),
         "pre_ln_frac": PRE_LN_FRAC, "norm_frac": NORM_FRAC,
         "gamma_frac": GAMMA_FRAC, "beta_frac": BETA_FRAC,
         "silu_tab_frac": SILU_TAB_FRAC, "logit_shift": LOGIT_SHIFT,
         "requant": [{"mult_basis": ql.mult_basis,
                      "mult_base": ql.mult_base, "shift": ql.shift,
                      "act_mult": ql.act_mult, "act_shift": ql.act_shift}
                     for ql in fp_model.layers]})


def export_g1(dense, test_loader, out_dir=Path("hw/golden/gram_fp32"),
              log_path="runs_hw/verification_log.json"):
    np_model = gfp.build_from_gram_dense(dense)
    l1 = run_l1_fp32(dense, np_model, test_loader, log_path)
    x, y = _collect_test_set(test_loader)
    idx = stratified_indices(y)
    gin = x[idx].astype(np.float32)
    gout = np_model.forward(gin).astype(np.float32)
    return _finish_export(
        "gram_fp32", Path(out_dir), emit_params_g1(np_model, dense.layers),
        gin, gout, y[idx].astype(np.uint8), idx,
        {"l1": l1, "family": "gram", "logit_tolerance_tb": 1e-3})


def export_g2(wrapper, test_loader, out_dir=Path("hw/golden/gram_lsq_w4a4"),
              dense_acc=None, log_path="runs_hw/verification_log.json"):
    model = gfp.build_from_g2(wrapper)
    return _export_gram_quant("gram_lsq_w4a4", wrapper, model, test_loader,
                              out_dir, dense_acc,
                              emit_params_g2(model, wrapper.gram_layers()),
                              log_path)


def export_g3(wrapper, test_loader,
              out_dir=Path("hw/golden/gram_funccode_w4a4"),
              dense_acc=None, log_path="runs_hw/verification_log.json"):
    model = gfp.build_from_g3(wrapper)
    return _export_gram_quant("gram_funccode_w4a4", wrapper, model,
                              test_loader, out_dir, dense_acc,
                              emit_params_g3(model, wrapper.gram_layers()),
                              log_path)


def _dense_acc(runs: Path):
    import pandas as pd
    csv = runs / "gram_hw_source" / "gram" / "summary.csv"
    if not csv.exists():
        return None
    df = pd.read_csv(csv)
    row = df[df["stage"] == "dense_fp32"]
    return float(row["test_acc"].iloc[0]) if len(row) else None


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--design", required=True, choices=["g1", "g2", "g3"])
    p.add_argument("--runs-dir", type=str, default="runs_hw")
    p.add_argument("--out-root", type=str, default="hw/golden")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args(argv)
    set_seed(args.seed)
    runs = Path(args.runs_dir)
    data = get_mnist_loaders(seed=args.seed, num_workers=0)
    dense_acc = _dense_acc(runs)
    if args.design == "g1":
        ckpt = torch.load(runs / "gram_hw_source" / "gram" / "dense.pt",
                          map_location="cpu", weights_only=True)
        dense = DirectKANVariant("gram", 784, 64, 10, degree=3)
        dense.load_state_dict(ckpt["model"])
        m = export_g1(dense, data.test_loader,
                      Path(args.out_root) / "gram_fp32")
    elif args.design == "g2":
        ckpt = torch.load(runs / "g2_gram_lsq_w4a4" / "model.pt",
                          map_location="cpu", weights_only=True)
        dense = DirectKANVariant("gram", 784, 64, 10, degree=3)
        wrapper = QuantGramKAN(dense)
        wrapper.load_state_dict(ckpt["state_dict"])
        m = export_g2(wrapper.eval(), data.test_loader,
                      Path(args.out_root) / "gram_lsq_w4a4", dense_acc)
    else:
        ckpt = torch.load(runs / "g3_gram_funccode_w4a4" / "model.pt",
                          map_location="cpu", weights_only=True)
        sd = ckpt["state_dict"]
        branch_sd = {k[len("branch."):]: v for k, v in sd.items()
                     if k.startswith("branch.")}
        branch = rebuild_gram_branch(branch_sd)
        wrapper = QuantGramBranchKAN(branch)
        wrapper.load_state_dict(sd)
        m = export_g3(wrapper.eval(), data.test_loader,
                      Path(args.out_root) / "gram_funccode_w4a4", dense_acc)
    l1 = m.get("l1", {})
    print(f"[gram_export] {args.design}: L1 passed={l1.get('passed')} "
          f"emulator_acc={l1.get('emulator_acc', l1.get('numpy_acc'))}")
    if not l1.get("passed"):
        raise SystemExit(f"L1 FAILED for {args.design}")


if __name__ == "__main__":
    main()
