"""Golden export for the KAGN-Conv designs (K1/K2/K3) -> hw/golden/kagnconv_*.

Same artifact contract as the other families: params.h + golden.h + .npy +
SHA256 manifest, with the export CLI running the L1 (and L4) gates.
Golden inputs are the [1,28,28] images flattened row-major to 784 values
(INT4 codes for K2/K3, float32 for K1); golden outputs are INT32 logits at
2^-16 (float32 for K1).

CLI:  python -m funcodekan.hw.kagn_export --design k1|k2|k3
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

import funcodekan.models  # noqa: F401  (import-order guard)
from funcodekan.data.bundles import get_dataset_bundle
from funcodekan.utils.training import set_seed

from . import kagn_fixed_point as kfp
from .export import (_finish_export, _fmt_f32, _fmt_hex32, _fmt_int, _guard,
                     c_array, pack_base_words, stratified_indices)
from .gram_export import _fmt_hex16, _silu_table_block, pack_basis_words16
from .gram_spec import ln_eps_int
from .kagn_conv import CompressedKagnConvNet, KagnConvNet
from .kagn_qat import QuantKagnBranchNet, QuantKagnConvNet
from .verify import _collect_test_set, run_l1_quant, run_l4
from .veriflog import append_verification_entry

CHANNELS = (16, 32)


def _dims_block(channels) -> str:
    c1, c2 = channels
    return (f"#define CONV1_IN_CH 1\n#define CONV1_OUT_CH {c1}\n"
            f"#define CONV2_IN_CH {c1}\n#define CONV2_OUT_CH {c2}\n"
            "#define CONV_K 3\n#define CONV_STRIDE 2\n#define CONV_PAD 1\n"
            "#define IMG_H 28\n#define IMG_W 28\n"
            "#define F1_H 14\n#define F1_W 14\n"
            "#define F2_H 7\n#define F2_W 7\n"
            f"#define HEAD_IN {c2}\n#define HEAD_OUT 10\n"
            "#define GRAM_BASIS_DIM 4\n\n")


CONV_POSITIONS = (196, 49)   # 14x14, 7x7 output maps


def _conv_block(i: int, cl: kfp.KagnConvQuantLayer) -> str:
    n = f"c{i + 1}"
    assert np.abs(cl.gamma_q).max() < 2 ** 15, "conv gamma exceeds int16"
    assert np.abs(cl.beta_q).max() < 2 ** 31, "conv beta exceeds int32"
    parts = [
        c_array(f"{n}_lut_b", "lut_t", cl.lut_b, dims=(16, 4)),
        c_array(f"{n}_lut_s", "lut_t", cl.lut_s),
        f"static const mult_t {n}_mult_poly = {cl.m_s};\n"
        f"static const mult_t {n}_mult_base = {cl.m_b};\n"
        f"static const int {n}_shift = {cl.shift};\n"
        f"static const long long {n}_eps_int = "
        f"{ln_eps_int(CONV_POSITIONS[i])}LL;\n"
        f"static const mult_t {n}_act_mult = {cl.act_mult};\n"
        f"static const int {n}_act_shift = {cl.act_shift};\n",
        c_array(f"{n}_gamma_q", "lut_t", cl.gamma_q),
        c_array(f"{n}_beta_q", "int", cl.beta_q),
    ]
    return "".join(parts)


def _head_block(hl) -> str:
    parts = [
        c_array("hd_lut_b", "lut_t", hl.lut_b, dims=(16, 4)),
        c_array("hd_lut_s", "lut_t", hl.lut_s),
        f"static const mult_t hd_mult_basis = {hl.mult_basis};\n"
        f"static const mult_t hd_mult_base = {hl.mult_base};\n"
        f"static const int hd_shift = {hl.shift};\n"
        f"static const long long hd_eps_int = "
        f"{ln_eps_int(len(hl.gamma_q))}LL;\n",
        c_array("hd_gamma_q", "lut_t", hl.gamma_q),
        c_array("hd_beta_q", "int", hl.beta_q),
    ]
    return "".join(parts)


def emit_params_quant(model: kfp.KagnFixedPointModel, design: str) -> str:
    body = _dims_block(CHANNELS) + _silu_table_block()
    body += (f"static const mult_t pool_mult = {model.pool_mult};\n"
             f"static const int pool_shift = {model.pool_shift};\n")
    for i, cl in enumerate(model.conv_layers):
        n = f"c{i + 1}"
        body += _conv_block(i, cl)
        if design == "k2":
            body += c_array(f"{n}_poly_pk", "unsigned short",
                            pack_basis_words16(cl.poly_q), fmt=_fmt_hex16,
                            per_line=8)
            body += c_array(f"{n}_base_pk", "unsigned int",
                            pack_base_words(
                                cl.base_q.reshape(cl.base_q.shape[0], -1)),
                            fmt=_fmt_hex32, per_line=8)
        else:
            ks = cl.poly_codebook_q.shape[0]
            kb = cl.base_codebook_q.shape[0]
            body += f"#define {n.upper()}_KS {ks}\n#define {n.upper()}_KB {kb}\n"
            body += c_array(f"{n}_poly_codebook_q", "w4_t",
                            cl.poly_codebook_q, dims=(ks, 4))
            body += c_array(f"{n}_base_codebook_q", "w4_t",
                            cl.base_codebook_q)
            body += c_array(f"{n}_poly_ids", "kidxrom_t",
                            cl.poly_ids)
            body += c_array(f"{n}_base_ids", "bidxrom_t", cl.base_ids)
    hl = model.head
    body += _head_block(hl)
    if design == "k2":
        out_f, in_f = hl.base_q.shape
        body += f"#define HD_BASE_GROUPS {(out_f + 7) // 8}\n"
        body += c_array("hd_basis_pk", "unsigned short",
                        pack_basis_words16(hl.basis_q), fmt=_fmt_hex16,
                        per_line=8)
        body += c_array("hd_base_pk", "unsigned int",
                        pack_base_words(hl.base_q), fmt=_fmt_hex32,
                        per_line=8)
    else:
        ks = hl.basis_codebook_q.shape[0]
        kb = hl.base_codebook_q.shape[0]
        body += f"#define HD_KS {ks}\n#define HD_KB {kb}\n"
        body += c_array("hd_basis_codebook_q", "w4_t", hl.basis_codebook_q,
                        dims=(ks, 4))
        body += c_array("hd_base_codebook_q", "w4_t", hl.base_codebook_q)
        body += c_array("hd_basis_ids", "kidxrom_t", hl.basis_ids)
        body += c_array("hd_base_ids", "bidxrom_t", hl.base_ids)
    return _guard(f"params_{design}", body)


def emit_params_k1(model: kfp.KagnFp32Model) -> str:
    body = _dims_block(CHANNELS)
    body += "#define COEFF_DIM 5\n"
    for i, (weight, gamma, beta, _, _) in enumerate(model.convs):
        n = f"c{i + 1}"
        o, cin = weight.shape[0], weight.shape[1]
        body += c_array(f"{n}_weight", "float", weight, fmt=_fmt_f32,
                        per_line=8, dims=(o * cin * 9, 5))
        body += c_array(f"{n}_gamma", "float", gamma, fmt=_fmt_f32)
        body += c_array(f"{n}_beta", "float", beta, fmt=_fmt_f32)
    hl = model.head
    o, cin, cd = hl.weight.shape
    body += c_array("hd_weight", "float", hl.weight, fmt=_fmt_f32,
                    per_line=8, dims=(o * cin, cd))
    body += c_array("hd_gamma", "float", hl.gamma, fmt=_fmt_f32)
    body += c_array("hd_beta", "float", hl.beta, fmt=_fmt_f32)
    return _guard("params_k1", body)


def _golden_from(fp_model, x, idx, quant: bool):
    if quant:
        gin4 = fp_model.quantize_input(x[idx])                # [N,1,28,28]
        gout = fp_model.forward_codes(gin4).astype(np.int32)
        gin = gin4.astype(np.int8).reshape(len(idx), -1)
    else:
        gin4 = x[idx].astype(np.float32)
        gout = fp_model.forward(gin4).astype(np.float32)
        gin = gin4.reshape(len(idx), -1)
    return gin, gout


def _kagn_l1_fp32(net, np_model, test_loader, log_path):
    """K1 L1 with a 1e-3 tolerance (documented: the numpy conv tap order
    differs from torch's conv kernels; TB tolerance is 1e-3 as well)."""
    x, y = _collect_test_set(test_loader)
    with torch.no_grad():
        outs = []
        for i in range(0, len(x), 512):
            outs.append(net(torch.from_numpy(x[i:i + 512]).float()).numpy())
    t_logits = np.concatenate(outs)
    n_logits = np_model.forward(x)
    max_abs = float(np.max(np.abs(t_logits - n_logits)))
    agree = int((t_logits.argmax(1) == n_logits.argmax(1)).sum())
    acc = 100.0 * float((n_logits.argmax(1) == y).mean())
    result = {"milestone": "K2", "rung": "L1", "design": "kagnconv_fp32",
              "n_images": len(y), "max_abs_logit_diff": max_abs,
              "argmax_agree": agree, "numpy_acc": acc,
              "tolerance": 1e-3,
              "passed": max_abs <= 1e-3 and agree == len(y)}
    append_verification_entry(result, log_path)
    return result


def export_k1(net, test_loader, out_dir=Path("hw/golden/kagnconv_fp32"),
              log_path="runs_hw/verification_log.json"):
    np_model = kfp.build_from_kagn_dense(net)
    l1 = _kagn_l1_fp32(net, np_model, test_loader, log_path)
    x, y = _collect_test_set(test_loader)
    idx = stratified_indices(y)
    gin, gout = _golden_from(np_model, x, idx, quant=False)
    return _finish_export("kagnconv_fp32", Path(out_dir),
                          emit_params_k1(np_model), gin, gout,
                          y[idx].astype(np.uint8), idx,
                          {"l1": l1, "family": "kagnconv",
                           "logit_tolerance_tb": 1e-3})


def _export_quant(design_name, design_key, wrapper, is_k3, test_loader,
                  out_dir, dense_acc,
                  log_path="runs_hw/verification_log.json"):
    fp_model = kfp.build_from_kagn(wrapper, is_k3=is_k3)
    l1 = run_l1_quant(design_name, wrapper, fp_model, test_loader, log_path)
    l4 = (run_l4(design_name, l1["emulator_acc"], dense_acc, log_path)
          if dense_acc is not None else None)
    x, y = _collect_test_set(test_loader)
    idx = stratified_indices(y)
    gin, gout = _golden_from(fp_model, x, idx, quant=True)
    return _finish_export(
        design_name, Path(out_dir), emit_params_quant(fp_model, design_key),
        gin, gout, y[idx].astype(np.uint8), idx,
        {"l1": l1, "l4": l4, "family": "kagnconv",
         "input_step": float(fp_model.input_step),
         "act_steps": wrapper.act_steps(),
         "pool": {"mult": fp_model.pool_mult,
                  "shift": fp_model.pool_shift}})


def _dense_acc(runs: Path):
    import pandas as pd
    csv = runs / "kagnconv_hw_source" / "summary.csv"
    if not csv.exists():
        return None
    df = pd.read_csv(csv)
    row = df[df["stage"] == "dense_fp32"]
    return float(row["test_acc"].iloc[0]) if len(row) else None


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--design", required=True, choices=["k1", "k2", "k3"])
    p.add_argument("--runs-dir", type=str, default="runs_hw")
    p.add_argument("--out-root", type=str, default="hw/golden")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args(argv)
    set_seed(args.seed)
    runs = Path(args.runs_dir)
    data = get_dataset_bundle("mnist", seed=args.seed, num_workers=0,
                              flatten=False)
    dense_acc = _dense_acc(runs)
    if args.design == "k1":
        ckpt = torch.load(runs / "kagnconv_hw_source" / "dense.pt",
                          map_location="cpu", weights_only=True)
        net = KagnConvNet(CHANNELS)
        net.load_state_dict(ckpt["model"])
        m = export_k1(net.eval(), data.test_loader,
                      Path(args.out_root) / "kagnconv_fp32")
    elif args.design == "k2":
        ckpt = torch.load(runs / "k2_kagnconv_lsq_w4a4" / "model.pt",
                          map_location="cpu", weights_only=True)
        wrapper = QuantKagnConvNet(KagnConvNet(CHANNELS))
        wrapper.load_state_dict(ckpt["state_dict"])
        m = _export_quant("kagnconv_lsq_w4a4", "k2", wrapper.eval(), False,
                          data.test_loader,
                          Path(args.out_root) / "kagnconv_lsq_w4a4",
                          dense_acc)
    else:
        ckpt = torch.load(runs / "k3_kagnconv_funccode_w4a4" / "model.pt",
                          map_location="cpu", weights_only=True)
        sd = ckpt["state_dict"]
        ks = sd["net.spline_codebooks.0"].shape[0]
        kb = sd["net.base_codebooks.0"].shape[0]
        comp = CompressedKagnConvNet(KagnConvNet(CHANNELS), ks=ks, kb=kb,
                                     seed=args.seed)
        wrapper = QuantKagnBranchNet(comp)
        wrapper.load_state_dict(sd)
        m = _export_quant("kagnconv_funccode_w4a4", "k3", wrapper.eval(),
                          True, data.test_loader,
                          Path(args.out_root) / "kagnconv_funccode_w4a4",
                          dense_acc)
    l1 = m.get("l1", {})
    print(f"[kagn_export] {args.design}: L1 passed={l1.get('passed')} "
          f"emulator_acc={l1.get('emulator_acc', l1.get('numpy_acc'))}")
    if not l1.get("passed"):
        raise SystemExit(f"L1 FAILED for {args.design}")


if __name__ == "__main__":
    main()
