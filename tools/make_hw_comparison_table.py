"""Build the D1/D2/D3 hardware comparison table (T-HW) from parsed reports.

Inputs (all traceable to files on disk):
  --csv        tidy report CSV from tools/parse_hls_reports.py
  --runs-dir   runs_hw/ (accuracy + analytical parameter storage):
     branch_k32_s32_b16/summary.csv   dense FP32 accuracy (D1)
     d2_lsq_w4a4/summary.csv          D2 QAT accuracy + W4 storage
     d3_funccode_w4a4/summary.csv     D3 accuracy + analytical storage
     ../hw/golden/*/manifest.json     emulator accuracies (L1)

Emits LaTeX (three rows, one stage) + a markdown twin. Post-synthesis and
post-implementation stages are separate tables; never mixed.

Usage: see scripts/hw/13_reports.sh
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

DESIGN_LABELS = {
    "fp32": ("D1", "Spline FP32 dense"),
    "lsq_w4a4": ("D2", "Spline LSQ W4A4"),
    "funccode_w4a4": ("D3", "Spline FuncCode W4A4"),
    "gram_fp32": ("G1", "GRAM FP32 dense"),
    "gram_lsq_w4a4": ("G2", "GRAM LSQ W4A4"),
    "gram_funccode_w4a4": ("G3", "GRAM FuncCode W4A4"),
    "kagnconv_fp32": ("K1", "KAGN-Conv FP32 dense"),
    "kagnconv_lsq_w4a4": ("K2", "KAGN-Conv LSQ W4A4"),
    "kagnconv_funccode_w4a4": ("K3", "KAGN-Conv FuncCode W4A4"),
}
GOLDEN_DIRS = {d: d for d in DESIGN_LABELS}
DESIGN_ORDER = list(DESIGN_LABELS)


def accuracy_for(design: str, runs: Path, golden_root: Path) -> float | None:
    """Full-10k-test accuracy: emulator accuracy from the golden manifest
    (the L1-verified number); falls back to the training summaries."""
    manifest = golden_root / GOLDEN_DIRS[design] / "manifest.json"
    if manifest.exists():
        m = json.loads(manifest.read_text())
        l1 = m.get("l1", {})
        acc = l1.get("emulator_acc", l1.get("numpy_acc"))
        if acc is not None:
            return float(acc)
    fallback = {
        "fp32": (runs / "branch_k32_s32_b16" / "summary.csv", "dense_fp32"),
        "lsq_w4a4": (runs / "d2_lsq_w4a4" / "summary.csv", "lsq_w4a4_qat"),
        "funccode_w4a4": (runs / "d3_funccode_w4a4" / "summary.csv",
                          "funccode_w4a4_ft"),
        "gram_fp32": (runs / "gram_hw_source" / "gram" / "summary.csv",
                      "dense_fp32"),
        "gram_lsq_w4a4": (runs / "g2_gram_lsq_w4a4" / "summary.csv",
                          "gram_lsq_w4a4_qat"),
        "gram_funccode_w4a4": (runs / "g3_gram_funccode_w4a4" /
                               "summary.csv", "gram_funccode_w4a4_ft"),
        "kagnconv_fp32": (runs / "kagnconv_hw_source" / "summary.csv",
                          "dense_fp32"),
        "kagnconv_lsq_w4a4": (runs / "k2_kagnconv_lsq_w4a4" / "summary.csv",
                              "kagnconv_lsq_w4a4_qat"),
        "kagnconv_funccode_w4a4": (runs / "k3_kagnconv_funccode_w4a4" /
                                   "summary.csv",
                                   "kagnconv_funccode_w4a4_ft"),
    }.get(design)
    if fallback is None:
        return None
    path, stage = fallback
    if path.exists():
        df = pd.read_csv(path)
        row = df[df["stage"] == stage]
        if len(row):
            return float(row["test_acc"].iloc[0])
    return None


def param_storage_kib(design: str, runs: Path) -> float | None:
    """Analytical parameter memory (Section 3 model / analysis.storage)."""
    if design == "fp32":
        return 50816 * 9 * 32 / 8 / 1024
    if design == "gram_fp32":
        return 254080 * 32 / 8 / 1024
    path = {
        "lsq_w4a4": runs / "d2_lsq_w4a4" / "summary.csv",
        "funccode_w4a4": runs / "d3_funccode_w4a4" / "summary.csv",
        "gram_lsq_w4a4": runs / "g2_gram_lsq_w4a4" / "summary.csv",
        "gram_funccode_w4a4": runs / "g3_gram_funccode_w4a4" / "summary.csv",
        "kagnconv_fp32": runs / "kagnconv_hw_source" / "summary.csv",
        "kagnconv_lsq_w4a4": runs / "k2_kagnconv_lsq_w4a4" / "summary.csv",
        "kagnconv_funccode_w4a4": runs / "k3_kagnconv_funccode_w4a4" /
                                  "summary.csv",
    }.get(design)
    if path is not None and path.exists():
        df = pd.read_csv(path)
        col = ("dense_storage_kib" if design == "kagnconv_fp32"
               and "dense_storage_kib" in df.columns else "storage_kib")
        return float(df[col].iloc[0])
    return None


def fmt(v, spec=".1f", dash="--"):
    if v is None or pd.isna(v):
        return dash
    return format(v, spec)


def build_rows(report: pd.DataFrame, stage: str, runs: Path,
               golden_root: Path) -> list[dict]:
    rows = []
    for design in DESIGN_ORDER:
        r = report[(report["design"] == design) & (report["stage"] == stage)]
        if not len(r):
            continue
        r = r.iloc[0]
        did, label = DESIGN_LABELS[design]
        rows.append({
            "id": did, "label": label,
            "acc": accuracy_for(design, runs, golden_root),
            "param_kib": param_storage_kib(design, runs),
            "lut": r.get("lut"), "ff": r.get("ff"), "dsp": r.get("dsp"),
            "bram18": r.get("bram18"), "uram": r.get("uram"),
            "clock_ns": r.get("clock_ns"),
            "latency_us": r.get("latency_us"),
            "images_per_s": r.get("images_per_s"),
            "source": r.get("source_report"),
        })
    return rows


STAGE_LABELS = {
    "post_synthesis": "post-synthesis, xczu9eg",
    "post_implementation": "post-implementation, xczu9eg",
    "post_implementation_zu7ev":
        "post-implementation, xczu7ev; all designs whose parameters fit "
        "this licensed part -- the spline FP32 baseline (D1) does not "
        "(HW_FAIRNESS.md)",
}


def to_latex(rows: list[dict], stage: str) -> str:
    stage_label = STAGE_LABELS.get(stage, stage)
    lines = [
        f"% Auto-generated by tools/make_hw_comparison_table.py ({stage_label})",
        "% Sources:",
    ]
    for r in rows:
        lines.append(f"%   {r['id']}: {r['source']}")
    lines += [
        r"\begin{table}[t]",
        r"\centering",
        (r"\caption{FPGA comparison on MNIST "
         f"(150\\,MHz target, batch=1; {stage_label}). "
         r"Parameter memory is the exact analytical model; accuracy is the "
         r"full-test-set emulator accuracy (L1-verified).}"),
        r"\label{tab:hw_comparison}",
        r"\begin{tabular}{llrrrrrrrr}",
        r"\toprule",
        (r"ID & Design & Acc.\,(\%) & Params\,(KiB) & LUT & FF & DSP & "
         r"BRAM18 & Lat.\,($\mu$s) & Img/s \\"),
        r"\midrule",
    ]
    for r in rows:
        lines.append(
            f"{r['id']} & {r['label']} & {fmt(r['acc'], '.2f')} & "
            f"{fmt(r['param_kib'], '.1f')} & {fmt(r['lut'], '.0f')} & "
            f"{fmt(r['ff'], '.0f')} & {fmt(r['dsp'], '.0f')} & "
            f"{fmt(r['bram18'], '.0f')} & {fmt(r['latency_us'], '.1f')} & "
            f"{fmt(r['images_per_s'], '.0f')} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]
    return "\n".join(lines)


def to_markdown(rows: list[dict], stage: str) -> str:
    stage_label = STAGE_LABELS.get(stage, stage)
    lines = [
        f"### Hardware comparison ({stage_label})",
        "",
        "| ID | Design | Acc (%) | Params (KiB) | LUT | FF | DSP | BRAM18 |"
        " Latency (us) | Img/s |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        lines.append(
            f"| {r['id']} | {r['label']} | {fmt(r['acc'], '.2f')} | "
            f"{fmt(r['param_kib'], '.1f')} | {fmt(r['lut'], '.0f')} | "
            f"{fmt(r['ff'], '.0f')} | {fmt(r['dsp'], '.0f')} | "
            f"{fmt(r['bram18'], '.0f')} | {fmt(r['latency_us'], '.1f')} | "
            f"{fmt(r['images_per_s'], '.0f')} |")
    lines += ["", "Sources:"]
    lines += [f"- {r['id']}: `{r['source']}`" for r in rows]
    lines.append("")
    return "\n".join(lines)


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--csv", type=str,
                   default="runs_hw/tables/hw_report_summary.csv")
    p.add_argument("--runs-dir", type=str, default="runs_hw")
    p.add_argument("--golden-root", type=str, default="hw/golden")
    p.add_argument("--out-tex", type=str,
                   default="runs_hw/tables/hw_comparison.tex")
    p.add_argument("--out-md", type=str, default="docs/HW_COMPARISON.md")
    args = p.parse_args(argv)
    report = pd.read_csv(args.csv)
    runs = Path(args.runs_dir)
    golden_root = Path(args.golden_root)

    tex_parts, md_parts = [], []
    for stage in ["post_synthesis", "post_implementation",
                  "post_implementation_zu7ev"]:
        rows = build_rows(report, stage, runs, golden_root)
        if rows:
            tex_parts.append(to_latex(rows, stage))
            md_parts.append(to_markdown(rows, stage))
    if not tex_parts:
        raise SystemExit("no rows found in " + args.csv)

    out_tex = Path(args.out_tex)
    out_tex.parent.mkdir(parents=True, exist_ok=True)
    out_tex.write_text("\n".join(tex_parts), encoding="utf-8")
    out_md = Path(args.out_md)
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text(
        "# FuncCode-KAN hardware comparison (auto-generated)\n\n"
        "Regenerate with `bash scripts/hw/13_reports.sh`.\n\n"
        + "\n".join(md_parts), encoding="utf-8")
    print("Wrote", out_tex, "and", out_md)


if __name__ == "__main__":
    main()
