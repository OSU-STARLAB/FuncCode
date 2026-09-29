"""Parse archived Vivado HLS 2019 reports into a tidy CSV.

Reads, per design, from hw/results/<design>/ (archived by
scripts/hw/12_run_hls.sh):
  syn/csynth.xml            post-csynth area/latency/timing estimates
  syn/kan_top_csynth.rpt    loop II table (text)
  impl/verilog/kan_top_export.rpt  post-implementation resources + timing
                                   (only if export_design -flow impl ran)

Emits one row per (design, stage) with: LUT, FF, DSP, BRAM18, URAM,
latency min/max cycles, clock target/estimate, derived latency_us and
images_per_s (batch=1). Post-synthesis and post-implementation rows are
labelled via the `stage` column and must never be mixed in a comparison
(docs/HW_FAIRNESS.md).

Usage:
  python tools/parse_hls_reports.py --results-root hw/results \
      --out runs_hw/tables/hw_report_summary.csv
"""

from __future__ import annotations

import argparse
import re
import xml.etree.ElementTree as ET
from pathlib import Path

import pandas as pd

DESIGNS = ["fp32", "lsq_w4a4", "funccode_w4a4",
           "gram_fp32", "gram_lsq_w4a4", "gram_funccode_w4a4",
           "kagnconv_fp32", "kagnconv_lsq_w4a4", "kagnconv_funccode_w4a4"]


def _to_num(text):
    if text is None:
        return None
    text = text.strip()
    if text in ("", "?", "N/A", "undef"):
        return None
    try:
        return int(text)
    except ValueError:
        try:
            return float(text)
        except ValueError:
            return None


def parse_csynth_xml(path: Path) -> dict:
    root = ET.parse(path).getroot()

    def find(p):
        el = root.find(p)
        return el.text if el is not None else None

    res = root.find("AreaEstimates/Resources")
    resources = {c.tag: _to_num(c.text) for c in res} if res is not None else {}
    row = {
        "part": find("UserAssignments/Part"),
        "clock_target_ns": _to_num(find("UserAssignments/TargetClockPeriod")),
        "clock_ns": _to_num(
            find("PerformanceEstimates/SummaryOfTimingAnalysis/"
                 "EstimatedClockPeriod")),
        "latency_cycles_min": _to_num(
            find("PerformanceEstimates/SummaryOfOverallLatency/"
                 "Best-caseLatency")),
        "latency_cycles_max": _to_num(
            find("PerformanceEstimates/SummaryOfOverallLatency/"
                 "Worst-caseLatency")),
        "lut": resources.get("LUT"),
        "ff": resources.get("FF"),
        "dsp": resources.get("DSP48E", resources.get("DSP")),
        "bram18": resources.get("BRAM_18K"),
        "uram": resources.get("URAM"),
    }
    return row


def parse_loop_iis(rpt_path: Path) -> str:
    """Achieved II of every pipelined loop from the csynth .rpt loop table.
    Rows look like: |- OUT0  | ... | 1| 1| ... with the loop name first."""
    if not rpt_path.exists():
        return ""
    iis = []
    for line in rpt_path.read_text(errors="replace").splitlines():
        # |- IN0_OUT0 | 50178| 50178| 4| 1| 1| 50176| yes |
        m = re.match(r"\s*\|-+\s*([A-Za-z0-9_]+)\s*\|(.*)\|\s*$", line)
        if not m:
            continue
        cells = [c.strip() for c in m.group(2).split("|")]
        # min, max, iteration latency, II achieved, II target, trip, pipelined
        if len(cells) == 7 and cells[6].lower() == "yes":
            iis.append(f"{m.group(1)}:II={cells[3]}")
    return ";".join(iis)


def parse_vivado_impl(util_rpt: Path, timing_rpt: Path,
                      target_ns: float) -> dict:
    """Post-implementation numbers from the plain-Vivado OOC route reports
    (hw/tcl/impl_eval.tcl; export_design is unusable on 2019.1 due to the
    Y2K22 IP-packager bug)."""
    util = util_rpt.read_text(errors="replace")

    def grab(pattern):
        m = re.search(pattern, util)
        return _to_num(m.group(1)) if m else None

    bram36 = grab(r"Block RAM Tile\s*\|\s*([\d.]+)")
    row = {
        "lut": grab(r"CLB LUTs\*?\s*\|\s*(\d+)"),
        "ff": grab(r"CLB Registers\s*\|\s*(\d+)"),
        "dsp": grab(r"DSPs\s*\|\s*(\d+)"),
        "bram18": int(round(bram36 * 2)) if bram36 is not None else None,
        "uram": grab(r"URAM\s*\|\s*(\d+)"),
        "clock_target_ns": target_ns,
    }
    if timing_rpt.exists():
        timing = timing_rpt.read_text(errors="replace")
        m = re.search(r"WNS\(ns\).*?\n[\s-]*\n?\s*(-?[\d.]+)", timing,
                      re.DOTALL)
        if m:
            wns = float(m.group(1))
            row["wns_ns"] = wns
            row["clock_ns"] = target_ns - wns  # achieved period
    return row


def derive(row: dict) -> dict:
    clk = row.get("clock_ns") or row.get("clock_target_ns")
    lat = row.get("latency_cycles_max")
    if clk and lat:
        # report clock period only counts if it meets target; latency in us
        # uses the TARGET period (that is what the deployed clock would be)
        target = row.get("clock_target_ns") or clk
        row["latency_us"] = lat * target * 1e-3
        row["images_per_s"] = 1e6 / row["latency_us"]
    return row


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--results-root", type=str, default="hw/results")
    p.add_argument("--out", type=str,
                   default="runs_hw/tables/hw_report_summary.csv")
    args = p.parse_args(argv)
    root = Path(args.results_root)
    rows = []
    for design in DESIGNS:
        syn_xml = root / design / "syn" / "csynth.xml"
        if syn_xml.exists():
            row = {"design": design, "stage": "post_synthesis",
                   **parse_csynth_xml(syn_xml)}
            # loop tables live in the per-layer submodule reports
            row["loop_iis"] = ";".join(filter(None, (
                parse_loop_iis(p)
                for p in sorted((root / design / "syn").glob("*_csynth.rpt")))))
            row["source_report"] = str(syn_xml)
            rows.append(derive(row))
        else:
            print(f"[parse] WARNING: {syn_xml} missing; skipping {design} syn")
        util_rpt = root / design / "impl" / "utilization_route.rpt"
        timing_rpt = root / design / "impl" / "timing_route.rpt"
        if util_rpt.exists():
            target = 6.67
            if syn_xml.exists():
                target = parse_csynth_xml(syn_xml)["clock_target_ns"] or 6.67
            row = {"design": design, "stage": "post_implementation",
                   **parse_vivado_impl(util_rpt, timing_rpt, target)}
            # cycles come from csynth (RTL latency unchanged by P&R)
            if syn_xml.exists():
                syn = parse_csynth_xml(syn_xml)
                row["latency_cycles_min"] = syn["latency_cycles_min"]
                row["latency_cycles_max"] = syn["latency_cycles_max"]
            row["source_report"] = str(util_rpt)
            rows.append(derive(row))
        # secondary post-impl on the licensed ZU7EV (D2/D3 only; D1's FP32
        # ROMs do not fit that part -- see docs/HW_FAIRNESS.md)
        alt_util = root / design / "impl_alt" / "utilization_route.rpt"
        alt_timing = root / design / "impl_alt" / "timing_route.rpt"
        alt_xml = root / design / "syn_alt" / "csynth.xml"
        if alt_util.exists():
            target = 6.67
            row = {"design": design, "stage": "post_implementation_zu7ev",
                   **parse_vivado_impl(alt_util, alt_timing, target)}
            if alt_xml.exists():
                syn = parse_csynth_xml(alt_xml)
                row["part"] = syn["part"]
                row["latency_cycles_min"] = syn["latency_cycles_min"]
                row["latency_cycles_max"] = syn["latency_cycles_max"]
            row["source_report"] = str(alt_util)
            rows.append(derive(row))
    if not rows:
        raise SystemExit("no reports found under " + str(root))
    df = pd.DataFrame(rows)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print(df.to_string(index=False))
    print("\nSaved to:", out)


if __name__ == "__main__":
    main()
