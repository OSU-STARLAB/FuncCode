#!/usr/bin/env python
"""Generate the job plan consumed by ``tools/gpu_queue.py``.

The grid is split so that dense training is a *separate job* from every
compression arm, with the arms declaring ``depends_on`` the dense job and
passing ``--dense-ckpt``. Three reasons this matters:

  * the 8-layer CIFAR-100 dense stage is ~4 GPU-hours; a crash in one
    compression arm must not cost it
  * once dense finishes, 10+ independent arms can saturate both cards, which is
    where the 2-GPU speedup actually comes from
  * arms can be re-run individually after a bug fix without retraining

Tiers
-----
tier1  the main table: both datasets, seed 42, all methods and baselines
tier2  seeds 123 and 2026 for error bars
tier3  ablations (metric, normalisation, skip-first/head, codebook bits)

  python tools/make_conv_plan.py --tier tier1 --out scripts/paper/plans/cifar_tier1.json
  python tools/gpu_queue.py --plan scripts/paper/plans/cifar_tier1.json --dry-run
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

PY = "python"
MOD = "-m", "funcodekan.experiments.conv_cifar"

# (dataset, preset, dense epochs, mixup, est minutes for dense, est per-arm minutes)
#
# est_minutes only drives the ETA readout and longest-first ordering within a
# priority tier. These are calibrated to measured step times (16 ms/step
# for the 4-layer net, 39 ms/step for the 8-layer, 351 steps/epoch) with
# headroom for EMA, per-epoch validation, and 2-slots-per-GPU contention.
#
# A key is a TRACK, not a dataset -- "dataset" says which data to load, so the
# same dataset can appear under several backbones/recipes without run-name
# collisions. cifar10_8l exists because the documented 4-layer CIFAR-10 preset
# tops out at 64.26% (measured): it shares dropout=0.25/0.5 with the 8-layer
# preset, which is sized for a 61x larger model and drives the small net into
# severe underfitting. The 8-layer backbone reaches 90.85% on CIFAR-10.
CONFIGS = {
    "cifar10": dict(dataset="cifar10", preset="kagn_simple_cifar10",
                    epochs=200, mixup=0.0,
                    dense_min=45, arm_min=10, iso_min=20, lr=1e-3, batch=128),
    "cifar100": dict(dataset="cifar100", preset="kagn_simple_cifar100_8_layer_v2",
                     epochs=200, mixup=0.2,
                     dense_min=115, arm_min=25, iso_min=45, lr=1e-3, batch=128),
    # 8-layer CIFAR-10, stock recipe (mixup 0.0) -- the arm this reports.
    "cifar10_8l": dict(dataset="cifar10", preset="kagn_simple_cifar10_8_layer_v2",
                       epochs=200, mixup=0.0,
                       dense_min=115, arm_min=25, iso_min=45, lr=1e-3, batch=128),
    # Same backbone with CIFAR-100's mixup. A post-hoc recipe change made after
    # seeing a 0.15pp miss, so it is kept as a separate, clearly-labelled track
    # rather than substituted into the primary one.
    "cifar10_8l_mixup": dict(dataset="cifar10", preset="kagn_simple_cifar10_8_layer_v2",
                             epochs=200, mixup=0.2,
                             dense_min=115, arm_min=25, iso_min=45, lr=1e-3, batch=128),
}

CLUSTERS = [8, 16, 32, 64, 256]
FT_EPOCHS = 30
# One iso-storage dense net per FuncCode index width: K=16 -> 4 bits/edge,
# K=32 -> 5, K=256 -> 8. Kept as separate jobs so they run concurrently.
ISO_CLUSTERS = [16, 32, 256]
# Tuned for 4 concurrent jobs on a 16-core host.
NUM_WORKERS = 3


def base_cmd(ds, cfg, seed, run_name, out_dir, data_root, extra):
    c = [PY, *MOD,
         "--dataset", cfg.get("dataset", ds), "--preset", cfg["preset"], "--seed", str(seed),
         "--data-root", data_root, "--out-dir", out_dir, "--run-name", run_name,
         "--batch-size", str(cfg["batch"]), "--lr", str(cfg["lr"]),
         "--epochs", str(cfg["epochs"]), "--mixup", str(cfg["mixup"]),
         "--finetune-epochs", str(FT_EPOCHS), "--num-workers", str(NUM_WORKERS)]
    return c + extra


def build(tier, seeds, datasets, out_dir, data_root, stages_only=None,
          methods=("function", "branch"), pq_clusters=(16, 32)):
    jobs = []
    for ds in datasets:
        cfg = CONFIGS[ds]
        for seed in seeds:
            stem = f"{ds}_s{seed}"
            dense_run = f"{stem}_dense"
            ckpt = f"{out_dir}/{dense_run}/dense.pt"

            # ---- dense backbone (priority 0: unblocks everything) --------
            jobs.append(dict(
                name=f"dense_{stem}", priority=0, est_minutes=cfg["dense_min"],
                cmd=base_cmd(ds, cfg, seed, dense_run, out_dir, data_root,
                             ["--stages", "dense"])))

            # Dense-only plans (extra seeds for error bars, recipe variants)
            # stop here -- no compression arms.
            if stages_only == "dense":
                continue

            # ---- FuncCode arms, one job per (method, K) ------------------
            for method in methods:
                for K in CLUSTERS:
                    name = f"fc_{method}_K{K}_{stem}"
                    jobs.append(dict(
                        name=name, priority=10, est_minutes=cfg["arm_min"],
                        depends_on=f"dense_{stem}",
                        cmd=base_cmd(ds, cfg, seed, name, out_dir, data_root,
                                     ["--stages", "funccode", "--methods", method,
                                      "--clusters-list", str(K),
                                      "--codebook-bits", "8", "4",
                                      "--dense-ckpt", ckpt, "--require-ckpt"])))

            # ---- baselines ----------------------------------------------
            # PTQ is inference-only and cheap: one job for every bit-width.
            jobs.append(dict(
                name=f"bl_uniform_{stem}", priority=20, est_minutes=5,
                depends_on=f"dense_{stem}",
                cmd=base_cmd(ds, cfg, seed, f"bl_uniform_{stem}", out_dir, data_root,
                             ["--stages", "baselines", "--baselines", "uniform",
                              "--uniform-bits", "8", "4", "3", "2",
                              "--dense-ckpt", ckpt, "--require-ckpt"])))

            for b in (4, 2):
                jobs.append(dict(
                    name=f"bl_lsq_w{b}_{stem}", priority=20, est_minutes=cfg["arm_min"],
                    depends_on=f"dense_{stem}",
                    cmd=base_cmd(ds, cfg, seed, f"bl_lsq_w{b}_{stem}", out_dir, data_root,
                                 ["--stages", "baselines", "--baselines", "lsq",
                                  "--lsq-bits", str(b),
                                  "--dense-ckpt", ckpt, "--require-ckpt"])))

            for K in pq_clusters:
                jobs.append(dict(
                    name=f"bl_pq_K{K}_{stem}", priority=25, est_minutes=cfg["arm_min"],
                    depends_on=f"dense_{stem}",
                    cmd=base_cmd(ds, cfg, seed, f"bl_pq_K{K}_{stem}", out_dir, data_root,
                                 ["--stages", "baselines", "--baselines", "pq",
                                  "--pq-subvectors", "2", "--clusters-list", str(K),
                                  "--dense-ckpt", ckpt, "--require-ckpt"])))

            jobs.append(dict(
                name=f"bl_prune_{stem}", priority=25, est_minutes=cfg["arm_min"] * 2,
                depends_on=f"dense_{stem}",
                cmd=base_cmd(ds, cfg, seed, f"bl_prune_{stem}", out_dir, data_root,
                             ["--stages", "baselines", "--baselines", "prune",
                              "--prune-sparsity", "0.9", "0.95",
                              "--dense-ckpt", ckpt, "--require-ckpt"])))

            # iso-storage dense: one narrow net trained from scratch per storage
            # budget, so each is priced like a (small) dense job rather than an
            # arm. Split one-job-per-budget: as a single job these three trained
            # serially and formed the tail of the schedule.
            #
            # --dense-ckpt is passed even though nothing here is compressed. The
            # driver always instantiates the full-size backbone to read n_edges
            # and the FP32 storage reference; without a checkpoint it retrains
            # it from scratch, which cost a redundant full dense run per dataset
            # (~2 GPU-hours on CIFAR-100) and produced nothing this job uses.
            for K in ISO_CLUSTERS:
                jobs.append(dict(
                    name=f"bl_iso_K{K}_{stem}", priority=30,
                    est_minutes=cfg["iso_min"],
                    depends_on=f"dense_{stem}",
                    cmd=base_cmd(ds, cfg, seed, f"bl_iso_K{K}_{stem}", out_dir, data_root,
                                 ["--stages", "baselines", "--baselines", "iso",
                                  "--clusters-list", str(K),
                                  "--dense-ckpt", ckpt, "--require-ckpt"])))

    # ---- tier 3 ablations, seed 42 / cifar100 only ------------------------
    if tier == "tier3":
        ds, cfg, seed = "cifar100", CONFIGS["cifar100"], 42
        ckpt = f"{out_dir}/{ds}_s{seed}_dense/dense.pt"
        ablations = [
            ("abl_metric_coefficient", ["--methods", "coefficient", "--clusters-list", "32"]),
            ("abl_signature_normalize", ["--methods", "function", "--clusters-list", "32",
                                         "--metric", "signature", "--signature-normalize"]),
            ("abl_skip_first_head", ["--methods", "function", "--clusters-list", "32",
                                     "--skip-first", "--skip-head"]),
            ("abl_codebook_bits", ["--methods", "function", "--clusters-list", "32",
                                   "--codebook-bits", "8", "6", "4", "3", "2"]),
        ]
        for name, extra in ablations:
            jobs.append(dict(
                name=f"{name}_{ds}_s{seed}", priority=40, est_minutes=cfg["arm_min"],
                depends_on=f"dense_{ds}_s{seed}",
                cmd=base_cmd(ds, cfg, seed, f"{name}_{ds}_s{seed}", out_dir, data_root,
                             ["--stages", "funccode", *extra,
                              "--dense-ckpt", ckpt, "--require-ckpt"])))

    return {"tier": tier, "seeds": seeds, "datasets": datasets, "jobs": jobs}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tier", default="tier1", choices=["tier1", "tier2", "tier3"])
    ap.add_argument("--datasets", nargs="+", default=["cifar10", "cifar100"],
                    help=f"track keys: {sorted(CONFIGS)}")
    ap.add_argument("--seeds", nargs="+", type=int, default=None,
                    help="override the tier's default seeds")
    ap.add_argument("--dense-only", action="store_true",
                    help="emit only dense jobs (extra seeds, recipe variants)")
    ap.add_argument("--methods", nargs="+", default=["function", "branch"],
                    help="FuncCode methods to emit arms for; drop 'branch' to "
                         "skip an ablation that tier1 showed is dominated at "
                         "every comparable budget on both datasets")
    ap.add_argument("--pq-clusters", nargs="+", type=int, default=[16, 32],
                    help="K for the product-quantisation baseline. m=2 subvectors "
                         "means 2*log2(K) bits/edge, so K=4 lands at 4 bits/edge "
                         "-- the band where FuncCode is otherwise unopposed.")
    ap.add_argument("--out", required=True)
    ap.add_argument("--out-dir", default="runs_conv_cifar")
    ap.add_argument("--data-root", default="./data")
    args = ap.parse_args()

    seeds = args.seeds or {"tier1": [42], "tier2": [123, 2026], "tier3": [42]}[args.tier]
    plan = build(args.tier, seeds, args.datasets, args.out_dir, args.data_root,
                 stages_only="dense" if args.dense_only else None,
                 methods=tuple(args.methods), pq_clusters=tuple(args.pq_clusters))

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(plan, indent=2))

    total = sum(j["est_minutes"] for j in plan["jobs"])
    print(f"wrote {out}: {len(plan['jobs'])} jobs, {total} GPU-minutes "
          f"(~{total/2/60:.1f} h wall on 2 GPUs)")


if __name__ == "__main__":
    main()
